"""
Multimodal Fusion Backbone for Tri-SACTA.

Architecture overview:
1. **ModalityProjection** - per-modality linear + LayerNorm -> fusion_dim.
2. **LowRankAdapter** - lightweight adaptation via down/up projection (fusion_dim -> rank -> fusion_dim).
3. **MaskedModalityTransformerLayer** - multi-head attention across modality tokens (B, M, D)
   with explicit key-padding mask (AUD-02) ensuring missing modalities receive zero attention.
4. **MultimodalFusionModel** - end-to-end model with pre-fusion unimodal heads (AUD-06)
   and stable, non-underflowing reliability fusion (AUD-01, AUD-03).

Key Design Goals:
- Total trainable parameters < 2 M.
- Fully compatible with PyTorch AMP (torch.amp / torch.cuda.amp).
- Pure mathematical mask invariance: missing modalities contribute 0 to attention and fusion.
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

MODALITY_ORDER: List[str] = ["visual", "audio", "text", "prosody"]


class ModalityProjection(nn.Module):
    """Project raw features of a single modality to the shared fusion dim."""

    def __init__(self, input_dim: int, fusion_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, fusion_dim)
        self.norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.norm(self.proj(self.input_norm(x))))


class LowRankAdapter(nn.Module):
    """Low-rank bottleneck adapter with residual connection."""

    def __init__(self, dim: int, rank: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Near-identity initialization
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.up(F.gelu(self.down(x))))


class MaskedModalityTransformerLayer(nn.Module):
    """Multi-head cross-attention across modality tokens with strict key-padding mask.

    Operates on sequence of M modality tokens: (B, M, D).
    Mask shape: (B, M) bool.
    Ensures missing modalities receive exactly 0 attention weight and produce 0 output.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, M, D) modality tokens.
            mask: (B, M) bool mask (True if modality is present, False if missing).
        """
        B, M, D = tokens.shape
        H, d = self.num_heads, self.head_dim

        # 1. Self/Cross Attention across modality tokens
        residual = tokens
        normed = self.norm1(tokens)

        Q = self.q_proj(normed).view(B, M, H, d).transpose(1, 2)  # (B, H, M, d)
        K = self.k_proj(normed).view(B, M, H, d).transpose(1, 2)  # (B, H, M, d)
        V = self.v_proj(normed).view(B, M, H, d).transpose(1, 2)  # (B, H, M, d)

        scores = (Q @ K.transpose(-2, -1)) * self.scale  # (B, H, M, M)

        # Key padding mask: mask out keys that are missing (dim=-1)
        key_mask = mask.view(B, 1, 1, M).bool()  # (B, 1, 1, M)
        scores = scores.masked_fill(~key_mask, -1e4)

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        # Context output
        context = (attn @ V).transpose(1, 2).reshape(B, M, D)  # (B, M, D)
        tokens = residual + self.out_proj(context)

        # 2. Feed-forward network
        tokens = tokens + self.ffn(self.norm2(tokens))

        # Enforce zero representation for missing modalities
        query_mask = mask.view(B, M, 1).float()
        tokens = tokens * query_mask

        return tokens


class MultimodalFusionModel(nn.Module):
    """Multimodal Fusion Model with pre-fusion unimodal heads and reliability-weighted fusion."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config

        model_cfg = config.get("model", {})
        feat_cfg = config.get("features", {})

        self.fusion_dim = model_cfg.get("fusion_dim", 256)
        self.num_heads = model_cfg.get("num_heads", 4)
        self.num_layers = model_cfg.get("num_layers", 2)
        self.adapter_rank = model_cfg.get("adapter_rank", 8)
        self.dropout = model_cfg.get("dropout", 0.1)
        self.num_classes = model_cfg.get("num_classes", 7)

        feature_dims = {
            "visual": feat_cfg.get("visual_dim", 512),
            "audio": feat_cfg.get("audio_dim", 768),
            "text": feat_cfg.get("text_dim", 768),
            "prosody": feat_cfg.get("prosody_dim", 88),
        }

        # 1. Per-modality projection layers
        self.projections = nn.ModuleDict({
            m: ModalityProjection(feature_dims[m], self.fusion_dim, self.dropout)
            for m in MODALITY_ORDER
        })

        # 2. Per-modality low-rank adapters
        self.adapters = nn.ModuleDict({
            m: LowRankAdapter(self.fusion_dim, self.adapter_rank, self.dropout)
            for m in MODALITY_ORDER
        })

        # 3. Pre-fusion unimodal auxiliary classifiers (AUD-06)
        self.modality_classifiers = nn.ModuleDict({
            m: nn.Sequential(
                nn.LayerNorm(self.fusion_dim),
                nn.Linear(self.fusion_dim, self.num_classes),
            )
            for m in MODALITY_ORDER
        })

        # 4. Multi-token modality transformer layers
        self.transformer_layers = nn.ModuleList([
            MaskedModalityTransformerLayer(self.fusion_dim, self.num_heads, self.dropout)
            for _ in range(self.num_layers)
        ])

        # 5. Final fused classifier
        self.classifier_norm = nn.LayerNorm(self.fusion_dim)
        self.classifier = nn.Linear(self.fusion_dim, self.num_classes)

        # Telemetry counters for fallback / isolation policy (V5-11, Task 6)
        self.fallback_count: int = 0
        self.abstain_count: int = 0

        # Learned modality importance prior (Task 3)
        self.modality_gate = nn.Parameter(torch.zeros(len(MODALITY_ORDER)))

        total, trainable = self.get_total_params()
        logger.info(f"MultimodalFusionModel initialized: {total:,} total params, {trainable:,} trainable ({trainable/1e6:.2f} M)")

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        modality_mask: Optional[Dict[str, torch.Tensor]] = None,
        reliability_weights: Optional[Dict[str, torch.Tensor]] = None,
        effective_mask: Optional[Dict[str, torch.Tensor]] = None,
        fallback_to_presence_on_all_rejected: bool = False,
    ) -> Dict[str, Any]:
        """Forward pass with strict mask gating and hard effective mask isolation (R3-07, V5-11)."""
        device = next(self.parameters()).device
        # Sanitize input features against NaN / Inf values (V5-11)
        clean_features = {
            m: torch.nan_to_num(v.to(device), nan=0.0, posinf=1e4, neginf=-1e4)
            for m, v in features.items()
        }
        present_mods = [m for m in MODALITY_ORDER if m in clean_features]
        B = clean_features[present_mods[0]].size(0)

        # 1. Project and adapt each available modality
        projected: Dict[str, torch.Tensor] = {}
        adapted: Dict[str, torch.Tensor] = {}
        per_modality_logits: Dict[str, torch.Tensor] = {}

        for m in MODALITY_ORDER:
            if m in clean_features:
                x = clean_features[m]
                proj = self.projections[m](x)
                ad = self.adapters[m](proj)
                projected[m] = proj
                adapted[m] = ad
                # Pre-fusion unimodal heads (AUD-06)
                per_modality_logits[m] = self.modality_classifiers[m](ad)
            else:
                projected[m] = torch.zeros(B, self.fusion_dim, device=device)
                adapted[m] = torch.zeros(B, self.fusion_dim, device=device)
                per_modality_logits[m] = torch.zeros(B, self.num_classes, device=device)

        # 2. Build modality token sequence (B, M, D) and attention mask (B, M)
        token_list = []
        presence_mask_list = []
        attn_mask_list = []
        active_mask_source = effective_mask if effective_mask is not None else modality_mask

        for m in MODALITY_ORDER:
            token_list.append(adapted[m])
            if modality_mask is not None and m in modality_mask:
                p_mask = modality_mask[m].to(device).bool().view(B)
            else:
                p_mask = torch.ones(B, dtype=torch.bool, device=device) if m in clean_features else torch.zeros(B, dtype=torch.bool, device=device)
            presence_mask_list.append(p_mask)

            # Hard gating mask (R3-07): isolates corrupted and missing tokens from attention
            if active_mask_source is not None and m in active_mask_source:
                a_mask = active_mask_source[m].to(device).bool().view(B)
            else:
                a_mask = p_mask
            attn_mask_list.append(a_mask)

        tokens = torch.stack(token_list, dim=1)           # (B, M, D)
        presence_mask = torch.stack(presence_mask_list, dim=1) # (B, M) bool
        attn_mask = torch.stack(attn_mask_list, dim=1)         # (B, M) bool

        # V5-11: Explicit Fallback and Abstain Policy
        # When all modalities are rejected by effective_mask for a sample:
        all_attn_masked = (~attn_mask).all(dim=1, keepdim=True)
        fallback_applied = torch.zeros_like(all_attn_masked)
        if all_attn_masked.any():
            if fallback_to_presence_on_all_rejected:
                # Explicit fallback: unmask physical presence and increment fallback counter
                attn_mask = torch.where(all_attn_masked, presence_mask, attn_mask)
                fallback_applied = all_attn_masked.clone()
                self.fallback_count += int(all_attn_masked.sum().item())
            else:
                # Strict channel isolation: do not leak corrupted channels; sample is abstained
                self.abstain_count += int(all_attn_masked.sum().item())

        # Zero out inactive tokens initially
        tokens = tokens * attn_mask.unsqueeze(-1).float()

        # 3. Cross-modal transformer layers with effective attention mask
        for layer in self.transformer_layers:
            tokens = layer(tokens, attn_mask)

        # 4. Safe reliability-weighted pooling with learned modality prior (AUD-01, AUD-03, R3-07, Task 3)
        gate_priors = F.softmax(self.modality_gate, dim=0)
        raw_weights_list = []
        for i, m in enumerate(MODALITY_ORDER):
            if reliability_weights is not None and m in reliability_weights:
                w = reliability_weights[m]
                if not isinstance(w, torch.Tensor):
                    w = torch.tensor(w, device=device).expand(B)
                else:
                    w = w.to(device).float().view(B)
            else:
                w = torch.ones(B, device=device)

            # Gate by effective attention mask and learned modality prior
            w = w * attn_mask[:, i].float() * gate_priors[i]
            raw_weights_list.append(w)

        raw_weights = torch.stack(raw_weights_list, dim=1)  # (B, M)
        weight_sum = raw_weights.sum(dim=1, keepdim=True)    # (B, 1)

        has_weight = weight_sum > 1e-7
        avail_count = attn_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        uniform_weights = attn_mask.float() / avail_count

        norm_weights = torch.where(has_weight, raw_weights / weight_sum.clamp(min=1e-7), uniform_weights)

        # Fused vector (B, D)
        fused = (norm_weights.unsqueeze(-1) * tokens).sum(dim=1)  # (B, D)

        # 5. Final classification
        logits = self.classifier(self.classifier_norm(fused))

        per_modality_features = {
            m: tokens[:, i, :] for i, m in enumerate(MODALITY_ORDER)
        }

        return {
            "logits": logits,
            "fused": fused,
            "per_modality_logits": per_modality_logits,
            "per_modality_features": per_modality_features,
            "norm_weights": norm_weights,
            "all_modalities_rejected": all_attn_masked.squeeze(1),
            "fallback_applied": fallback_applied.squeeze(1),
        }

    def get_adaptable_params(self, active_modalities: Optional[List[str]] = None) -> List[nn.Parameter]:
        """Collect adaptable parameters (LayerNorm + LowRankAdapters).

        If active_modalities is provided, only returns parameters for those modalities (AUD-05).
        """
        params: List[nn.Parameter] = []
        target_mods = active_modalities if active_modalities is not None else MODALITY_ORDER

        for m in target_mods:
            if m in self.projections:
                # LayerNorm parameters in projection
                for p in self.projections[m].norm.parameters():
                    params.append(p)
            if m in self.adapters:
                # Low-rank adapter parameters
                for p in self.adapters[m].parameters():
                    params.append(p)

        return params

    def get_total_params(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def save_source_checkpoint(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def reset_counters(self) -> None:
        """Reset telemetry counters for fallback and abstentions (Task 6)."""
        self.fallback_count = 0
        self.abstain_count = 0

    def load_source_checkpoint(self, path: str, device: str = "cpu") -> None:
        state = torch.load(path, map_location=device)
        self.load_state_dict(state)
