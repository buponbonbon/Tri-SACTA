"""
Multimodal Fusion Backbone for Tri-SACTA.

Architecture overview
---------------------
1. **ModalityProjection** – per-modality linear + LayerNorm → ``fusion_dim``.
2. **LowRankAdapter** – lightweight adaptation via down- / up-projection
   (``fusion_dim → rank → fusion_dim``) with residual connection.
3. **CrossModalAttention** – multi-head cross-attention between any two
   modality streams.
4. **MultimodalFusionModel** – end-to-end model that wires the above blocks
   together, performs reliability-weighted fusion, and produces per-modality
   and fused classification logits.

Design goals
~~~~~~~~~~~~
* Total trainable parameters **< 2 M**.
* Compatible with PyTorch AMP (``torch.cuda.amp``).
* All forward outputs collected in a single ``dict`` for downstream use.
"""

from __future__ import annotations

import copy
import logging
import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Canonical modality order (used for deterministic iteration)
MODALITY_ORDER: List[str] = ["visual", "audio", "text", "prosody"]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ModalityProjection(nn.Module):
    """Project raw features of a single modality to the shared fusion dim.

    ``Linear(input_dim, fusion_dim)`` → ``LayerNorm(fusion_dim)``.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the raw modality features.
    fusion_dim : int
        Target dimensionality in the shared latent space.
    dropout : float
        Dropout probability applied after projection.
    """

    def __init__(self, input_dim: int, fusion_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, fusion_dim)
        self.norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(dropout)

        # Xavier init for the projection
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, D_in)`` raw features.

        Returns:
            ``(B, D_fused)`` normalised projected features.
        """
        return self.dropout(self.norm(self.linear(x)))


class LowRankAdapter(nn.Module):
    """Low-rank bottleneck adapter with residual connection.

    ``x → x + dropout(W_up(GELU(W_down(x))))``

    Parameters
    ----------
    dim : int
        Input (and output) dimensionality (== fusion_dim).
    rank : int
        Bottleneck rank.  Keeping this small (e.g. 8) limits parameter count.
    dropout : float
        Dropout on the adapter output.
    """

    def __init__(self, dim: int, rank: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Small-scale init so adapter is near-identity at the start
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, D)``

        Returns:
            ``(B, D)`` adapted features (residual added).
        """
        return x + self.dropout(self.up(F.gelu(self.down(x))))


class CrossModalAttention(nn.Module):
    """Multi-head cross-attention between two modality streams.

    Query comes from one modality, key/value from another.

    Parameters
    ----------
    dim : int
        Embedding dimension (== fusion_dim).
    num_heads : int
        Number of attention heads.
    dropout : float
        Attention dropout.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        assert dim % num_heads == 0, (
            f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        )
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: ``(B, D)`` – modality that *receives* information.
            key_value: ``(B, D)`` – modality that *sends* information.
            mask: Optional ``(B,)`` or ``(B, 1)`` availability mask for key_value.

        Returns:
            ``(B, D)`` – updated query representation.
        """
        B = query.size(0)
        H, d = self.num_heads, self.head_dim

        # Pre-norm
        q = self.norm_q(query)
        kv = self.norm_kv(key_value)

        # Treat each sample as a length-1 sequence → (B, 1, H, d)
        Q = self.q_proj(q).view(B, 1, H, d).transpose(1, 2)   # (B, H, 1, d)
        K = self.k_proj(kv).view(B, 1, H, d).transpose(1, 2)  # (B, H, 1, d)
        V = self.v_proj(kv).view(B, 1, H, d).transpose(1, 2)  # (B, H, 1, d)

        attn = (Q @ K.transpose(-2, -1)) * self.scale  # (B, H, 1, 1)

        # Optional mask: zero out attention for missing modalities
        if mask is not None:
            # mask shape: (B,) or (B, 1) → (B, 1, 1, 1)
            attn_mask = mask.view(B, 1, 1, 1).float()
            # Replace masked positions with large negative so softmax -> 0 (FP16 safe)
            attn = attn.masked_fill(attn_mask == 0, -1e4)

        attn = self.attn_drop(torch.softmax(attn, dim=-1))
        out = (attn @ V).transpose(1, 2).reshape(B, -1)  # (B, D)
        return query + self.out_proj(out)


# ---------------------------------------------------------------------------
# Cross-modal attention layer (one "layer" attends across all modality pairs)
# ---------------------------------------------------------------------------


class CrossModalAttentionLayer(nn.Module):
    """One layer of pairwise cross-modal attention + per-modality FFN.

    For each modality, we aggregate cross-attention outputs from *all other
    present modalities* (mean-pooled), then apply a small feed-forward block.

    Parameters
    ----------
    dim : int
        Fusion dimensionality.
    num_heads : int
        Attention heads.
    dropout : float
        Dropout.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        # Shared cross-attention block (parameter-efficient: one set of Q/K/V)
        self.cross_attn = CrossModalAttention(dim, num_heads, dropout)
        # Per-modality FFN (small, shared across modalities to save params)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        modality_mask: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: ``{modality: (B, D)}``.
            modality_mask: ``{modality: (B,) or scalar}``.

        Returns:
            Updated ``{modality: (B, D)}``.
        """
        updated: Dict[str, torch.Tensor] = {}
        modalities = [m for m in MODALITY_ORDER if m in features]

        for m_q in modalities:
            q = features[m_q]
            # Gather cross-attention from every *other* modality
            cross_outs = []
            for m_kv in modalities:
                if m_kv == m_q:
                    continue
                kv = features[m_kv]
                kv_mask = modality_mask.get(m_kv)
                cross_outs.append(self.cross_attn(q, kv, kv_mask))
            if cross_outs:
                # Mean-pool across contributing modalities
                aggregated = torch.stack(cross_outs, dim=0).mean(dim=0)
            else:
                aggregated = q  # only one modality available

            # FFN with residual
            updated[m_q] = aggregated + self.ffn(aggregated)

        return updated


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class MultimodalFusionModel(nn.Module):
    """End-to-end multimodal fusion model for Tri-SACTA.

    Architecture
    ~~~~~~~~~~~~
    1. Per-modality projection (Linear + LayerNorm) to ``fusion_dim``.
    2. Per-modality low-rank adapters.
    3. ``num_layers`` cross-modal attention layers.
    4. Reliability-weighted average pooling across modalities.
    5. Classification head: LayerNorm → Linear → logits.

    Parameters
    ----------
    config : dict
        Full experiment config (or at least the ``model`` and ``features``
        sub-dicts).  See the shared interface specification.
    """

    # Mapping from canonical modality name to config key for input dim
    _DIM_KEYS = {
        "visual": "visual_dim",
        "audio": "audio_dim",
        "text": "text_dim",
        "prosody": "prosody_dim",
    }

    def __init__(self, config: dict) -> None:
        super().__init__()

        feat_cfg = config["features"]
        model_cfg = config["model"]

        self.fusion_dim: int = model_cfg["fusion_dim"]
        self.num_heads: int = model_cfg.get("num_heads", 4)
        self.num_layers: int = model_cfg.get("num_layers", 2)
        self.adapter_rank: int = model_cfg.get("adapter_rank", 8)
        self.dropout: float = model_cfg.get("dropout", 0.1)
        self.num_classes: int = model_cfg["num_classes"]

        # --- Per-modality projection heads ---
        self.projections = nn.ModuleDict()
        for mod, key in self._DIM_KEYS.items():
            in_dim = feat_cfg[key]
            self.projections[mod] = ModalityProjection(
                in_dim, self.fusion_dim, self.dropout
            )

        # --- Per-modality low-rank adapters ---
        self.adapters = nn.ModuleDict()
        for mod in MODALITY_ORDER:
            self.adapters[mod] = LowRankAdapter(
                self.fusion_dim, self.adapter_rank, self.dropout
            )

        # --- Cross-modal attention layers ---
        self.cross_layers = nn.ModuleList([
            CrossModalAttentionLayer(
                self.fusion_dim, self.num_heads, self.dropout
            )
            for _ in range(self.num_layers)
        ])

        # --- Classification head ---
        self.classifier_norm = nn.LayerNorm(self.fusion_dim)
        self.classifier = nn.Linear(self.fusion_dim, self.num_classes)

        # --- Per-modality auxiliary classifiers (for per-modality logits) ---
        self.modality_classifiers = nn.ModuleDict()
        for mod in MODALITY_ORDER:
            self.modality_classifiers[mod] = nn.Linear(
                self.fusion_dim, self.num_classes
            )

        # Initialise classifier heads
        for m in [self.classifier, *self.modality_classifiers.values()]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

        # Log parameter count
        total, trainable = self.get_total_params()
        logger.info(
            "MultimodalFusionModel initialised: %d total params, %d trainable (%.2f M)",
            total, trainable, trainable / 1e6,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        modality_mask: Dict[str, torch.Tensor],
        reliability_weights: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass.

        Args:
            features: ``{modality_name: (B, D_m)}`` raw extracted features.
            modality_mask: ``{modality_name: (B,)}`` binary availability masks.
            reliability_weights: ``{modality_name: (B,) or scalar}`` soft
                weights from the shift detector.  If *None*, uniform weights
                (gated by ``modality_mask``) are used.

        Returns:
            dict with keys:

            - ``logits`` – ``(B, C)`` fused classification logits.
            - ``fused`` – ``(B, fusion_dim)`` fused representation.
            - ``per_modality_logits`` – ``{mod: (B, C)}``.
            - ``per_modality_features`` – ``{mod: (B, fusion_dim)}``.
        """
        device = next(self.parameters()).device
        present = [m for m in MODALITY_ORDER if m in features]
        B = features[present[0]].size(0)

        # 1) Project to fusion_dim
        projected: Dict[str, torch.Tensor] = {}
        for mod in present:
            x = features[mod].to(device)
            projected[mod] = self.projections[mod](x)  # (B, fusion_dim)

        # 2) Low-rank adapters
        adapted: Dict[str, torch.Tensor] = {}
        for mod in present:
            adapted[mod] = self.adapters[mod](projected[mod])

        # 3) Cross-modal attention layers
        hidden = adapted
        for layer in self.cross_layers:
            hidden = layer(hidden, modality_mask)

        # 4) Per-modality outputs
        per_modality_features: Dict[str, torch.Tensor] = {}
        per_modality_logits: Dict[str, torch.Tensor] = {}
        for mod in present:
            per_modality_features[mod] = hidden[mod]
            per_modality_logits[mod] = self.modality_classifiers[mod](hidden[mod])

        # 5) Reliability-weighted fusion
        if reliability_weights is None:
            # Uniform weights gated by mask
            rw = {
                mod: modality_mask.get(mod, torch.ones(B, device=device)).float()
                for mod in present
            }
        else:
            rw = {}
            for mod in present:
                w = reliability_weights.get(mod, torch.ones(B, device=device))
                if not isinstance(w, torch.Tensor):
                    w = torch.tensor(w, device=device).expand(B)
                else:
                    w = w.to(device).float()
                    if w.dim() == 0:
                        w = w.expand(B)
                rw[mod] = w

        # Weighted average over modalities
        # Stack weights: (num_mod, B) and features: (num_mod, B, D)
        weight_stack = torch.stack([rw[m] for m in present], dim=0)  # (M, B)
        feat_stack = torch.stack([hidden[m] for m in present], dim=0)  # (M, B, D)

        # Normalise weights so they sum to 1 across modalities (per sample)
        weight_sum = weight_stack.sum(dim=0, keepdim=True).clamp(min=1e-8)  # (1, B)
        norm_weights = weight_stack / weight_sum  # (M, B)

        fused = (norm_weights.unsqueeze(-1) * feat_stack).sum(dim=0)  # (B, D)

        # 6) Classification
        logits = self.classifier(self.classifier_norm(fused))  # (B, C)

        return {
            "logits": logits,
            "fused": fused,
            "per_modality_logits": per_modality_logits,
            "per_modality_features": per_modality_features,
        }

    # ------------------------------------------------------------------
    # Adaptable parameters (for TTA)
    # ------------------------------------------------------------------

    def get_adaptable_params(self) -> List[nn.Parameter]:
        """Return only the parameters that should be adapted at test time.

        These are:
        * All **LayerNorm** parameters (affine weight & bias).
        * All **LowRankAdapter** parameters.
        """
        params: List[nn.Parameter] = []
        for name, module in self.named_modules():
            if isinstance(module, (nn.LayerNorm, LowRankAdapter)):
                for p in module.parameters():
                    params.append(p)
        # Deduplicate (in case a LayerNorm is inside an adapter)
        seen_ids = set()
        unique: List[nn.Parameter] = []
        for p in params:
            if id(p) not in seen_ids:
                seen_ids.add(id(p))
                unique.append(p)
        return unique

    # ------------------------------------------------------------------
    # Parameter counts
    # ------------------------------------------------------------------

    def get_total_params(self) -> Tuple[int, int]:
        """Return ``(total_params, trainable_params)``."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_source_checkpoint(self, path: str) -> None:
        """Save model weights as the *source* (pre-adaptation) checkpoint.

        Only the ``state_dict`` is saved so that loading is architecture-
        agnostic.

        Args:
            path: File path (e.g. ``"checkpoints/source.pt"``).
        """
        torch.save(
            {"model_state_dict": self.state_dict()},
            path,
        )
        logger.info("Source checkpoint saved → %s", path)

    def load_source_checkpoint(self, path: str, strict: bool = True) -> None:
        """Load model weights from a source checkpoint.

        Args:
            path: Path to a checkpoint saved by :meth:`save_source_checkpoint`.
            strict: Whether to enforce exact key matching.
        """
        ckpt = torch.load(path, map_location="cpu")
        self.load_state_dict(ckpt["model_state_dict"], strict=strict)
        logger.info("Source checkpoint loaded ← %s", path)
