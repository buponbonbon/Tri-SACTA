"""
Tri-SACTA: Anchor-Guided Selective Continual Test-Time Adaptation.

Key scientific improvements:
1. Classification into Anchor (A), Adaptable (U), and Corrupted (D) sets based on
   calibrated reliability weights r_t in [0, 1] (AUD-01, AUD-05).
2. Strict Selective Adaptation:
   - Freezes Anchor and Corrupted modality streams completely.
   - Updates ONLY parameters in adaptable_set U (LayerNorm + LowRankAdapters).
   - Regularization loss L_reg is strictly computed over active adaptable params.
3. Supervised anchor alignment:
   - Adaptable predictions are aligned to detached clean anchor predictions.
4. Shift velocity & cross-modal disagreement (JS divergence) explicitly tracked and returned.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from src.models.spectral_detector import SpectralShiftDetector
from src.models.fusion_backbone import MultimodalFusionModel

logger = logging.getLogger(__name__)


def _js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute Jensen-Shannon divergence between two probability distributions."""
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    m = 0.5 * (p + q)
    kl_pm = F.kl_div(m.log(), p, reduction="batchmean", log_target=False)
    kl_qm = F.kl_div(m.log(), q, reduction="batchmean", log_target=False)
    return 0.5 * (kl_pm + kl_qm)


class TriSACTA:
    """Anchor-Guided Selective Continual Test-Time Adaptation."""

    MODALITY_NAMES: List[str] = ["visual", "audio", "text", "prosody"]

    def __init__(
        self,
        model: MultimodalFusionModel,
        detector: SpectralShiftDetector,
        config: dict,
    ) -> None:
        self.model = model
        self.detector = detector
        self.config = config

        tta_cfg = config.get("tta", {})
        self.tau = float(tta_cfg.get("tau", 1.0))
        self.theta_high = float(tta_cfg.get("theta_high", 0.7))
        self.theta_low = float(tta_cfg.get("theta_low", 0.3))
        self.lambda_align = float(tta_cfg.get("lambda_align", 0.5))
        self.lambda_reg = float(tta_cfg.get("lambda_reg", 0.01))
        self.lr = float(tta_cfg.get("lr", 1e-4))

        self.num_classes = config.get("model", {}).get("num_classes", 7)

        # Freeze source state snapshot
        self.source_state: Dict[str, torch.Tensor] = {
            k: v.clone().detach().cpu() for k, v in model.state_dict().items()
        }

        # Optimizer on adaptable parameters
        adaptable_params = model.get_adaptable_params()
        self.optimizer = torch.optim.AdamW(
            adaptable_params if adaptable_params else [torch.zeros(1, requires_grad=True)],
            lr=self.lr,
            weight_decay=0.0,
        )

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.scaler = GradScaler(device_type, enabled=(device_type == "cuda"))

        self.prev_shift_scores: Optional[Dict[str, float]] = None
        self._prev_prev_shift_scores: Optional[Dict[str, float]] = None
        self._step_count: int = 0

        logger.info(
            f"TriSACTA initialized — tau={self.tau:.2f}, theta_high={self.theta_high:.2f}, "
            f"theta_low={self.theta_low:.2f}, lambda_align={self.lambda_align:.3f}, lambda_reg={self.lambda_reg:.4f}, lr={self.lr:.1e}"
        )

    def adapt_step(
        self,
        batch: Dict[str, torch.Tensor],
        modality_mask: Dict[str, torch.Tensor],
        precomputed_shift_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Execute one Tri-SACTA adaptation step with selective parameter updates."""
        self._step_count += 1
        device = next(self.model.parameters()).device

        # 1. Compute spectral shift scores excluding missing samples (or reuse precomputed)
        if precomputed_shift_scores is not None:
            shift_scores = precomputed_shift_scores
        else:
            try:
                shift_scores = self.detector.compute_shift_scores(batch, modality_mask, update_history=True)
            except TypeError:
                shift_scores = self.detector.compute_shift_scores(batch, modality_mask)

        # 2. Derive reliability weights
        reliability_weights = self.detector.compute_reliability_weights(
            shift_scores, modality_mask, self.tau
        )

        # 3. Classify modalities into Anchor (A), Adaptable (U), Corrupted (D)
        anchor_set, adaptable_set, corrupted_set = self._classify_modalities(
            reliability_weights, modality_mask
        )

        # Build effective mask isolating corrupted channels from attention (R3-07)
        effective_mask = {}
        B = next(iter(batch.values())).shape[0] if batch else 1
        for m in self.MODALITY_NAMES:
            p_mask = modality_mask[m].to(device).float().view(-1) if m in modality_mask else torch.zeros(B, device=device)
            is_active = 1.0 if m not in corrupted_set else 0.0
            effective_mask[m] = p_mask * is_active

        # Initial forward pass to compute unimodal outputs and disagreement
        self.model.eval()
        with torch.no_grad():
            init_out = self.model(
                batch,
                modality_mask=modality_mask,
                reliability_weights=reliability_weights,
                effective_mask=effective_mask,
            )
            per_mod_logits = init_out["per_modality_logits"]

        # 4. Cross-modal disagreement tracking on pairwise valid samples (R3-07)
        available_mods = [m for m in self.MODALITY_NAMES if m in anchor_set or m in adaptable_set]
        disagreement = self.get_cross_modal_disagreement(per_mod_logits, available_mods, modality_mask=modality_mask)

        result: Dict[str, Any] = {
            "shift_scores": shift_scores,
            "reliability_weights": {m: float(w.mean().item()) if isinstance(w, torch.Tensor) else float(w) for m, w in reliability_weights.items()},
            "anchor_set": list(anchor_set),
            "adaptable_set": list(adaptable_set),
            "corrupted_set": list(corrupted_set),
            "cross_modal_disagreement": float(disagreement),
            "loss": 0.0,
            "loss_ent": 0.0,
            "loss_align": 0.0,
            "loss_reg": 0.0,
            "adapted": False,
        }

        # 5. Selective Adaptation (strictly requires clean anchor, R3-07)
        if adaptable_set and anchor_set:
            self.model.train()

            # Strictly freeze all non-adaptable parameters (AUD-05)
            for param in self.model.parameters():
                param.requires_grad = False

            adaptable_params = self.model.get_adaptable_params(active_modalities=list(adaptable_set))
            for p in adaptable_params:
                p.requires_grad = True

            self.optimizer.zero_grad(set_to_none=True)

            device_type = "cuda" if device.type == "cuda" else "cpu"
            with autocast(device_type, enabled=(device_type == "cuda")):
                out = self.model(
                    features=batch,
                    modality_mask=modality_mask,
                    reliability_weights=reliability_weights,
                    effective_mask=effective_mask,
                )
                logits = out["logits"]
                curr_per_mod_logits = out["per_modality_logits"]

                # (a) Entropy minimization on fused prediction
                loss_ent = self._entropy_loss(logits)

                # (b) Cross-modal alignment: align adaptable to detached anchor heads
                loss_align = self._alignment_loss(
                    curr_per_mod_logits, anchor_set, adaptable_set, modality_mask
                )

                # (c) Parameter regularization: only on active adaptable parameters
                loss_reg = self._regularization_loss(adaptable_set, device)

                loss_total = (
                    loss_ent
                    + self.lambda_align * loss_align
                    + self.lambda_reg * loss_reg
                )

            if device_type == "cuda":
                self.scaler.scale(loss_total).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_total.backward()
                self.optimizer.step()

            # Clean gradients
            for p in self.model.parameters():
                p.grad = None

            # Clean post-adaptation evaluation forward pass in eval() mode (AUD-25)
            self.model.eval()
            with torch.no_grad():
                post_out = self.model(
                    features=batch,
                    modality_mask=modality_mask,
                    reliability_weights=reliability_weights,
                )
            result["logits"] = post_out["logits"].detach()
            result["loss"] = float(loss_total.item())
            result["loss_ent"] = float(loss_ent.item())
            result["loss_align"] = float(loss_align.item())
            result["loss_reg"] = float(loss_reg.item())
            result["adapted"] = True
        else:
            # Inference only (no reliable anchor or no adaptable modality)
            result["logits"] = init_out["logits"].detach()

        # Update shift history for velocity calculation
        self._prev_prev_shift_scores = self.prev_shift_scores
        self.prev_shift_scores = {m: float(v) for m, v in shift_scores.items()}

        return result

    def _classify_modalities(
        self,
        reliability_weights: Dict[str, torch.Tensor],
        modality_mask: Dict[str, torch.Tensor],
    ) -> Tuple[Set[str], Set[str], Set[str]]:
        """Classify modalities using calibrated reliability weights r_t in [0, 1]."""
        anchor: Set[str] = set()
        adaptable: Set[str] = set()
        corrupted: Set[str] = set()

        for m in self.MODALITY_NAMES:
            mask_val = modality_mask.get(m, None)
            if mask_val is None or (isinstance(mask_val, torch.Tensor) and mask_val.sum().item() == 0):
                corrupted.add(m)
                continue

            rw = reliability_weights.get(m, 0.0)
            r_val = float(rw.mean().item()) if isinstance(rw, torch.Tensor) else float(rw)

            if r_val >= self.theta_high:
                anchor.add(m)
            elif r_val >= self.theta_low:
                adaptable.add(m)
            else:
                corrupted.add(m)

        # Strict no-anchor policy (R3-07): do NOT promote adaptable modalities to clean anchors.
        # If no clean anchor exists, anchor remains empty, safely skipping adaptation.
        return anchor, adaptable, corrupted

    def _entropy_loss(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        return entropy

    def _alignment_loss(
        self,
        per_modality_logits: Dict[str, torch.Tensor],
        anchor_set: Set[str],
        adaptable_set: Set[str],
        modality_mask: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Align adaptable predictions to stop-gradient anchor predictions."""
        device = next(self.model.parameters()).device
        loss = torch.tensor(0.0, device=device)
        num_pairs = 0

        for a in anchor_set:
            if a not in per_modality_logits:
                continue
            # Stop gradient on anchor teacher
            p_a = F.softmax(per_modality_logits[a].detach(), dim=-1)
            a_mask = modality_mask.get(a, torch.ones(p_a.size(0), device=device)).bool().view(-1)

            for u in adaptable_set:
                if u not in per_modality_logits:
                    continue
                p_u_log = F.log_softmax(per_modality_logits[u], dim=-1)
                u_mask = modality_mask.get(u, torch.ones(p_u_log.size(0), device=device)).bool().view(-1)

                valid_pair_mask = a_mask & u_mask
                if valid_pair_mask.any():
                    # KL(p_u || sg(p_a))
                    kl = F.kl_div(
                        p_u_log[valid_pair_mask],
                        p_a[valid_pair_mask],
                        reduction="batchmean",
                        log_target=False,
                    )
                    loss = loss + kl
                    num_pairs += 1

        return loss / max(num_pairs, 1)

    def _regularization_loss(self, adaptable_set: Set[str], device: torch.device) -> torch.Tensor:
        """Compute L2 regularization only over active adaptable parameters."""
        reg_loss = torch.tensor(0.0, device=device)
        for m in adaptable_set:
            if m in self.model.projections:
                for name, p in self.model.projections[m].norm.named_parameters():
                    full_name = f"projections.{m}.norm.{name}"
                    if full_name in self.source_state:
                        reg_loss = reg_loss + (p - self.source_state[full_name].to(device)).pow(2).sum()
            if m in self.model.adapters:
                for name, p in self.model.adapters[m].named_parameters():
                    full_name = f"adapters.{m}.{name}"
                    if full_name in self.source_state:
                        reg_loss = reg_loss + (p - self.source_state[full_name].to(device)).pow(2).sum()
        return reg_loss

    def get_shift_velocity(self) -> Dict[str, float]:
        """Compute per-modality shift velocity Δd_t."""
        return self.detector.compute_shift_velocity()

    def get_cross_modal_disagreement(
        self,
        per_modality_logits: Dict[str, torch.Tensor],
        available_modalities: List[str],
        modality_mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> float:
        """Compute pairwise JS divergence strictly on valid sample intersections (R3-07)."""
        valid_mods = [m for m in available_modalities if m in per_modality_logits]
        M = len(valid_mods)
        if M < 2:
            return 0.0

        probs = {m: F.softmax(per_modality_logits[m], dim=-1) for m in valid_mods}
        js_total = 0.0
        pairs = 0

        for i in range(M):
            for j in range(i + 1, M):
                m_i, m_j = valid_mods[i], valid_mods[j]
                if modality_mask is not None and m_i in modality_mask and m_j in modality_mask:
                    mask_i = modality_mask[m_i].view(-1) > 0.5
                    mask_j = modality_mask[m_j].view(-1) > 0.5
                    valid_inter = mask_i & mask_j
                    if valid_inter.sum() == 0:
                        continue
                    p_i = probs[m_i][valid_inter]
                    p_j = probs[m_j][valid_inter]
                else:
                    p_i = probs[m_i]
                    p_j = probs[m_j]

                js = _js_divergence(p_i, p_j)
                js_total += float(js.item())
                pairs += 1

        return float(js_total / max(pairs, 1))

    def reset(self) -> None:
        """Reset model parameters to source checkpoint."""
        self.model.load_state_dict(copy.deepcopy(self.source_state))
        adaptable_params = self.model.get_adaptable_params()
        self.optimizer = torch.optim.AdamW(
            adaptable_params if adaptable_params else [torch.zeros(1, requires_grad=True)],
            lr=self.lr,
            weight_decay=0.0,
        )
        self.prev_shift_scores = None
        self._prev_prev_shift_scores = None
        self.detector.reset_history()
        self._step_count = 0
