"""
Tri-SACTA: Anchor-Guided Selective Continual Test-Time Adaptation.

This module implements the core Tri-SACTA algorithm for multimodal
sentiment analysis under distribution shift. The key idea is to:

1. Use spectral shift detection to classify modalities into three sets:
   - Anchor (A): Low-shift, reliable modalities used as supervision signal
   - Adaptable (U): Mid-shift modalities that benefit from adaptation
   - Corrupted (C): High-shift modalities that are detached/suppressed

2. Perform selective adaptation only on adaptable parameters, guided by:
   - Entropy minimization on adaptable modality predictions
   - Cross-modal alignment: KL divergence between adaptable and anchor
   - Parameter regularization: L2 distance from source checkpoint

3. Track shift velocity and cross-modal disagreement for diagnostics.

Reference:
    Tri-SACTA: Spectral Anchor-Guided Continual Test-Time Adaptation
    with Conformal Prediction for Robust Multimodal Sentiment Analysis
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

try:
    from src.models.spectral_detector import SpectralShiftDetector
    from src.models.fusion_backbone import MultimodalFusionModel
except ImportError:
    from ..models.spectral_detector import SpectralShiftDetector
    from ..models.fusion_backbone import MultimodalFusionModel

logger = logging.getLogger(__name__)


def _js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute Jensen-Shannon divergence between two probability distributions.

    Args:
        p: Probability distribution of shape (batch, num_classes).
        q: Probability distribution of shape (batch, num_classes).
        eps: Small constant for numerical stability.

    Returns:
        Scalar tensor with the mean JS divergence across the batch.
    """
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    m = 0.5 * (p + q)
    kl_pm = F.kl_div(m.log(), p, reduction="batchmean", log_target=False)
    kl_qm = F.kl_div(m.log(), q, reduction="batchmean", log_target=False)
    return 0.5 * (kl_pm + kl_qm)


class TriSACTA:
    """Anchor-Guided Selective Continual Test-Time Adaptation.

    At each adaptation step, modalities are classified into anchor (A),
    adaptable (U), and corrupted (C) sets based on spectral shift scores.
    Adaptation is performed only when the adaptable set is non-empty,
    using anchor modalities as a reference signal and suppressing
    corrupted modalities.

    Attributes:
        model: The multimodal fusion model to adapt.
        detector: Spectral shift detector for per-modality shift scores.
        config: Full experiment config dict.
        optimizer: AdamW optimizer over adaptable parameters only.
        scaler: AMP GradScaler for mixed-precision training.
        source_state: Saved source model checkpoint for reset/regularization.
        prev_shift_scores: Previous step's shift scores for velocity tracking.
    """

    # Modality names in canonical order
    MODALITY_NAMES = ["visual", "audio", "text", "prosody"]

    def __init__(
        self,
        model: MultimodalFusionModel,
        detector: SpectralShiftDetector,
        config: dict,
    ) -> None:
        """Initialize Tri-SACTA.

        Saves a deep copy of the model's state dict as the source checkpoint,
        sets up AdamW optimizer on adaptable parameters only, and initializes
        the AMP GradScaler.

        Args:
            model: Pre-trained multimodal fusion model.
            detector: Spectral shift detector with registered source stats.
            config: Full experiment configuration dict. Expected keys:
                - config['tta']['tau']: Temperature for reliability weighting.
                - config['tta']['theta_high']: Upper threshold → corrupted set.
                - config['tta']['theta_low']: Lower threshold → anchor set.
                - config['tta']['lambda_align']: Weight for alignment loss.
                - config['tta']['lambda_reg']: Weight for regularization loss.
                - config['tta']['lr']: Learning rate for adaptation.
                - config['model']['num_classes']: Number of output classes.
        """
        self.model = model
        self.detector = detector
        self.config = config

        # TTA hyper-parameters
        tta_cfg = config["tta"]
        self.tau = tta_cfg["tau"]
        self.theta_high = tta_cfg["theta_high"]
        self.theta_low = tta_cfg["theta_low"]
        self.lambda_align = tta_cfg["lambda_align"]
        self.lambda_reg = tta_cfg["lambda_reg"]
        self.lr = tta_cfg["lr"]

        self.num_classes = config["model"]["num_classes"]

        # Save source checkpoint for reset and regularization
        self.source_state: Dict[str, torch.Tensor] = copy.deepcopy(
            model.state_dict()
        )

        # Set up optimizer on adaptable parameters only
        adaptable_params = model.get_adaptable_params()
        if not adaptable_params:
            logger.warning(
                "No adaptable parameters found in the model. "
                "Adaptation steps will be no-ops."
            )
        self.optimizer = torch.optim.AdamW(
            adaptable_params if adaptable_params else [torch.zeros(1, requires_grad=True)],
            lr=self.lr,
            weight_decay=0.0,  # Explicit L2 reg is used instead
        )

        # AMP scaler for mixed-precision
        self.scaler = GradScaler()

        # Tracking for shift velocity
        self.prev_shift_scores: Optional[Dict[str, float]] = None

        # Step counter
        self._step_count: int = 0

        logger.info(
            "TriSACTA initialized — tau=%.2f, θ_high=%.2f, θ_low=%.2f, "
            "λ_align=%.3f, λ_reg=%.4f, lr=%.1e",
            self.tau, self.theta_high, self.theta_low,
            self.lambda_align, self.lambda_reg, self.lr,
        )

    # ------------------------------------------------------------------
    # Core adaptation
    # ------------------------------------------------------------------

    def adapt_step(
        self,
        batch: Dict[str, torch.Tensor],
        modality_mask: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """Execute one Tri-SACTA adaptation step.

        Pipeline:
            1. Compute spectral shift scores for each modality.
            2. Derive reliability weights via softmax weighting.
            3. Classify modalities into anchor (A) / adaptable (U) / corrupted (C).
            4. If adaptable set U is non-empty, compute the TTA loss:
               L_TTA = L_ent(p_t)
                     + λ_align * Σ_{m∈U, a∈A} KL(p_m || sg(p_a))
                     + λ_reg  * ||θ − θ_0||²
               and perform one gradient step on adaptable parameters only.
            5. Return diagnostics dict.

        Args:
            batch: Dict with keys per modality → Tensor of shape (B, D_m).
                   Expected keys: 'visual', 'audio', 'text', 'prosody'.
            modality_mask: Dict with keys per modality → Tensor of shape (B,)
                           with 1.0 for available and 0.0 for missing.

        Returns:
            Dict with:
                - 'logits': (B, C) final fused logits.
                - 'shift_scores': {modality: float} spectral shift scores.
                - 'reliability_weights': {modality: float} per-modality weights.
                - 'anchor_set': list of anchor modality names.
                - 'adaptable_set': list of adaptable modality names.
                - 'corrupted_set': list of corrupted modality names.
                - 'loss': float total TTA loss (0.0 if no adaptation).
                - 'loss_ent': float entropy component.
                - 'loss_align': float alignment component.
                - 'loss_reg': float regularization component.
                - 'adapted': bool whether adaptation was performed.
        """
        self._step_count += 1
        device = next(self.model.parameters()).device

        # ----- 1. Spectral shift scores -----
        features_by_modality = self._extract_features(batch)
        shift_scores = self.detector.compute_shift_scores(features_by_modality)

        # ----- 2. Reliability weights -----
        reliability_weights = self.detector.compute_reliability_weights(
            shift_scores, modality_mask, self.tau
        )

        # ----- 3. Classify into A / U / C -----
        anchor_set, adaptable_set, corrupted_set = self._classify_modalities(
            shift_scores, modality_mask
        )

        # ----- Prepare result dict (populated progressively) -----
        result: Dict[str, Any] = {
            "shift_scores": shift_scores,
            "reliability_weights": reliability_weights,
            "anchor_set": anchor_set,
            "adaptable_set": adaptable_set,
            "corrupted_set": corrupted_set,
            "loss": 0.0,
            "loss_ent": 0.0,
            "loss_align": 0.0,
            "loss_reg": 0.0,
            "adapted": False,
        }

        # Build effective modality mask: zero-out corrupted modalities
        effective_mask = {}
        for m in self.MODALITY_NAMES:
            if m in corrupted_set:
                # Detach corrupted modalities → zero mask
                effective_mask[m] = torch.zeros_like(modality_mask.get(
                    m, torch.zeros(1, device=device)
                ))
            else:
                effective_mask[m] = modality_mask.get(
                    m, torch.ones(1, device=device)
                )

        # ----- 4. Adaptation (if adaptable set non-empty) -----
        if adaptable_set:
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            with autocast():
                # Forward pass with gradient through adaptable params
                out = self.model(
                    features=batch,
                    modality_mask=effective_mask,
                    reliability_weights=reliability_weights,
                )
                logits = out["logits"]  # (B, C)
                per_modality_logits = out.get("per_modality_logits", {})

                # --- (a) Entropy loss on fused predictions ---
                loss_ent = self._entropy_loss(logits)

                # --- (b) Cross-modal alignment loss ---
                loss_align = self._alignment_loss(
                    per_modality_logits, anchor_set, adaptable_set
                )

                # --- (c) Parameter regularization loss ---
                loss_reg = self._regularization_loss()

                # --- Total TTA loss ---
                loss_total = (
                    loss_ent
                    + self.lambda_align * loss_align
                    + self.lambda_reg * loss_reg
                )

            # Backward + optimizer step with AMP scaling
            self.scaler.scale(loss_total).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            result["logits"] = logits.detach()
            result["loss"] = loss_total.item()
            result["loss_ent"] = loss_ent.item()
            result["loss_align"] = loss_align.item()
            result["loss_reg"] = loss_reg.item()
            result["adapted"] = True

        else:
            # No adaptation — inference only
            self.model.eval()
            with torch.no_grad(), autocast():
                out = self.model(
                    features=batch,
                    modality_mask=effective_mask,
                    reliability_weights=reliability_weights,
                )
            result["logits"] = out["logits"]

        # Update shift velocity tracking
        self.prev_shift_scores = {m: float(v) for m, v in shift_scores.items()}

        return result

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the model to the saved source checkpoint.

        Restores model weights, re-creates the optimizer, resets the
        GradScaler, and clears internal tracking state.
        """
        self.model.load_state_dict(copy.deepcopy(self.source_state))

        adaptable_params = self.model.get_adaptable_params()
        self.optimizer = torch.optim.AdamW(
            adaptable_params if adaptable_params else [torch.zeros(1, requires_grad=True)],
            lr=self.lr,
            weight_decay=0.0,
        )
        self.scaler = GradScaler()
        self.prev_shift_scores = None
        self._step_count = 0

        logger.info("TriSACTA reset to source checkpoint.")

    # ------------------------------------------------------------------
    # Diagnostic metrics
    # ------------------------------------------------------------------

    def get_shift_velocity(self) -> Optional[Dict[str, float]]:
        """Compute per-modality shift velocity Δd_t.

        The shift velocity is the difference between the current and
        previous shift scores: Δd_t^m = d_t^m − d_{t-1}^m.

        Returns:
            Dict mapping modality name → Δd_t, or None if fewer than
            two steps have been taken.
        """
        if self.prev_shift_scores is None:
            return None
        # We need the current shift scores — but after adapt_step
        # prev_shift_scores already has the most recent scores.
        # Velocity is computed between consecutive calls, so we
        # cache a secondary buffer.
        if not hasattr(self, "_prev_prev_shift_scores") or self._prev_prev_shift_scores is None:
            return None
        velocity = {}
        for m in self.MODALITY_NAMES:
            cur = self.prev_shift_scores.get(m, 0.0)
            prev = self._prev_prev_shift_scores.get(m, 0.0)
            velocity[m] = cur - prev
        return velocity

    def get_cross_modal_disagreement(
        self,
        per_modality_logits: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute cross-modal disagreement κ_t via pairwise JS divergence.

        κ_t = (2 / M(M-1)) Σ_{i<j} JS(p_i || p_j)

        where the sum is over all pairs of available modality predictions.

        Args:
            per_modality_logits: Dict mapping modality name → (B, C) logits.

        Returns:
            Scalar tensor with the mean pairwise JS divergence.
        """
        modalities = [m for m in self.MODALITY_NAMES if m in per_modality_logits]
        if len(modalities) < 2:
            return torch.tensor(0.0)

        probs = {m: F.softmax(per_modality_logits[m], dim=-1) for m in modalities}

        js_sum = torch.tensor(0.0, device=next(iter(probs.values())).device)
        num_pairs = 0
        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                js_sum = js_sum + _js_divergence(probs[modalities[i]], probs[modalities[j]])
                num_pairs += 1

        return js_sum / max(num_pairs, 1)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_features(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Extract features from batch for spectral shift detection.

        Args:
            batch: Dict with modality name → Tensor of shape (B, D_m).

        Returns:
            Dict with modality name → Tensor suitable for the detector.
        """
        features: Dict[str, torch.Tensor] = {}
        for m in self.MODALITY_NAMES:
            if m in batch and batch[m] is not None:
                feat = batch[m]
                # Ensure 2D: (B, D)
                if feat.ndim > 2:
                    feat = feat.reshape(feat.shape[0], -1)
                features[m] = feat.detach()
        return features

    def _classify_modalities(
        self,
        shift_scores: Dict[str, float],
        modality_mask: Dict[str, torch.Tensor],
    ) -> Tuple[List[str], List[str], List[str]]:
        """Classify modalities into anchor, adaptable, and corrupted sets.

        Classification rules:
            - Anchor (A): d_t^m ≤ θ_low   AND modality is available
            - Corrupted (C): d_t^m ≥ θ_high  OR modality is unavailable
            - Adaptable (U): everything else (θ_low < d_t^m < θ_high)

        If all available modalities are corrupted, the least-shifted one
        is promoted to anchor to ensure at least one reference signal.

        Args:
            shift_scores: Per-modality spectral shift scores.
            modality_mask: Per-modality availability masks.

        Returns:
            Tuple of (anchor_set, adaptable_set, corrupted_set) lists.
        """
        # Cache previous scores for velocity computation
        if self.prev_shift_scores is not None:
            self._prev_prev_shift_scores = dict(self.prev_shift_scores)
        else:
            self._prev_prev_shift_scores = None

        anchor: List[str] = []
        adaptable: List[str] = []
        corrupted: List[str] = []

        for m in self.MODALITY_NAMES:
            # Check availability: mask might be batch tensor or scalar
            mask_val = modality_mask.get(m, None)
            if mask_val is None:
                corrupted.append(m)
                continue

            # Modality is available if any sample has it
            if isinstance(mask_val, torch.Tensor):
                is_available = mask_val.sum().item() > 0
            else:
                is_available = float(mask_val) > 0

            if not is_available:
                corrupted.append(m)
                continue

            score = float(shift_scores.get(m, 0.0))

            if score <= self.theta_low:
                anchor.append(m)
            elif score >= self.theta_high:
                corrupted.append(m)
            else:
                adaptable.append(m)

        # Safety: if no anchor and there are non-corrupted modalities,
        # promote the least-shifted adaptable modality to anchor.
        if not anchor and adaptable:
            adaptable_scores = {m: float(shift_scores.get(m, 0.0)) for m in adaptable}
            best = min(adaptable_scores, key=adaptable_scores.get)
            anchor.append(best)
            adaptable.remove(best)
            logger.debug(
                "No anchor found — promoted '%s' (score=%.4f) to anchor.",
                best, adaptable_scores[best],
            )

        # If everything is corrupted, promote the least-shifted one
        if not anchor and not adaptable and corrupted:
            available_corrupted = [
                m for m in corrupted
                if m in shift_scores
            ]
            if available_corrupted:
                scores_c = {m: float(shift_scores[m]) for m in available_corrupted}
                best = min(scores_c, key=scores_c.get)
                anchor.append(best)
                corrupted.remove(best)
                logger.debug(
                    "All corrupted — promoted '%s' (score=%.4f) to anchor.",
                    best, scores_c[best],
                )

        logger.debug(
            "Step %d sets — A=%s, U=%s, C=%s",
            self._step_count, anchor, adaptable, corrupted,
        )

        return anchor, adaptable, corrupted

    @staticmethod
    def _entropy_loss(logits: torch.Tensor) -> torch.Tensor:
        """Compute entropy minimization loss on fused predictions.

        L_ent = -Σ_c p_c log(p_c)  averaged over the batch.

        Args:
            logits: Tensor of shape (B, C) — unnormalized class scores.

        Returns:
            Scalar entropy loss tensor.
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)  # (B,)
        return entropy.mean()

    def _alignment_loss(
        self,
        per_modality_logits: Dict[str, torch.Tensor],
        anchor_set: List[str],
        adaptable_set: List[str],
    ) -> torch.Tensor:
        """Compute cross-modal alignment loss between adaptable and anchor.

        L_align = Σ_{m∈U} Σ_{a∈A} KL(p_m || sg(p_a))

        where sg(·) denotes stop-gradient (anchor predictions are detached).

        Args:
            per_modality_logits: Dict modality name → (B, C) logits.
            anchor_set: List of anchor modality names.
            adaptable_set: List of adaptable modality names.

        Returns:
            Scalar alignment loss tensor.
        """
        if not anchor_set or not adaptable_set:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)

        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        num_pairs = 0

        for m in adaptable_set:
            if m not in per_modality_logits:
                continue
            log_probs_m = F.log_softmax(per_modality_logits[m], dim=-1)

            for a in anchor_set:
                if a not in per_modality_logits:
                    continue
                # Stop-gradient on anchor probabilities
                probs_a = F.softmax(per_modality_logits[a].detach(), dim=-1)
                # KL(p_m || sg(p_a)) = Σ p_m * (log p_m - log sg(p_a))
                kl = F.kl_div(
                    log_probs_m, probs_a,
                    reduction="batchmean", log_target=False,
                )
                loss = loss + kl
                num_pairs += 1

        return loss / max(num_pairs, 1)

    def _regularization_loss(self) -> torch.Tensor:
        """Compute L2 regularization loss: ||θ − θ_0||².

        Only computed over adaptable parameters to match the optimizer scope.

        Returns:
            Scalar regularization loss tensor.
        """
        device = next(self.model.parameters()).device
        reg_loss = torch.tensor(0.0, device=device)

        current_state = self.model.state_dict()
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.source_state:
                source_param = self.source_state[name].to(device)
                reg_loss = reg_loss + (param - source_param).pow(2).sum()

        return reg_loss

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """Return serializable state for checkpointing.

        Returns:
            Dict with model state, optimizer state, scaler state, and
            internal tracking variables.
        """
        return {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "source_state": self.source_state,
            "prev_shift_scores": self.prev_shift_scores,
            "step_count": self._step_count,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load state from a checkpoint.

        Args:
            state: Dict produced by :meth:`state_dict`.
        """
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scaler.load_state_dict(state["scaler_state_dict"])
        self.source_state = state["source_state"]
        self.prev_shift_scores = state.get("prev_shift_scores")
        self._step_count = state.get("step_count", 0)
