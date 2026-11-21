"""
Mask-and-Disagreement-Driven Online Adaptive Conformal Inference (OnlineACI).

Implements online conformal prediction with:
  - Per-mask-pattern adaptive significance levels (alpha_t^m)
  - Shift-velocity and cross-modal disagreement modulated learning rates
  - APS (Adaptive Prediction Sets) and LAC (Least Ambiguous Conformity) scores
  - Sliding-window quantile estimation per modality-mask pattern
  - Delayed feedback buffering for realistic streaming scenarios

Reference: Gibbs & Candès (2021), "Adaptive Conformal Inference Under
Distribution Shift", extended with mask-conditional and multi-modal controls.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class OnlineACI:
    """Online Adaptive Conformal Inference with mask-conditional control.

    Maintains per-mask-pattern sliding windows of nonconformity scores and
    adaptively adjusts the significance level based on:
      1. Empirical mis-coverage feedback  (standard ACI)
      2. Distribution shift velocity      (β₁ · δ_d)
      3. Cross-modal disagreement         (β₂ · κ)

    The effective learning rate is:
        η_t = η₀ · (1 + β₁ · δ_d + β₂ · κ)

    Args:
        num_classes: Number of target classes (e.g. 7 for MOSEI sentiment).
        alpha: Target mis-coverage rate (e.g. 0.1 for 90 % coverage).
        config: ``config['conformal']`` sub-dict with hyper-parameters.
    """

    # --------------------------------------------------------------------- #
    # Construction
    # --------------------------------------------------------------------- #

    def __init__(self, num_classes: int, alpha: float, config: Dict[str, Any]):
        self.num_classes = num_classes
        self.alpha = alpha

        # --- hyper-parameters from config -------------------------------- #
        conf = config.get("conformal", config)  # accept either nested or flat
        self.eta_0: float = float(conf.get("eta_0", 0.01))
        self.beta_1: float = float(conf.get("beta_1", 0.5))
        self.beta_2: float = float(conf.get("beta_2", 0.5))
        self.window_size: int = int(conf.get("window_size", 200))
        self.delay: int = int(conf.get("delay", 0))
        self.score_fn_name: str = str(conf.get("score_fn", "aps")).lower()

        # --- per-mask adaptive state ------------------------------------- #
        # alpha_t^m  for each mask pattern key
        self.alphas: Dict[str, float] = defaultdict(lambda: self.alpha)
        # sliding window of (score, label_hit) tuples per mask
        self.score_windows: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )

        # --- delayed-feedback buffer ------------------------------------- #
        # stores (logits, label, mask_key, shift_vel, disagree, t) waiting
        self._delay_buffer: deque = deque()
        self._global_t: int = 0

        # --- rolling metrics --------------------------------------------- #
        self._coverage_history: List[float] = []
        self._set_size_history: List[float] = []
        self._per_mask_coverage: Dict[str, List[float]] = defaultdict(list)
        self._per_mask_set_size: Dict[str, List[float]] = defaultdict(list)

    # --------------------------------------------------------------------- #
    # Nonconformity scores
    # --------------------------------------------------------------------- #

    def compute_nonconformity_scores(
        self,
        logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute nonconformity scores for a batch.

        Supports two scoring functions:

        **APS** (Adaptive Prediction Sets):
            Sort softmax probabilities in descending order, compute cumulative
            sums, and return the cumulative mass at which the true class is
            included.  For prediction (no ``labels``), returns the full matrix
            of per-class cumulative scores ``(B, C)``.

        **LAC** (Least Ambiguous Conformity):
            ``s(x, y) = 1 − p̂(y | x)``.  For prediction, returns
            ``1 − softmax(logits)`` of shape ``(B, C)``.

        Args:
            logits: Raw model outputs ``(B, C)``.
            labels: Ground-truth labels ``(B,)``.  When *None*, per-class
                scores are returned for prediction-set construction.

        Returns:
            If ``labels`` is given: ``(B,)`` scalar scores.
            Otherwise: ``(B, C)`` per-class scores.
        """
        # Ensure float32 for numerical stability (AMP safe)
        logits_f = logits.float()
        probs = F.softmax(logits_f, dim=-1)  # (B, C)

        if self.score_fn_name == "aps":
            return self._aps_scores(probs, labels)
        elif self.score_fn_name == "lac":
            return self._lac_scores(probs, labels)
        else:
            raise ValueError(
                f"Unknown score function: '{self.score_fn_name}'. "
                "Supported: 'aps', 'lac'."
            )

    # ---- APS ------------------------------------------------------------ #

    @staticmethod
    def _aps_scores(
        probs: torch.Tensor, labels: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Adaptive Prediction Sets nonconformity scores.

        Sorts softmax probabilities in descending order and computes the
        cumulative sum.  The score for a class *y* is the cumulative mass up
        to (and including) *y* in the sorted order.  Smaller scores mean the
        class is more likely; a prediction set includes all classes whose
        score ≤ quantile.

        When ``labels`` is None, returns the full ``(B, C)`` matrix so that
        ``predict_set`` can threshold each class independently.
        """
        B, C = probs.shape

        # Sort descending
        sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
        cum_probs = sorted_probs.cumsum(dim=-1)  # (B, C)

        # Map cumulative values back to original class positions
        # inv_indices[b, sorted_indices[b, j]] = j  →  cum_scores[b, c]
        inv_indices = sorted_indices.argsort(dim=-1)
        cum_scores = cum_probs.gather(dim=-1, index=inv_indices)  # (B, C)

        if labels is None:
            return cum_scores  # (B, C)

        # Gather the score for the true label
        labels_expanded = labels.long().unsqueeze(-1)  # (B, 1)
        scores = cum_scores.gather(dim=-1, index=labels_expanded).squeeze(-1)
        return scores  # (B,)

    # ---- LAC ------------------------------------------------------------ #

    @staticmethod
    def _lac_scores(
        probs: torch.Tensor, labels: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Least Ambiguous Conformity scores: ``1 − p̂(y | x)``."""
        if labels is None:
            return 1.0 - probs  # (B, C)

        labels_expanded = labels.long().unsqueeze(-1)
        p_true = probs.gather(dim=-1, index=labels_expanded).squeeze(-1)
        return 1.0 - p_true  # (B,)

    # --------------------------------------------------------------------- #
    # Prediction sets
    # --------------------------------------------------------------------- #

    def predict_set(
        self, logits: torch.Tensor, modality_mask_key: str
    ) -> List[List[int]]:
        """Construct conformal prediction sets for a batch.

        Uses the sliding-window quantile of past nonconformity scores (for the
        given mask pattern) to threshold per-class APS / LAC scores.

        Args:
            logits: ``(B, C)`` raw model outputs.
            modality_mask_key: String key encoding the active modality pattern,
                e.g. ``"1110"`` when visual/audio/text are present but prosody
                is missing.

        Returns:
            List of length ``B``, each element a sorted list of class indices
            included in the prediction set.
        """
        # Compute per-class nonconformity scores (B, C)
        with torch.no_grad():
            per_class_scores = self.compute_nonconformity_scores(logits, labels=None)

        # Determine the adaptive quantile threshold for this mask
        q_hat = self._get_quantile(modality_mask_key)

        # Build prediction sets
        per_class_np = per_class_scores.detach().cpu().numpy()  # (B, C)
        prediction_sets: List[List[int]] = []
        for i in range(per_class_np.shape[0]):
            pset = [
                int(c)
                for c in range(self.num_classes)
                if per_class_np[i, c] <= q_hat
            ]
            # Guarantee non-empty set: include top-1 prediction
            if len(pset) == 0:
                top1 = int(logits[i].argmax().item())
                pset = [top1]
            prediction_sets.append(sorted(pset))
        return prediction_sets

    # --------------------------------------------------------------------- #
    # Online update
    # --------------------------------------------------------------------- #

    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        modality_mask_key: str,
        shift_velocity: float = 0.0,
        disagreement: float = 0.0,
    ) -> None:
        """Update adaptive alpha and score window after observing true labels.

        Implements the mask-and-disagreement-driven ACI update:

        .. math::

            \\alpha_{t+1}^m = \\text{clip}\\bigl(
                \\alpha_t^m + \\eta_t \\cdot (\\alpha_{\\text{target}}
                - \\mathbb{1}[y_t \\notin C_t]),\\; 0,\\; 1
            \\bigr)

        where the step size adapts to shift and disagreement:

        .. math::

            \\eta_t = \\eta_0 \\cdot (1 + \\beta_1 \\cdot \\delta_d
                      + \\beta_2 \\cdot \\kappa)

        Args:
            logits: ``(B, C)`` raw model outputs.
            labels: ``(B,)`` ground-truth class indices.
            modality_mask_key: Mask pattern string.
            shift_velocity: ``δ_d`` – rate of change of distribution shift
                magnitude (from :class:`SpectralShiftDetector`).
            disagreement: ``κ`` – cross-modal prediction disagreement.
        """
        self._global_t += 1

        # If delayed feedback is configured, buffer and process matured items
        if self.delay > 0:
            self._delay_buffer.append(
                (logits.detach().cpu(), labels.detach().cpu(),
                 modality_mask_key, shift_velocity, disagreement,
                 self._global_t)
            )
            self._flush_delay_buffer()
            return

        # Immediate update
        self._do_update(logits, labels, modality_mask_key,
                        shift_velocity, disagreement)

    def _do_update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        modality_mask_key: str,
        shift_velocity: float,
        disagreement: float,
    ) -> None:
        """Core ACI update logic (called directly or after delay)."""
        with torch.no_grad():
            scores = self.compute_nonconformity_scores(logits, labels)  # (B,)

        scores_np = scores.detach().cpu().numpy()
        alpha_m = self.alphas[modality_mask_key]

        # Compute adaptive learning rate
        eta_t = self.eta_0 * (1.0 + self.beta_1 * shift_velocity
                              + self.beta_2 * disagreement)

        # Build prediction sets to check coverage
        pred_sets = self.predict_set(logits, modality_mask_key)
        labels_np = labels.detach().cpu().numpy()

        batch_hits: List[float] = []
        batch_sizes: List[float] = []

        for i in range(len(labels_np)):
            true_label = int(labels_np[i])
            hit = 1.0 if true_label in pred_sets[i] else 0.0
            miss = 1.0 - hit
            batch_hits.append(hit)
            batch_sizes.append(float(len(pred_sets[i])))

            # Append score to per-mask sliding window
            self.score_windows[modality_mask_key].append(float(scores_np[i]))

        # Average miss indicator over the batch
        avg_miss = 1.0 - float(np.mean(batch_hits))

        # ACI alpha update: alpha_new = clip(alpha + eta * (alpha_target - miss), 0, 1)
        alpha_target = self.alpha  # desired mis-coverage
        alpha_new = alpha_m + eta_t * (alpha_target - avg_miss)
        alpha_new = float(np.clip(alpha_new, 0.0, 1.0))
        self.alphas[modality_mask_key] = alpha_new

        # Record metrics
        mean_coverage = float(np.mean(batch_hits))
        mean_set_size = float(np.mean(batch_sizes))
        self._coverage_history.append(mean_coverage)
        self._set_size_history.append(mean_set_size)
        self._per_mask_coverage[modality_mask_key].append(mean_coverage)
        self._per_mask_set_size[modality_mask_key].append(mean_set_size)

    # --------------------------------------------------------------------- #
    # Delayed feedback
    # --------------------------------------------------------------------- #

    def update_delayed(self) -> None:
        """Process any items in the delay buffer whose feedback has matured.

        In realistic streaming settings, ground-truth labels may arrive with
        a fixed delay ``d``.  This method is called each step to flush entries
        whose timestamp satisfies ``t_now − t_entry ≥ d``.
        """
        self._flush_delay_buffer()

    def _flush_delay_buffer(self) -> None:
        """Flush matured entries from the delay buffer."""
        while self._delay_buffer:
            logits, labels, mask_key, sv, dis, t_entry = self._delay_buffer[0]
            if self._global_t - t_entry >= self.delay:
                self._delay_buffer.popleft()
                self._do_update(logits, labels, mask_key, sv, dis)
            else:
                break  # entries are in order; stop at first immature one

    # --------------------------------------------------------------------- #
    # Quantile helpers
    # --------------------------------------------------------------------- #

    def _get_quantile(self, modality_mask_key: str) -> float:
        """Compute the ``(1 − α_t^m)``-quantile from the sliding window.

        Falls back to ``1 − α`` when the window is empty (cold-start), which
        produces conservatively large prediction sets.

        Args:
            modality_mask_key: Mask pattern string.

        Returns:
            Quantile threshold (scalar).
        """
        window = self.score_windows[modality_mask_key]
        alpha_m = self.alphas[modality_mask_key]

        if len(window) == 0:
            # Cold start: use a conservative quantile based on score range.
            # For APS scores ∈ (0, 1], return a high threshold.
            return 1.0 - alpha_m

        scores_arr = np.array(window)
        # Finite-sample corrected quantile level: (1 − α)(1 + 1/n)
        n = len(scores_arr)
        level = (1.0 - alpha_m) * (1.0 + 1.0 / n)
        level = min(level, 1.0)  # clip to valid range

        q = float(np.quantile(scores_arr, level))
        return q

    # --------------------------------------------------------------------- #
    # Metrics
    # --------------------------------------------------------------------- #

    def get_metrics(self) -> Dict[str, Any]:
        """Return current conformal metrics.

        Returns:
            Dictionary with keys:

            - ``coverage``: Overall empirical marginal coverage.
            - ``gap``: ``|coverage − (1 − alpha)|`` (should → 0).
            - ``avg_set_size``: Mean prediction-set cardinality.
            - ``coverage_history``: Full rolling list of per-step coverages.
            - ``set_size_history``: Full rolling list of per-step set sizes.
            - ``per_mask_coverage``: ``{mask_key: mean_coverage}``.
            - ``per_mask_set_size``: ``{mask_key: mean_set_size}``.
            - ``current_alphas``: ``{mask_key: alpha_t^m}``.
            - ``window_sizes``: ``{mask_key: len(window)}``.
        """
        # Overall metrics
        if self._coverage_history:
            coverage = float(np.mean(self._coverage_history))
            avg_set = float(np.mean(self._set_size_history))
        else:
            coverage = 0.0
            avg_set = 0.0

        target_coverage = 1.0 - self.alpha
        gap = abs(coverage - target_coverage)

        # Per-mask summaries
        per_mask_cov = {
            k: float(np.mean(v)) if v else 0.0
            for k, v in self._per_mask_coverage.items()
        }
        per_mask_sz = {
            k: float(np.mean(v)) if v else 0.0
            for k, v in self._per_mask_set_size.items()
        }

        return {
            "coverage": coverage,
            "gap": gap,
            "avg_set_size": avg_set,
            "coverage_history": list(self._coverage_history),
            "set_size_history": list(self._set_size_history),
            "per_mask_coverage": per_mask_cov,
            "per_mask_set_size": per_mask_sz,
            "current_alphas": dict(self.alphas),
            "window_sizes": {
                k: len(v) for k, v in self.score_windows.items()
            },
        }

    # --------------------------------------------------------------------- #
    # Utilities
    # --------------------------------------------------------------------- #

    def reset(self) -> None:
        """Reset all adaptive state (for new evaluation runs)."""
        self.alphas = defaultdict(lambda: self.alpha)
        self.score_windows = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )
        self._delay_buffer.clear()
        self._global_t = 0
        self._coverage_history.clear()
        self._set_size_history.clear()
        self._per_mask_coverage.clear()
        self._per_mask_set_size.clear()

    def __repr__(self) -> str:
        masks = list(self.alphas.keys())
        return (
            f"OnlineACI(alpha={self.alpha}, score_fn={self.score_fn_name}, "
            f"window={self.window_size}, delay={self.delay}, "
            f"masks_tracked={len(masks)})"
        )
