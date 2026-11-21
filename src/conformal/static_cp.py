"""
Baseline (static / offline) conformal prediction methods.

Provides two baselines for comparison against the online adaptive approach
(:class:`OnlineACI`):

1. **SplitConformalPredictor** – Standard split conformal prediction.
   Calibrates a quantile threshold on a held-out calibration set, then
   applies it uniformly at test time.

2. **MaskConditionalCP** – Mask-Conditional Conformal Framework (MCCF).
   Performs separate calibration for each observed modality-mask pattern,
   yielding mask-specific quantile thresholds.  Falls back to the marginal
   (global) quantile for mask patterns unseen during calibration.

Both default to APS (Adaptive Prediction Sets) nonconformity scores and
gracefully handle edge cases such as too few calibration samples.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ========================================================================= #
# Shared scoring utilities
# ========================================================================= #

def _softmax_np(logits: np.ndarray) -> np.ndarray:
    """Numerically-stable softmax over the last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum(axis=-1, keepdims=True)


def _compute_aps_scores(
    probs: np.ndarray, labels: Optional[np.ndarray] = None
) -> np.ndarray:
    """APS nonconformity scores (NumPy).

    Args:
        probs: ``(N, C)`` probability matrix.
        labels: ``(N,)`` integer ground-truth labels.  When *None*, returns
            the full ``(N, C)`` per-class cumulative score matrix.

    Returns:
        ``(N,)`` if labels given, ``(N, C)`` otherwise.
    """
    N, C = probs.shape
    # Sort descending
    sorted_idx = np.argsort(-probs, axis=-1)
    sorted_probs = np.take_along_axis(probs, sorted_idx, axis=-1)
    cum_probs = np.cumsum(sorted_probs, axis=-1)  # (N, C)

    # Map back to original class order
    inv_idx = np.argsort(sorted_idx, axis=-1)
    cum_scores = np.take_along_axis(cum_probs, inv_idx, axis=-1)  # (N, C)

    if labels is None:
        return cum_scores

    rows = np.arange(N)
    return cum_scores[rows, labels.astype(int)]  # (N,)


def _compute_lac_scores(
    probs: np.ndarray, labels: Optional[np.ndarray] = None
) -> np.ndarray:
    """LAC nonconformity scores: ``1 − p̂(y | x)``."""
    if labels is None:
        return 1.0 - probs  # (N, C)
    rows = np.arange(probs.shape[0])
    return 1.0 - probs[rows, labels.astype(int)]  # (N,)


_SCORE_FNS = {
    "aps": _compute_aps_scores,
    "lac": _compute_lac_scores,
}


def _get_score_fn(name: str):
    """Retrieve a scoring function by name."""
    name = name.lower()
    if name not in _SCORE_FNS:
        raise ValueError(
            f"Unknown score function '{name}'. Supported: {list(_SCORE_FNS.keys())}"
        )
    return _SCORE_FNS[name]


# ========================================================================= #
# 1. SplitConformalPredictor
# ========================================================================= #

class SplitConformalPredictor:
    """Standard split (inductive) conformal predictor.

    **Workflow**:
        1. Train the model on the training split.
        2. Call :meth:`calibrate` with held-out calibration logits & labels.
        3. Call :meth:`predict_set` on new test logits.
        4. Call :meth:`evaluate` to compute coverage & set-size metrics.

    Args:
        num_classes: Number of target classes.
        alpha: Target mis-coverage rate (e.g. 0.1 → 90 % coverage guarantee).
        score_fn: Scoring function name (``'aps'`` or ``'lac'``).
    """

    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.1,
        score_fn: str = "aps",
    ):
        self.num_classes = num_classes
        self.alpha = alpha
        self.score_fn_name = score_fn
        self._score_fn = _get_score_fn(score_fn)
        self.q_hat: Optional[float] = None
        self._cal_scores: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # Calibration
    # ------------------------------------------------------------------ #

    def calibrate(
        self,
        logits: torch.Tensor | np.ndarray,
        labels: torch.Tensor | np.ndarray,
    ) -> float:
        """Calibrate the quantile threshold on a held-out set.

        Args:
            logits: ``(N_cal, C)`` raw model outputs.
            labels: ``(N_cal,)`` ground-truth class labels.

        Returns:
            The calibrated quantile threshold ``q̂``.

        Raises:
            ValueError: If the calibration set has fewer than 2 samples.
        """
        logits_np = self._to_numpy(logits)
        labels_np = self._to_numpy(labels).astype(int)

        n_cal = logits_np.shape[0]
        if n_cal < 2:
            raise ValueError(
                f"Calibration set too small ({n_cal} samples). "
                "Need at least 2 for a meaningful quantile."
            )

        probs = _softmax_np(logits_np)
        self._cal_scores = self._score_fn(probs, labels_np)  # (N_cal,)

        # Finite-sample corrected quantile: ceil((n+1)(1−α)) / n
        level = np.ceil((n_cal + 1) * (1.0 - self.alpha)) / n_cal
        level = min(level, 1.0)
        self.q_hat = float(np.quantile(self._cal_scores, level))
        return self.q_hat

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #

    def predict_set(
        self, logits: torch.Tensor | np.ndarray
    ) -> List[List[int]]:
        """Construct conformal prediction sets using the calibrated threshold.

        Args:
            logits: ``(B, C)`` raw model outputs for test samples.

        Returns:
            List of length ``B``, each a sorted list of included class indices.

        Raises:
            RuntimeError: If :meth:`calibrate` has not been called.
        """
        if self.q_hat is None:
            raise RuntimeError(
                "Must call calibrate() before predict_set()."
            )

        logits_np = self._to_numpy(logits)
        probs = _softmax_np(logits_np)
        per_class_scores = self._score_fn(probs, labels=None)  # (B, C)

        prediction_sets: List[List[int]] = []
        for i in range(per_class_scores.shape[0]):
            pset = [
                int(c)
                for c in range(self.num_classes)
                if per_class_scores[i, c] <= self.q_hat
            ]
            # Guarantee non-empty: include argmax
            if len(pset) == 0:
                pset = [int(np.argmax(probs[i]))]
            prediction_sets.append(sorted(pset))
        return prediction_sets

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        logits: torch.Tensor | np.ndarray,
        labels: torch.Tensor | np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate conformal coverage and set size on test data.

        Args:
            logits: ``(N, C)`` raw outputs.
            labels: ``(N,)`` ground-truth labels.

        Returns:
            Dictionary with ``coverage``, ``gap``, ``avg_set_size``,
            ``median_set_size``.
        """
        pred_sets = self.predict_set(logits)
        labels_np = self._to_numpy(labels).astype(int)

        hits = [
            1.0 if int(labels_np[i]) in pred_sets[i] else 0.0
            for i in range(len(labels_np))
        ]
        sizes = [float(len(s)) for s in pred_sets]

        coverage = float(np.mean(hits))
        target = 1.0 - self.alpha
        return {
            "coverage": coverage,
            "gap": abs(coverage - target),
            "avg_set_size": float(np.mean(sizes)),
            "median_set_size": float(np.median(sizes)),
            "q_hat": self.q_hat,
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().float().numpy()
        return np.asarray(x, dtype=np.float64)

    def __repr__(self) -> str:
        cal = "calibrated" if self.q_hat is not None else "uncalibrated"
        return (
            f"SplitConformalPredictor(alpha={self.alpha}, "
            f"score_fn={self.score_fn_name}, {cal})"
        )


# ========================================================================= #
# 2. MaskConditionalCP  (MCCF)
# ========================================================================= #

class MaskConditionalCP:
    """Mask-Conditional Conformal Framework (MCCF).

    Performs **separate calibration** for each modality-mask pattern observed
    during calibration, yielding pattern-specific quantile thresholds.

    At test time, if a mask pattern was not seen during calibration, the
    predictor falls back to the **marginal** (all-data) quantile so that
    coverage is never undefined.

    Args:
        num_classes: Number of target classes.
        alpha: Target mis-coverage rate.
        score_fn: Scoring function name (``'aps'`` or ``'lac'``).
        min_cal_samples: Minimum calibration-set size per mask to trust the
            conditional quantile; patterns with fewer samples fall back to
            the marginal quantile.
    """

    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.1,
        score_fn: str = "aps",
        min_cal_samples: int = 10,
    ):
        self.num_classes = num_classes
        self.alpha = alpha
        self.score_fn_name = score_fn
        self._score_fn = _get_score_fn(score_fn)
        self.min_cal_samples = min_cal_samples

        # Calibrated quantiles: {mask_key: q_hat}
        self.q_hats: Dict[str, float] = {}
        self.q_hat_marginal: Optional[float] = None
        # Track calibration sizes for diagnostics
        self._cal_counts: Dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # Calibration
    # ------------------------------------------------------------------ #

    def calibrate(
        self,
        logits: torch.Tensor | np.ndarray,
        labels: torch.Tensor | np.ndarray,
        mask_keys: List[str],
    ) -> Dict[str, float]:
        """Calibrate per-mask quantile thresholds.

        Args:
            logits: ``(N_cal, C)`` raw model outputs.
            labels: ``(N_cal,)`` ground-truth labels.
            mask_keys: Length-``N_cal`` list of mask-pattern strings (e.g.
                ``"1110"`` for visual+audio+text present, prosody missing).

        Returns:
            Dictionary ``{mask_key: q_hat, '__marginal__': q_hat_marginal}``.
        """
        logits_np = self._to_numpy(logits)
        labels_np = self._to_numpy(labels).astype(int)
        probs = _softmax_np(logits_np)
        all_scores = self._score_fn(probs, labels_np)  # (N_cal,)

        # Marginal quantile (over all masks)
        n_total = len(all_scores)
        if n_total < 2:
            raise ValueError(
                f"Calibration set too small ({n_total} samples)."
            )
        level_marginal = np.ceil((n_total + 1) * (1.0 - self.alpha)) / n_total
        level_marginal = min(level_marginal, 1.0)
        self.q_hat_marginal = float(np.quantile(all_scores, level_marginal))

        # Group by mask pattern
        mask_groups: Dict[str, List[int]] = defaultdict(list)
        for i, mk in enumerate(mask_keys):
            mask_groups[mk].append(i)

        self.q_hats.clear()
        self._cal_counts.clear()

        for mk, indices in mask_groups.items():
            n_mk = len(indices)
            self._cal_counts[mk] = n_mk
            if n_mk < self.min_cal_samples:
                # Too few samples → fall back to marginal
                self.q_hats[mk] = self.q_hat_marginal
                continue

            scores_mk = all_scores[np.array(indices)]
            level = np.ceil((n_mk + 1) * (1.0 - self.alpha)) / n_mk
            level = min(level, 1.0)
            self.q_hats[mk] = float(np.quantile(scores_mk, level))

        result = dict(self.q_hats)
        result["__marginal__"] = self.q_hat_marginal
        return result

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #

    def predict_set(
        self,
        logits: torch.Tensor | np.ndarray,
        mask_keys: List[str],
    ) -> List[List[int]]:
        """Construct prediction sets using mask-conditional quantiles.

        Args:
            logits: ``(B, C)`` raw model outputs.
            mask_keys: Length-``B`` list of modality-mask patterns.

        Returns:
            List of length ``B``, each a sorted list of included class indices.

        Raises:
            RuntimeError: If :meth:`calibrate` has not been called.
        """
        if self.q_hat_marginal is None:
            raise RuntimeError("Must call calibrate() before predict_set().")

        logits_np = self._to_numpy(logits)
        probs = _softmax_np(logits_np)
        per_class_scores = self._score_fn(probs, labels=None)  # (B, C)

        prediction_sets: List[List[int]] = []
        for i in range(per_class_scores.shape[0]):
            mk = mask_keys[i]
            q = self.q_hats.get(mk, self.q_hat_marginal)
            pset = [
                int(c)
                for c in range(self.num_classes)
                if per_class_scores[i, c] <= q
            ]
            if len(pset) == 0:
                pset = [int(np.argmax(probs[i]))]
            prediction_sets.append(sorted(pset))
        return prediction_sets

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        logits: torch.Tensor | np.ndarray,
        labels: torch.Tensor | np.ndarray,
        mask_keys: List[str],
    ) -> Dict[str, Any]:
        """Evaluate coverage and set size, both marginal and per-mask.

        Args:
            logits: ``(N, C)`` raw outputs.
            labels: ``(N,)`` ground-truth labels.
            mask_keys: Length-``N`` list of mask patterns.

        Returns:
            Dictionary with ``coverage``, ``gap``, ``avg_set_size``,
            ``per_mask_coverage``, ``per_mask_set_size``.
        """
        pred_sets = self.predict_set(logits, mask_keys)
        labels_np = self._to_numpy(labels).astype(int)

        hits = []
        sizes = []
        per_mask_hits: Dict[str, List[float]] = defaultdict(list)
        per_mask_sizes: Dict[str, List[float]] = defaultdict(list)

        for i in range(len(labels_np)):
            h = 1.0 if int(labels_np[i]) in pred_sets[i] else 0.0
            s = float(len(pred_sets[i]))
            mk = mask_keys[i]
            hits.append(h)
            sizes.append(s)
            per_mask_hits[mk].append(h)
            per_mask_sizes[mk].append(s)

        coverage = float(np.mean(hits))
        target = 1.0 - self.alpha

        return {
            "coverage": coverage,
            "gap": abs(coverage - target),
            "avg_set_size": float(np.mean(sizes)),
            "median_set_size": float(np.median(sizes)),
            "per_mask_coverage": {
                k: float(np.mean(v)) for k, v in per_mask_hits.items()
            },
            "per_mask_set_size": {
                k: float(np.mean(v)) for k, v in per_mask_sizes.items()
            },
            "q_hats": dict(self.q_hats),
            "q_hat_marginal": self.q_hat_marginal,
            "cal_counts": dict(self._cal_counts),
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().float().numpy()
        return np.asarray(x, dtype=np.float64)

    def __repr__(self) -> str:
        cal = "calibrated" if self.q_hat_marginal is not None else "uncalibrated"
        n_masks = len(self.q_hats)
        return (
            f"MaskConditionalCP(alpha={self.alpha}, "
            f"score_fn={self.score_fn_name}, "
            f"masks={n_masks}, {cal})"
        )
