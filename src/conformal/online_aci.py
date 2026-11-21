"""
Mask-and-Disagreement-Driven Online Adaptive Conformal Inference (OnlineACI).

Key scientific improvements:
1. Exact finite-sample order statistic quantile:
   k = ceil((n + 1) * (1 - alpha_t^m))
   Strictly valid for conformal calibration without linear interpolation artifacts (AUD-09).
2. Safe adaptive step size:
   eta_t = clip(eta_0 * (1 + beta_1 * |delta_d| + beta_2 * max(0, kappa)), eta_min, eta_max)
   Guaranteed positive and bounded away from zero and infinity (AUD-07).
3. Immutable PredictionRecord and faithful delayed feedback:
   Prediction sets emitted at time t are frozen into an immutable record.
   When delayed feedback arrives at t + delta, miscoverage is evaluated against
   the emitted prediction set, not re-computed with future weights/thresholds (AUD-08).
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class PredictionRecord:
    """Immutable record of prediction sets emitted at test time."""
    record_id: int
    timestamp: int
    modality_mask_key: str
    prediction_sets: List[List[int]]
    logits: torch.Tensor
    shift_velocity: float
    disagreement: float


class OnlineACI:
    """Online Adaptive Conformal Inference with mask-conditional control and delayed feedback."""

    def __init__(self, num_classes: int, alpha: float, config: Dict[str, Any]):
        self.num_classes = num_classes
        self.alpha = alpha

        conf = config.get("conformal", config)
        self.eta_0: float = float(conf.get("eta_0", 0.01))
        self.beta_1: float = float(conf.get("beta_1", 0.5))
        self.beta_2: float = float(conf.get("beta_2", 0.5))
        self.eta_min: float = float(conf.get("eta_min", 0.001))
        self.eta_max: float = float(conf.get("eta_max", 0.10))
        self.window_size: int = int(conf.get("window_size", 200))
        self.delay: int = int(conf.get("delay", 0))
        self.score_fn_name: str = str(conf.get("score_fn", "aps")).lower()

        # Per-mask adaptive state
        self.alphas: Dict[str, float] = defaultdict(lambda: self.alpha)
        self.score_windows: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )
        # Global fallback window for rare masks (AUD-18)
        self.global_score_window: deque = deque(maxlen=self.window_size)
        self.min_mask_support: int = int(conf.get("min_mask_support", 10))

        # Immutable prediction records tracking
        self.records: Dict[int, PredictionRecord] = {}
        self._delay_buffer: deque = deque()
        self._global_t: int = 0

        # Rolling telemetry
        self._coverage_history: List[float] = []
        self._set_size_history: List[float] = []
        self._per_mask_coverage: Dict[str, List[float]] = defaultdict(list)
        self._per_mask_set_size: Dict[str, List[float]] = defaultdict(list)

    def compute_nonconformity_scores(
        self,
        logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute nonconformity scores (APS or LAC) in float32."""
        logits_f = logits.float()
        probs = F.softmax(logits_f, dim=-1)

        if self.score_fn_name == "aps":
            return self._aps_scores(probs, labels)
        elif self.score_fn_name == "lac":
            return self._lac_scores(probs, labels)
        else:
            raise ValueError(f"Unknown score function: '{self.score_fn_name}'")

    @staticmethod
    def _aps_scores(probs: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
        sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
        cum_probs = sorted_probs.cumsum(dim=-1)

        inv_indices = sorted_indices.argsort(dim=-1)
        cum_scores = cum_probs.gather(dim=-1, index=inv_indices)

        if labels is None:
            return cum_scores

        labels_expanded = labels.long().unsqueeze(-1)
        return cum_scores.gather(dim=-1, index=labels_expanded).squeeze(-1)

    @staticmethod
    def _lac_scores(probs: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
        if labels is None:
            return 1.0 - probs
        labels_expanded = labels.long().unsqueeze(-1)
        p_true = probs.gather(dim=-1, index=labels_expanded).squeeze(-1)
        return 1.0 - p_true

    def predict_set(
        self,
        logits: torch.Tensor,
        modality_mask_key: str,
        shift_velocity: float = 0.0,
        disagreement: float = 0.0,
    ) -> List[List[int]]:
        """Construct conformal prediction sets and record emitted state."""
        self._global_t += 1
        record_id = self._global_t

        with torch.no_grad():
            per_class_scores = self.compute_nonconformity_scores(logits, labels=None)

        q_hat = self._get_quantile(modality_mask_key)
        per_class_np = per_class_scores.detach().cpu().numpy()

        prediction_sets: List[List[int]] = []
        for i in range(per_class_np.shape[0]):
            pset = [
                int(c)
                for c in range(self.num_classes)
                if per_class_np[i, c] <= q_hat
            ]
            # Ensure non-empty set
            if len(pset) == 0:
                top1 = int(logits[i].argmax().item())
                pset = [top1]
            prediction_sets.append(sorted(pset))

        # Store immutable prediction record (AUD-08)
        self.records[record_id] = PredictionRecord(
            record_id=record_id,
            timestamp=self._global_t,
            modality_mask_key=modality_mask_key,
            prediction_sets=prediction_sets,
            logits=logits.detach().cpu(),
            shift_velocity=shift_velocity,
            disagreement=disagreement,
        )

        return prediction_sets

    def observe(self, record_id: int, labels: torch.Tensor) -> None:
        """Evaluate coverage of the emitted record and update mask significance level."""
        if record_id not in self.records:
            return

        record = self.records.pop(record_id)
        logits = record.logits
        mask_key = record.modality_mask_key
        pred_sets = record.prediction_sets
        labels_np = labels.detach().cpu().numpy().reshape(-1)

        with torch.no_grad():
            scores = self.compute_nonconformity_scores(logits, labels.cpu().view(-1))
        scores_np = scores.numpy()

        # 1. Compute empirical coverage on the ACTUALLY EMITTED prediction sets
        batch_hits: List[float] = []
        batch_sizes: List[float] = []

        for i in range(len(labels_np)):
            true_label = int(labels_np[i])
            hit = 1.0 if true_label in pred_sets[i] else 0.0
            batch_hits.append(hit)
            batch_sizes.append(float(len(pred_sets[i])))
            score_val = float(scores_np[i])
            self.score_windows[mask_key].append(score_val)
            self.global_score_window.append(score_val)

        avg_miss = 1.0 - float(np.mean(batch_hits))

        # 2. Safe adaptive step size (AUD-07)
        vel_mag = abs(record.shift_velocity)
        dis_val = max(0.0, record.disagreement)
        eta_t = self.eta_0 * (1.0 + self.beta_1 * vel_mag + self.beta_2 * dis_val)
        eta_t = float(np.clip(eta_t, self.eta_min, self.eta_max))

        # 3. Online ACI update
        alpha_m = self.alphas[mask_key]
        alpha_new = alpha_m + eta_t * (self.alpha - avg_miss)
        alpha_new = float(np.clip(alpha_new, 0.01, 0.99))
        self.alphas[mask_key] = alpha_new

        # 4. Telemetry
        mean_cov = float(np.mean(batch_hits))
        mean_sz = float(np.mean(batch_sizes))
        self._coverage_history.append(mean_cov)
        self._set_size_history.append(mean_sz)
        self._per_mask_coverage[mask_key].append(mean_cov)
        self._per_mask_set_size[mask_key].append(mean_sz)

    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        modality_mask_key: str,
        shift_velocity: float = 0.0,
        disagreement: float = 0.0,
    ) -> None:
        """Standard batch update with delay queue support (AUD-08)."""
        current_id = self._global_t

        if self.delay > 0:
            # Enqueue to delayed feedback buffer
            self._delay_buffer.append((current_id, labels.detach().cpu(), self._global_t))
            self._flush_delay_buffer()
        else:
            # Immediate feedback
            self.observe(current_id, labels)

    def _flush_delay_buffer(self) -> None:
        """Process records whose feedback has matured after delay steps."""
        while self._delay_buffer:
            rec_id, labels, t_entry = self._delay_buffer[0]
            if self._global_t - t_entry >= self.delay:
                self._delay_buffer.popleft()
                self.observe(rec_id, labels)
            else:
                break

    def _get_quantile(self, modality_mask_key: str) -> float:
        """Compute the exact finite-sample order statistic quantile with rare-mask fallback (AUD-09, AUD-18).

        k = ceil((n + 1) * (1 - alpha_t^m))
        """
        window = self.score_windows[modality_mask_key]
        alpha_m = self.alphas[modality_mask_key]
        n = len(window)

        # Fallback to global window for rare/unseen masks (AUD-18)
        if n < self.min_mask_support and len(self.global_score_window) >= self.min_mask_support:
            window_to_use = list(self.global_score_window)
            n_eff = len(window_to_use)
        elif n > 0:
            window_to_use = list(window)
            n_eff = n
        else:
            # Cold-start fallback: conservative full set
            return 1.0

        scores_arr = np.sort(np.array(window_to_use))
        # Exact finite-sample order statistic rank
        k = math.ceil((n_eff + 1) * (1.0 - alpha_m))

        if k > n_eff:
            return 1.0
        elif k < 1:
            return float(scores_arr[0])
        else:
            return float(scores_arr[k - 1])

    def get_metrics(self) -> Dict[str, Any]:
        cov = float(np.mean(self._coverage_history)) if self._coverage_history else 0.0
        sz = float(np.mean(self._set_size_history)) if self._set_size_history else 0.0
        return {
            "coverage": cov,
            "gap": abs(cov - (1.0 - self.alpha)),
            "avg_set_size": sz,
            "coverage_history": list(self._coverage_history),
            "set_size_history": list(self._set_size_history),
            "per_mask_coverage": {k: float(np.mean(v)) for k, v in self._per_mask_coverage.items()},
            "per_mask_set_size": {k: float(np.mean(v)) for k, v in self._per_mask_set_size.items()},
            "current_alphas": dict(self.alphas),
        }
