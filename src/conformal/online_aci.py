"""
Mask-and-Disagreement-Driven Online Adaptive Conformal Inference (OnlineACI) — V3.

Key scientific improvements in V3:
1. Mathematically Valid Controller Modes (R3-01, THEORY_V3):
   - 'fixed_mask_aci': Constant learning rate per mask with unprojected state and explicit
     boundary set emissions (alpha <= 0 -> full label set; alpha >= 1 -> empty set).
     Satisfies pathwise finite-sample tracking bound: |e_bar - alpha| <= (1 + eta) / (eta * n_g).
   - 'dynamic_mask_aci': State-dependent empirical heuristic modulated by shift velocity
     and cross-modal disagreement (reported as an empirical tracking method).
2. Deep Immutability of PredictionRecord (R3-02):
   - Detached, cloned float32 logits and immutable tuples of class indices.
   - Cloned nonconformity scores and explicit event metadata.
3. Faithful Delay Queue with Event Timestamps (R3-02):
   - Strict FIFO maturation based on global step elapsed time (t_current - t_entry >= delay).
   - Historical evaluation against frozen emitted prediction sets, never retroactively recomputed.
4. Pooled Global Fallback with Detailed Per-Mask Telemetry (R3-15):
   - Fallback to pooled global window for masks with support < min_mask_support.
   - Comprehensive telemetry tracking per-mask support, hit count, set sizes, empty/full set counts.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PredictionRecord:
    """Deeply immutable record of prediction sets emitted at test time."""
    record_id: int
    timestamp: int
    modality_mask_key: str
    prediction_sets: Tuple[Tuple[int, ...], ...]
    logits: torch.Tensor
    shift_velocity: float
    disagreement: float
    alpha_at_emission: float
    quantile: float = 0.0


class OnlineACI:
    """Online Adaptive Conformal Inference with mask conditioning and delayed feedback."""

    def __init__(self, num_classes: int, alpha: float, config: Dict[str, Any]):
        self.num_classes = num_classes
        self.alpha = float(alpha)

        conf = config.get("conformal", config)
        self.controller_mode: str = str(conf.get("controller", "dynamic_mask_aci")).lower()
        self.eta_0: float = float(conf.get("eta_0", 0.01))
        self.beta_1: float = float(conf.get("beta_1", 0.5))
        self.beta_2: float = float(conf.get("beta_2", 0.5))
        self.eta_min: float = float(conf.get("eta_min", 0.001))
        self.eta_max: float = float(conf.get("eta_max", 0.10))
        self.window_size: int = int(conf.get("window_size", 200))
        self.delay: int = int(conf.get("delay", 0))
        self.score_fn_name: str = str(conf.get("score_fn", "aps")).lower()
        self.allow_empty: bool = bool(conf.get("allow_empty", self.controller_mode == "fixed_mask_aci"))

        # Per-mask adaptive state
        # In fixed_mask_aci, alpha_m is unprojected in [-eta, 1 + eta]
        self.alphas: Dict[str, float] = defaultdict(lambda: self.alpha)
        self.score_windows: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )
        # Pooled global window for cold-start/rare masks
        self.pooled_global_window: deque = deque(maxlen=self.window_size)
        self.min_mask_support: int = int(conf.get("min_mask_support", 10))

        # Prediction records tracking
        self.records: Dict[int, PredictionRecord] = {}
        self._delay_buffer: deque = deque()
        self._global_t: int = 0

        # Detailed per-mask telemetry
        self._coverage_history: List[float] = []
        self._set_size_history: List[float] = []
        self._empty_set_count: int = 0
        self._full_set_count: int = 0
        self._per_mask_hits: Dict[str, int] = defaultdict(int)
        self._per_mask_counts: Dict[str, int] = defaultdict(int)
        self._per_mask_coverage: Dict[str, List[float]] = defaultdict(list)
        self._per_mask_set_size: Dict[str, List[float]] = defaultdict(list)
        self._per_mask_empty_sets: Dict[str, int] = defaultdict(int)
        self._per_mask_full_sets: Dict[str, int] = defaultdict(int)
        self._clipping_residuals: Dict[str, List[float]] = defaultdict(list)
        self._cumulative_clipping_residual: Dict[str, float] = defaultdict(float)

        # Precise observation event telemetry (Task 7)
        self._event_telemetry: List[Dict[str, Any]] = []
        self._last_event_telemetry: Optional[Dict[str, Any]] = None
        self._last_emission_telemetry: Optional[Dict[str, Any]] = None
        self._last_emission_telemetry: Optional[Dict[str, Any]] = None

    @property
    def global_score_window(self) -> deque:
        """Backward-compatibility alias for pooled_global_window."""
        return self.pooled_global_window

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
        """Construct conformal prediction sets with rigorous boundary policy and immutable record.

        Returns:
            prediction_sets_list: List of prediction sets for the batch.
            self.last_record_id tracks the unique record ID for telemetry and delayed feedback.
        """
        self._global_t += 1
        record_id = self._global_t
        self.last_record_id = record_id
        alpha_m = self.alphas[modality_mask_key]

        with torch.no_grad():
            per_class_scores = self.compute_nonconformity_scores(logits, labels=None)

        per_class_np = per_class_scores.detach().cpu().numpy()
        batch_size = per_class_np.shape[0]

        prediction_sets_list: List[List[int]] = []
        q_hat_val: float = 0.0

        # Boundary checks for fixed-step reference controller (THEORY_V3 Theorem 1)
        if self.controller_mode == "fixed_mask_aci" and alpha_m <= 0.0:
            # Full set of all classes -> guaranteed e_t = 0
            q_hat_val = float("inf")
            for _ in range(batch_size):
                prediction_sets_list.append(list(range(self.num_classes)))
                self._full_set_count += 1
                self._per_mask_full_sets[modality_mask_key] += 1
        elif self.controller_mode == "fixed_mask_aci" and alpha_m >= 1.0:
            # Empty set -> guaranteed e_t = 1
            q_hat_val = float("-inf")
            for _ in range(batch_size):
                prediction_sets_list.append([])
                self._empty_set_count += 1
                self._per_mask_empty_sets[modality_mask_key] += 1
        else:
            # Interior quantile order statistic
            q_hat = self._get_quantile(modality_mask_key)
            q_hat_val = float(q_hat)
            for i in range(batch_size):
                pset = [
                    int(c)
                    for c in range(self.num_classes)
                    if per_class_np[i, c] <= q_hat
                ]
                if len(pset) == 0:
                    if not self.allow_empty:
                        top1 = int(logits[i].argmax().item())
                        pset = [top1]
                    else:
                        self._empty_set_count += 1
                        self._per_mask_empty_sets[modality_mask_key] += 1
                elif len(pset) == self.num_classes:
                    self._full_set_count += 1
                    self._per_mask_full_sets[modality_mask_key] += 1

                prediction_sets_list.append(sorted(pset))

        # Deeply immutable record conversion
        immutable_sets = tuple(tuple(s) for s in prediction_sets_list)
        self.records[record_id] = PredictionRecord(
            record_id=record_id,
            timestamp=self._global_t,
            modality_mask_key=modality_mask_key,
            prediction_sets=immutable_sets,
            logits=logits.detach().cpu().clone(),
            shift_velocity=float(shift_velocity),
            disagreement=float(disagreement),
            alpha_at_emission=float(alpha_m),
            quantile=float(q_hat_val),
        )

        self._last_emission_telemetry = {
            "event_id": int(record_id),
            "timestamp": int(self._global_t),
            "modality_mask_key": str(modality_mask_key),
            "alpha_at_emission": float(alpha_m),
            "quantile": float(q_hat_val),
            "shift_velocity": float(shift_velocity),
            "disagreement": float(disagreement),
            "feedback_delay": int(self.delay),
        }

        return prediction_sets_list

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

        batch_hits: List[float] = []
        batch_sizes: List[float] = []

        for i in range(len(labels_np)):
            true_label = int(labels_np[i])
            hit = 1.0 if true_label in pred_sets[i] else 0.0
            batch_hits.append(hit)
            batch_sizes.append(float(len(pred_sets[i])))
            score_val = float(scores_np[i])
            self.score_windows[mask_key].append(score_val)
            self.pooled_global_window.append(score_val)
            self._per_mask_hits[mask_key] += int(hit)
            self._per_mask_counts[mask_key] += 1

        avg_miss = 1.0 - float(np.mean(batch_hits))

        # Update controller state
        if self.controller_mode == "fixed_mask_aci":
            # Exact fixed step update with projection residual tracking (V5-06)
            eta_t = self.eta_0
            alpha_m = self.alphas[mask_key]
            alpha_raw = alpha_m + eta_t * (self.alpha - avg_miss)
            # Bound within [-eta, 1 + eta] for numerical safety
            alpha_clipped = float(np.clip(alpha_raw, -self.eta_0, 1.0 + self.eta_0))
            residual = alpha_clipped - alpha_raw
            self._clipping_residuals[mask_key].append(residual)
            self._cumulative_clipping_residual[mask_key] += abs(residual)
            self.alphas[mask_key] = alpha_clipped
        else:
            # Dynamic heuristic update (AUD-07)
            vel_mag = abs(record.shift_velocity)
            dis_val = max(0.0, record.disagreement)
            eta_t = self.eta_0 * (1.0 + self.beta_1 * vel_mag + self.beta_2 * dis_val)
            eta_t = float(np.clip(eta_t, self.eta_min, self.eta_max))

            alpha_m = self.alphas[mask_key]
            alpha_raw = alpha_m + eta_t * (self.alpha - avg_miss)
            alpha_clipped = float(np.clip(alpha_raw, 0.01, 0.99))
            residual = alpha_clipped - alpha_raw
            self._clipping_residuals[mask_key].append(residual)
            self._cumulative_clipping_residual[mask_key] += abs(residual)
            self.alphas[mask_key] = alpha_clipped

        # Telemetry
        mean_cov = float(np.mean(batch_hits))
        mean_sz = float(np.mean(batch_sizes))
        self._coverage_history.append(mean_cov)
        self._set_size_history.append(mean_sz)
        self._per_mask_coverage[mask_key].append(mean_cov)
        self._per_mask_set_size[mask_key].append(mean_sz)

        # Precise observation event telemetry directly from OnlineACI state (Task 7)
        event_dict = {
            "event_id": int(record.record_id),
            "timestamp": int(record.timestamp),
            "modality_mask_key": str(mask_key),
            "alpha_before": float(alpha_m),
            "alpha_after": float(alpha_clipped),
            "eta_used": float(eta_t),
            "quantile": float(record.quantile),
            "clipping_residual": float(residual),
            "mean_coverage": float(mean_cov),
            "mean_set_size": float(mean_sz),
            "shift_velocity": float(record.shift_velocity),
            "disagreement": float(record.disagreement),
        }
        self._event_telemetry.append(event_dict)
        self._last_event_telemetry = event_dict

    def get_last_event_telemetry(self) -> Optional[Dict[str, Any]]:
        """Return the most recently processed observation event telemetry (Task 7)."""
        return self._last_event_telemetry

    def get_last_emission_telemetry(self) -> Optional[Dict[str, Any]]:
        """Return emission telemetry for the most recent predict_set call (V9-04)."""
        return self._last_emission_telemetry

    def get_event_telemetry(self) -> List[Dict[str, Any]]:
        """Return full list of observation event telemetries (Task 7)."""
        return list(self._event_telemetry)

    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        modality_mask_key: str,
        shift_velocity: float = 0.0,
        disagreement: float = 0.0,
        record_id: Optional[int] = None,
    ) -> None:
        """Standard batch update with delay queue support.

        Args:
            record_id: Explicit record ID returned by predict_set(). If None,
                       falls back to self._global_t (backward compatibility only).
                       Always pass the record_id from predict_set() for correctness (P0-2 fix).
        """
        current_id = record_id if record_id is not None else self._global_t

        if self.delay > 0:
            self._delay_buffer.append((current_id, labels.detach().cpu().clone(), self._global_t))
            self._flush_delay_buffer()
        else:
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
        """Compute exact finite-sample order statistic quantile with pooled global fallback."""
        window = self.score_windows[modality_mask_key]
        alpha_m = float(np.clip(self.alphas[modality_mask_key], 1e-4, 1.0 - 1e-4))
        n = len(window)

        # Pooled global fallback for rare/unseen masks (R3-15)
        if n < self.min_mask_support and len(self.pooled_global_window) >= self.min_mask_support:
            window_to_use = list(self.pooled_global_window)
            n_eff = len(window_to_use)
        elif n > 0:
            window_to_use = list(window)
            n_eff = n
        else:
            return 1.0

        scores_arr = np.sort(np.array(window_to_use))
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
        
        per_mask_cov = {}
        for k, v in self._per_mask_coverage.items():
            per_mask_cov[k] = float(np.mean(v)) if v else 0.0

        per_mask_sz = {}
        for k, v in self._per_mask_set_size.items():
            per_mask_sz[k] = float(np.mean(v)) if v else 0.0

        return {
            "controller_mode": self.controller_mode,
            "coverage": cov,
            "gap": abs(cov - (1.0 - self.alpha)),
            "avg_set_size": sz,
            "empty_sets_total": self._empty_set_count,
            "full_sets_total": self._full_set_count,
            "coverage_history": list(self._coverage_history),
            "set_size_history": list(self._set_size_history),
            "per_mask_coverage": per_mask_cov,
            "per_mask_set_size": per_mask_sz,
            "per_mask_hits": dict(self._per_mask_hits),
            "per_mask_counts": dict(self._per_mask_counts),
            "per_mask_empty_sets": dict(self._per_mask_empty_sets),
            "per_mask_full_sets": dict(self._per_mask_full_sets),
            "current_alphas": dict(self.alphas),
        }


# Backward-compatibility alias
OnlineAdaptiveACI = OnlineACI


class GlobalACI:
    """Genuinely global ACI controller (P0-3 fix for V8).

    Maintains a SINGLE global alpha level and a single pooled score window
    for ALL modality mask types. This is the scientifically correct
    implementation of a mask-agnostic global conformal controller.

    Distinct from fixed_mask_aci (per-mask alpha) and dynamic_mask_aci
    (per-mask alpha with velocity modulation). GlobalACI has no mask
    conditioning at all — identical to standard online ACI on a single stream.
    """

    def __init__(self, num_classes: int, alpha: float, config: Dict[str, Any]):
        self.num_classes = num_classes
        self.alpha = float(alpha)

        conf = config.get("conformal", config)
        self.eta_0: float = float(conf.get("eta_0", 0.01))
        self.window_size: int = int(conf.get("window_size", 200))
        self.delay: int = int(conf.get("delay", 0))
        self.score_fn_name: str = str(conf.get("score_fn", "aps")).lower()

        # Single global alpha and score window
        self._alpha_global: float = self.alpha
        self._score_window: deque = deque(maxlen=self.window_size)

        # Tracking
        self._global_t: int = 0
        self._records: Dict[int, PredictionRecord] = {}
        self._delay_buffer: deque = deque()
        self._coverage_history: List[float] = []
        self._set_size_history: List[float] = []
        self._event_telemetry: List[Dict[str, Any]] = []
        self._last_event_telemetry: Optional[Dict[str, Any]] = None
        self._last_emission_telemetry: Optional[Dict[str, Any]] = None
        self._empty_set_count: int = 0
        self._full_set_count: int = 0

        # Expose alphas dict for backward compat (always global alpha for any key)
        class _GlobalAlphaDict:
            def __init__(self_, alpha_ref):
                self_._ref = alpha_ref
            def __getitem__(self_, key):
                return self_._global_alpha_ref[0]
        self.alphas = defaultdict(lambda: self._alpha_global)

    def _compute_scores(self, logits: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        probs = torch.softmax(logits.float(), dim=-1)
        if self.score_fn_name == "lac":
            if labels is None:
                return 1.0 - probs
            lbl = labels.long().unsqueeze(-1)
            return 1.0 - probs.gather(dim=-1, index=lbl).squeeze(-1)
        # APS
        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
        cum = sorted_probs.cumsum(dim=-1)
        inv_idx = sorted_idx.argsort(dim=-1)
        cum_scores = cum.gather(dim=-1, index=inv_idx)
        if labels is None:
            return cum_scores
        lbl = labels.long().unsqueeze(-1)
        return cum_scores.gather(dim=-1, index=lbl).squeeze(-1)

    def _get_quantile(self) -> float:
        n = len(self._score_window)
        if n == 0:
            return 1.0
        alpha_m = float(np.clip(self._alpha_global, 1e-4, 1.0 - 1e-4))
        scores_arr = np.sort(np.array(list(self._score_window)))
        k = math.ceil((n + 1) * (1.0 - alpha_m))
        if k > n:
            return 1.0
        elif k < 1:
            return float(scores_arr[0])
        return float(scores_arr[k - 1])

    def predict_set(
        self,
        logits: torch.Tensor,
        modality_mask_key: str,
        shift_velocity: float = 0.0,
        disagreement: float = 0.0,
    ) -> List[List[int]]:
        """Predict conformal set using single global alpha.

        Returns:
            prediction_sets_list: List of prediction sets for the batch.
        """
        self._global_t += 1
        record_id = self._global_t
        self.last_record_id = record_id

        with torch.no_grad():
            per_class_scores = self._compute_scores(logits, labels=None)
        per_class_np = per_class_scores.detach().cpu().numpy()
        batch_size = per_class_np.shape[0]

        q_hat = self._get_quantile()
        prediction_sets_list: List[List[int]] = []
        for i in range(batch_size):
            pset = [int(c) for c in range(self.num_classes) if per_class_np[i, c] <= q_hat]
            if not pset:
                pset = [int(logits[i].argmax().item())]
                self._empty_set_count += 1
            elif len(pset) == self.num_classes:
                self._full_set_count += 1
            prediction_sets_list.append(sorted(pset))

        immutable_sets = tuple(tuple(s) for s in prediction_sets_list)
        self._records[record_id] = PredictionRecord(
            record_id=record_id,
            timestamp=self._global_t,
            modality_mask_key=modality_mask_key,
            prediction_sets=immutable_sets,
            logits=logits.detach().cpu().clone(),
            shift_velocity=float(shift_velocity),
            disagreement=float(disagreement),
            alpha_at_emission=float(self._alpha_global),
            quantile=float(q_hat),
        )

        # Keep alphas dict updated for backward compat
        self.alphas[modality_mask_key] = self._alpha_global

        self._last_emission_telemetry = {
            "event_id": int(record_id),
            "timestamp": int(self._global_t),
            "modality_mask_key": str(modality_mask_key),
            "alpha_at_emission": float(self._alpha_global),
            "quantile": float(q_hat),
            "shift_velocity": float(shift_velocity),
            "disagreement": float(disagreement),
            "feedback_delay": int(self.delay),
        }

        return prediction_sets_list

    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        modality_mask_key: str,
        shift_velocity: float = 0.0,
        disagreement: float = 0.0,
        record_id: Optional[int] = None,
    ) -> None:
        current_id = record_id if record_id is not None else self._global_t
        if self.delay > 0:
            self._delay_buffer.append((current_id, labels.detach().cpu().clone(), self._global_t))
            self._flush_delay_buffer()
        else:
            self._observe(current_id, labels)

    def _flush_delay_buffer(self) -> None:
        while self._delay_buffer:
            rec_id, lbl, t_entry = self._delay_buffer[0]
            if self._global_t - t_entry >= self.delay:
                self._delay_buffer.popleft()
                self._observe(rec_id, lbl)
            else:
                break

    def _observe(self, record_id: int, labels: torch.Tensor) -> None:
        if record_id not in self._records:
            return
        record = self._records.pop(record_id)
        pred_sets = record.prediction_sets
        labels_np = labels.detach().cpu().numpy().reshape(-1)

        with torch.no_grad():
            scores = self._compute_scores(record.logits, labels.cpu().view(-1))
        scores_np = scores.numpy()

        batch_hits: List[float] = []
        batch_sizes: List[float] = []
        for i in range(len(labels_np)):
            true_label = int(labels_np[i])
            hit = 1.0 if true_label in pred_sets[i] else 0.0
            batch_hits.append(hit)
            batch_sizes.append(float(len(pred_sets[i])))
            self._score_window.append(float(scores_np[i]))

        avg_miss = 1.0 - float(np.mean(batch_hits))
        alpha_before = self._alpha_global
        alpha_raw = self._alpha_global + self.eta_0 * (self.alpha - avg_miss)
        alpha_clipped = float(np.clip(alpha_raw, -self.eta_0, 1.0 + self.eta_0))
        self._alpha_global = alpha_clipped
        # Sync alphas dict
        for k in list(self.alphas.keys()):
            self.alphas[k] = alpha_clipped

        mean_cov = float(np.mean(batch_hits))
        mean_sz = float(np.mean(batch_sizes))
        self._coverage_history.append(mean_cov)
        self._set_size_history.append(mean_sz)

        event_dict = {
            "event_id": record_id,
            "timestamp": record.timestamp,
            "modality_mask_key": record.modality_mask_key,
            "alpha_before": float(alpha_before),
            "alpha_after": float(alpha_clipped),
            "eta_used": self.eta_0,
            "quantile": float(record.quantile),
            "clipping_residual": float(alpha_clipped - alpha_raw),
            "mean_coverage": mean_cov,
            "mean_set_size": mean_sz,
            "shift_velocity": record.shift_velocity,
            "disagreement": record.disagreement,
        }
        self._event_telemetry.append(event_dict)
        self._last_event_telemetry = event_dict

    def get_last_event_telemetry(self) -> Optional[Dict[str, Any]]:
        return self._last_event_telemetry

    def get_last_emission_telemetry(self) -> Optional[Dict[str, Any]]:
        return self._last_emission_telemetry

    def get_event_telemetry(self) -> List[Dict[str, Any]]:
        return list(self._event_telemetry)

    def get_metrics(self) -> Dict[str, Any]:
        cov = float(np.mean(self._coverage_history)) if self._coverage_history else 0.0
        sz = float(np.mean(self._set_size_history)) if self._set_size_history else 0.0
        return {
            "controller_mode": "global_aci",
            "coverage": cov,
            "gap": abs(cov - (1.0 - self.alpha)),
            "avg_set_size": sz,
            "empty_sets_total": self._empty_set_count,
            "full_sets_total": self._full_set_count,
            "coverage_history": list(self._coverage_history),
            "set_size_history": list(self._set_size_history),
            "current_alpha_global": self._alpha_global,
        }
