"""
Evaluation Metrics, Conformal Coverage Tracking, and Profilers for Tri-SACTA.

Implements:
- Predictive: Macro-F1, Weighted-F1, Accuracy.
- Calibration: Expected Calibration Error (ECE), NLL, Brier Score.
- Conformal: Empirical Coverage, Coverage Gap (|Cov - (1-alpha)|), Rolling Coverage, Average Set Size.
- Hardware Profilers: Peak VRAM Profiler (MB), Latency Profiler (ms/sample).
"""

from __future__ import annotations

import time
import logging
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, brier_score_loss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standalone Metric Functions
# ---------------------------------------------------------------------------

def compute_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    """Compute overall classification accuracy."""
    return float(accuracy_score(labels, preds))


def compute_macro_f1(preds: np.ndarray, labels: np.ndarray, num_classes: int = 7) -> float:
    """Compute unweighted Macro-F1 score."""
    return float(f1_score(labels, preds, average="macro", zero_division=0))


def compute_weighted_f1(preds: np.ndarray, labels: np.ndarray, num_classes: int = 7) -> float:
    """Compute support-weighted F1 score."""
    return float(f1_score(labels, preds, average="weighted", zero_division=0))


def compute_nll(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute Negative Log-Likelihood."""
    return float(F.cross_entropy(logits, labels).item())


def compute_brier_score(probs: np.ndarray, labels: np.ndarray, num_classes: int = 7) -> float:
    """Compute multi-class Brier score."""
    one_hot = np.eye(num_classes)[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error (ECE) with equal-width bins."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(labels)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def compute_coverage(prediction_sets: List[List[int]], labels: List[int]) -> float:
    """Compute empirical coverage rate: fraction of samples where true label is in prediction set."""
    if not labels:
        return 0.0
    covered = sum(1 for y, s in zip(labels, prediction_sets) if y in s)
    return float(covered / len(labels))


def compute_coverage_gap(prediction_sets: List[List[int]], labels: List[int], target_coverage: float = 0.90) -> float:
    """Absolute difference between empirical coverage and target coverage."""
    return float(abs(compute_coverage(prediction_sets, labels) - target_coverage))


def compute_avg_set_size(prediction_sets: List[List[int]]) -> float:
    """Compute average cardinality of conformal prediction sets."""
    if not prediction_sets:
        return 0.0
    return float(np.mean([len(s) for s in prediction_sets]))


def compute_rolling_coverage(coverage_binary_stream: List[int], window_size: int = 100) -> List[float]:
    """Compute rolling window coverage over the streaming sequence."""
    rolling = []
    for i in range(len(coverage_binary_stream)):
        start_idx = max(0, i - window_size + 1)
        sub = coverage_binary_stream[start_idx : i + 1]
        rolling.append(float(np.mean(sub)))
    return rolling


# ---------------------------------------------------------------------------
# Metrics Tracker
# ---------------------------------------------------------------------------

class MetricsTracker:
    """Aggregates batch predictions, ground truth, and prediction sets to compute summary metrics."""

    def __init__(self, num_classes: int = 7, target_coverage: float = 0.90):
        self.num_classes = num_classes
        self.target_coverage = target_coverage
        self.reset()

    def reset(self):
        self.all_logits: List[torch.Tensor] = []
        self.all_probs: List[np.ndarray] = []
        self.all_preds: List[int] = []
        self.all_labels: List[int] = []
        self.all_sets: List[List[int]] = []
        self.coverage_binary_stream: List[int] = []

    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        prediction_sets: Optional[List[List[int]]] = None,
    ):
        probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
        preds = np.argmax(probs, axis=1)
        lbls = labels.detach().cpu().numpy()

        self.all_logits.append(logits.detach().cpu())
        self.all_probs.extend(probs)
        self.all_preds.extend(preds.tolist())
        self.all_labels.extend(lbls.tolist())

        if prediction_sets is not None:
            self.all_sets.extend(prediction_sets)
            for y, s in zip(lbls, prediction_sets):
                self.coverage_binary_stream.append(1 if y in s else 0)

    def compute(self) -> Dict[str, float]:
        preds_arr = np.array(self.all_preds)
        labels_arr = np.array(self.all_labels)
        probs_arr = np.array(self.all_probs)

        metrics = {
            "accuracy": compute_accuracy(preds_arr, labels_arr),
            "macro_f1": compute_macro_f1(preds_arr, labels_arr, self.num_classes),
            "weighted_f1": compute_weighted_f1(preds_arr, labels_arr, self.num_classes),
            "ece": compute_ece(probs_arr, labels_arr),
            "brier_score": compute_brier_score(probs_arr, labels_arr, self.num_classes),
        }

        if self.all_logits:
            cat_logits = torch.cat(self.all_logits, dim=0)
            cat_labels = torch.tensor(self.all_labels, dtype=torch.long)
            metrics["nll"] = compute_nll(cat_logits, cat_labels)

        if self.all_sets:
            metrics["empirical_coverage"] = compute_coverage(self.all_sets, self.all_labels)
            metrics["coverage_gap"] = compute_coverage_gap(self.all_sets, self.all_labels, self.target_coverage)
            metrics["avg_set_size"] = compute_avg_set_size(self.all_sets)

        return metrics


# ---------------------------------------------------------------------------
# Profilers (VRAM and Latency)
# ---------------------------------------------------------------------------

class VRAMProfiler:
    """Tracks peak and current GPU VRAM usage in MegaBytes."""

    def __init__(self, device: Optional[torch.device] = None):
        self.is_cuda = torch.cuda.is_available() and (device is None or device.type == "cuda")
        self.reset_peak()

    def reset_peak(self):
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats()

    def snapshot(self) -> Dict[str, float]:
        if not self.is_cuda:
            return {"allocated_mb": 0.0, "reserved_mb": 0.0, "peak_mb": 0.0}
        return {
            "allocated_mb": float(torch.cuda.memory_allocated() / (1024 ** 2)),
            "reserved_mb": float(torch.cuda.memory_reserved() / (1024 ** 2)),
            "peak_mb": float(torch.cuda.max_memory_allocated() / (1024 ** 2)),
        }


class LatencyProfiler:
    """Measures step and per-sample latency in milliseconds."""

    def __init__(self):
        self.start_time: float = 0.0
        self.elapsed_times: List[float] = []
        self.sample_counts: List[int] = []

    def start(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()

    def stop(self, num_samples: int = 1) -> float:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0
        self.elapsed_times.append(elapsed_ms)
        self.sample_counts.append(num_samples)
        return elapsed_ms

    def average_latency_per_sample(self) -> float:
        total_time = sum(self.elapsed_times)
        total_samples = sum(self.sample_counts)
        return float(total_time / total_samples) if total_samples > 0 else 0.0
