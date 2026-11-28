"""
src/service/reliability.py: Distributional drift and conformal uncertainty quantification.
Replaces spurious 'spectral' terminology with mathematically accurate StandardizedCentroidDistance,
and implements true Split Conformal Prediction calibrated on CMU-MOSEI 'cal' split.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F


CLASS_NAMES = ["negative", "neutral", "positive"]


class StandardizedCentroidDistance:
    """Computes normalized mean absolute deviation from clean training distribution.
    
    Formula:
        d_std(x) = (1 / D) * sum_{d=1}^D (|x_d - mu_d| / (sigma_d + eps))
    
    NOTE: This is a standardized centroid distance metric measuring feature drift,
    NOT an eigenvalue/spectral decomposition method.
    """

    def __init__(self, mean: Union[torch.Tensor, np.ndarray], std: Union[torch.Tensor, np.ndarray], eps: float = 1e-6):
        if isinstance(mean, np.ndarray):
            mean = torch.from_numpy(mean).float()
        if isinstance(std, np.ndarray):
            std = torch.from_numpy(std).float()

        self.mean = mean.clone().detach().float().view(-1)
        self.std = std.clone().detach().float().view(-1) + eps
        self.dim = self.mean.numel()

    def to(self, device: torch.device) -> StandardizedCentroidDistance:
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def compute(self, x: torch.Tensor) -> float:
        """Computes standardized centroid distance for a 1D or [1, D] tensor."""
        x_flat = x.view(-1)
        if x_flat.numel() != self.dim:
            raise ValueError(f"Feature dim mismatch: expected {self.dim}, got {x_flat.numel()}")
        norm_diff = torch.abs(x_flat - self.mean) / self.std
        return float(torch.mean(norm_diff).item())


class SplitConformalPredictor:
    """True Inductive/Split Conformal Prediction Engine for multi-class classification.
    
    Calibrated on the designated 250-sample 'cal' split.
    Nonconformity score: s_i = 1 - p(y_i)
    Quantile level: ceil((N + 1) * (1 - alpha)) / N
    Prediction set: C(x) = {y in {0, 1, 2} : 1 - p(y) <= q_hat}
    """

    def __init__(self, alpha: float = 0.10):
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.calibrated_quantile: Optional[float] = None
        self.num_calibration_samples: int = 0
        self.calibration_scores: Optional[np.ndarray] = None

    def fit_from_probabilities(self, cal_probs: np.ndarray, cal_labels: np.ndarray) -> float:
        """Calibrates conformal quantile from 2D probability matrix [N, C] and true labels [N]."""
        N = len(cal_labels)
        if N == 0:
            raise ValueError("Calibration set must be non-empty.")
        if cal_probs.shape[0] != N:
            raise ValueError(f"Shape mismatch: cal_probs {cal_probs.shape[0]} vs labels {N}")

        # Compute nonconformity: s_i = 1 - p(y_i)
        true_probs = cal_probs[np.arange(N), cal_labels]
        scores = 1.0 - true_probs
        self.calibration_scores = np.sort(scores)
        self.num_calibration_samples = N

        # Standard conformal quantile: ceil((N + 1) * (1 - alpha)) / N
        q_level = min(1.0, math.ceil((N + 1) * (1.0 - self.alpha)) / N)
        self.calibrated_quantile = float(np.quantile(self.calibration_scores, q_level, method="higher"))
        return self.calibrated_quantile

    def predict_set(self, probs: Union[np.ndarray, List[float]]) -> Tuple[List[str], List[int], float, bool]:
        """Given class probabilities, constructs conformal prediction set C(x).
        
        Returns:
            (set_names, set_indices, nonconformity_of_argmax, should_abstain)
        """
        if self.calibrated_quantile is None:
            raise RuntimeError("Conformal predictor must be calibrated before predict_set(). Call fit_from_probabilities().")

        if isinstance(probs, list):
            probs = np.array(probs, dtype=np.float32)

        pred_class = int(np.argmax(probs))
        nonconformity_argmax = float(1.0 - probs[pred_class])

        # Prediction set C(x) = {y : 1 - p(y) <= q_hat}
        nonconformities = 1.0 - probs
        in_set = nonconformities <= self.calibrated_quantile
        set_indices = [i for i, included in enumerate(in_set) if included]
        set_names = [CLASS_NAMES[i] for i in set_indices]

        # Abstain if prediction set is empty (extreme OOD) or ambiguous (|C(x)| > 1)
        should_abstain = (len(set_indices) != 1)

        return set_names, set_indices, nonconformity_argmax, should_abstain

    def evaluate_coverage_and_size(
        self,
        eval_probs: np.ndarray,
        eval_labels: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluates empirical coverage and mean set size on a test/validation split."""
        if self.calibrated_quantile is None:
            raise RuntimeError("Conformal predictor not calibrated.")

        N = len(eval_labels)
        covered_count = 0
        set_sizes = []
        singleton_count = 0
        empty_count = 0

        for i in range(N):
            p = eval_probs[i]
            y = eval_labels[i]
            _, set_indices, _, _ = self.predict_set(p)
            size = len(set_indices)
            set_sizes.append(size)
            if y in set_indices:
                covered_count += 1
            if size == 1:
                singleton_count += 1
            elif size == 0:
                empty_count += 1

        return {
            "empirical_coverage": float(covered_count / N),
            "target_coverage": float(1.0 - self.alpha),
            "mean_set_size": float(np.mean(set_sizes)),
            "singleton_rate": float(singleton_count / N),
            "empty_rate": float(empty_count / N),
            "calibrated_quantile": float(self.calibrated_quantile),
            "num_samples": N,
        }
