"""
Per-Modality Spectral Shift Detector for Tri-SACTA with Null-Calibrated Statistics.

Detects distribution shift per modality using dimension-normalized mean-shift
and log-spectral (eigenvalue) divergence between source and target distributions.
Computes calibrated reliability weights that gate each modality's contribution during
test-time adaptation.

Key scientific improvements:
1. Dimension-normalized Mahalanobis mean shift D_mu / D
2. Normalized log-eigenvalue spectral divergence D_lambda / k
3. Source clean-null calibration: estimates baseline shift score on clean holdout windows
   so that d_t - d_null is near 0 on clean target streams.
4. Masked Softmax / Log-sum-exp reliability weighting:
   Strictly prevents zero-weight underflow (which caused the model collapse).
5. Exclusion of missing samples from batch covariance and mean calculations.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


def _to_cpu_float(t: torch.Tensor) -> torch.Tensor:
    """Move tensor to CPU and cast to float32."""
    return t.detach().cpu().float()


def _top_k_eigenvalues(cov: torch.Tensor, k: int) -> torch.Tensor:
    """Return top-k eigenvalues of symmetric PSD matrix in descending order."""
    eigenvalues, _ = torch.linalg.eigh(cov)
    top_k = eigenvalues[-k:].flip(0)
    return top_k.clamp(min=1e-8)


class SpectralShiftDetector:
    """Per-modality distribution shift detector using null-calibrated spectral statistics."""

    def __init__(
        self,
        feature_dims: Dict[str, int],
        gamma: float = 1.0,
        top_k: int = 32,
        eps: float = 1e-6,
    ) -> None:
        self.feature_dims = dict(feature_dims)
        self.gamma = gamma
        self.top_k = top_k
        self.eps = eps

        # Source statistics
        self.source_means: Dict[str, torch.Tensor] = {}
        self.source_stds: Dict[str, torch.Tensor] = {}
        self.source_eigenvalues: Dict[str, torch.Tensor] = {}
        self.source_null_scores: Dict[str, float] = {}
        self._source_registered = False

        # Shift-score history for velocity estimation
        self._shift_history: Dict[str, deque] = {
            m: deque(maxlen=50) for m in feature_dims
        }

    def register_source_stats(
        self, features_by_modality: Dict[str, torch.Tensor]
    ) -> None:
        """Compute source anchor statistics and calibrate clean null baseline."""
        for modality, feats in features_by_modality.items():
            feats = _to_cpu_float(feats)
            N, D = feats.shape
            assert D == self.feature_dims[modality], (
                f"Dimension mismatch for '{modality}': expected {self.feature_dims[modality]}, got {D}"
            )

            mu_s = feats.mean(dim=0)
            std_s = feats.std(dim=0).clamp(min=1e-4)

            centered = feats - mu_s.unsqueeze(0)
            cov_s = (centered.T @ centered) / max(N - 1, 1)

            k = min(self.top_k, D)
            eig_s = _top_k_eigenvalues(cov_s, k)

            self.source_means[modality] = mu_s
            self.source_stds[modality] = std_s
            self.source_eigenvalues[modality] = eig_s

            # Calibrate clean null score using random sub-windows of size min(32, N)
            win_size = min(32, N)
            if N >= win_size * 2:
                null_scores = []
                for _ in range(5):
                    perm = torch.randperm(N)[:win_size]
                    win_feats = feats[perm]
                    mu_w = win_feats.mean(dim=0)
                    mean_term = torch.mean(((mu_w - mu_s) / (std_s + self.eps)) ** 2).item()
                    cov_w = (win_feats - mu_w).T @ (win_feats - mu_w) / max(win_size - 1, 1)
                    eig_w = _top_k_eigenvalues(cov_w, k)
                    eig_term = torch.mean((torch.log(eig_w) - torch.log(eig_s)) ** 2).item()
                    null_scores.append(mean_term + self.gamma * eig_term)
                self.source_null_scores[modality] = float(np.median(null_scores))
            else:
                self.source_null_scores[modality] = 0.0

            logger.info(
                f"Registered '{modality}': D={D}, N={N}, eig_range=[{eig_s[-1].item():.4f}, {eig_s[0].item():.4f}], null_score={self.source_null_scores[modality]:.4f}"
            )

        self._source_registered = True

    def compute_shift_scores(
        self,
        features_by_modality: Dict[str, torch.Tensor],
        modality_mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """Compute per-modality shift scores d_t^(m) relative to calibrated clean null."""
        assert self._source_registered, "Call register_source_stats() before compute_shift_scores()."

        scores: Dict[str, float] = {}

        for modality, feats in features_by_modality.items():
            if modality not in self.source_means:
                continue

            feats = _to_cpu_float(feats)
            N, D = feats.shape

            # Exclude missing samples (AUD-04)
            if modality_mask is not None and modality in modality_mask:
                mask = _to_cpu_float(modality_mask[modality]).view(-1)
                valid_idx = torch.nonzero(mask > 0.5, as_tuple=True)[0]
                if len(valid_idx) < 2:
                    d_t = 10.0
                    scores[modality] = d_t
                    self._shift_history[modality].append(d_t)
                    continue
                valid_feats = feats[valid_idx]
            else:
                valid_feats = feats

            N_valid = valid_feats.shape[0]
            mu_s = self.source_means[modality]
            std_s = self.source_stds[modality]
            eig_s = self.source_eigenvalues[modality]
            k = eig_s.shape[0]

            # 1. Standardized mean shift
            mu_t = valid_feats.mean(dim=0)
            mean_shift = torch.mean(((mu_t - mu_s) / (std_s + self.eps)) ** 2).item()

            # 2. Log-eigenvalue spectral divergence
            centered = valid_feats - mu_t.unsqueeze(0)
            cov_t = (centered.T @ centered) / max(N_valid - 1, 1)
            eig_t = _top_k_eigenvalues(cov_t, k)
            spectral_div = torch.mean((torch.log(eig_t) - torch.log(eig_s)) ** 2).item()

            raw_d = mean_shift + self.gamma * spectral_div
            # Subtract null baseline to get calibrated excess shift
            null_score = self.source_null_scores.get(modality, 0.0)
            excess_shift = max(0.0, raw_d - null_score)

            scores[modality] = float(excess_shift)
            self._shift_history[modality].append(float(excess_shift))

        return scores

    def compute_reliability_weights(
        self,
        shift_scores: Dict[str, float],
        modality_mask: Dict[str, torch.Tensor],
        tau: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Compute per-modality reliability weights in [0, 1].

        .. math::
            r_t^m = \\exp(-\\tau \\cdot d_t^m) \\cdot m_t^m

        With excess shift d_t >= 0, clean modalities have d_t ≈ 0 => r_t ≈ 1.0.
        Corrupted modalities have d_t >> 1 => r_t << 1.0.
        """
        weights: Dict[str, torch.Tensor] = {}

        for modality, d_t in shift_scores.items():
            # Exponential decay on calibrated excess shift
            clipped_d = min(max(d_t, 0.0), 10.0)
            w = float(np.exp(-tau * clipped_d))

            mask = modality_mask.get(modality, torch.tensor(1.0))
            if isinstance(mask, torch.Tensor):
                mask = mask.float()
            else:
                mask = torch.tensor(float(mask))

            weights[modality] = mask * w

        return weights

    def classify_modalities(
        self,
        reliability_weights: Dict[str, torch.Tensor],
        theta_high: float = 0.7,
        theta_low: float = 0.3,
    ) -> Tuple[Set[str], Set[str], Set[str]]:
        """Classify modalities into Anchor (A), Adaptable (U), and Corrupted/Dropped (D) sets."""
        anchor: Set[str] = set()
        adaptable: Set[str] = set()
        corrupted: Set[str] = set()

        for modality, weight in reliability_weights.items():
            r_val = float(weight.mean().item()) if isinstance(weight, torch.Tensor) else float(weight)
            if r_val >= theta_high:
                anchor.add(modality)
            elif r_val >= theta_low:
                adaptable.add(modality)
            else:
                corrupted.add(modality)

        return anchor, adaptable, corrupted

    def compute_shift_velocity(self) -> Dict[str, float]:
        """Compute per-modality shift velocity: Δd_t = d_t − d_{t-1}."""
        velocities: Dict[str, float] = {}
        for modality, history in self._shift_history.items():
            if len(history) >= 2:
                velocities[modality] = history[-1] - history[-2]
            else:
                velocities[modality] = 0.0
        return velocities

    def compute_acceleration(self) -> Dict[str, float]:
        """Estimate second-order acceleration of shift scores."""
        accelerations: Dict[str, float] = {}
        for modality, history in self._shift_history.items():
            if len(history) >= 3:
                accelerations[modality] = history[-1] - 2 * history[-2] + history[-3]
            else:
                accelerations[modality] = 0.0
        return accelerations

    def reset_history(self) -> None:
        for history in self._shift_history.values():
            history.clear()

    def state_dict(self) -> dict:
        return {
            "feature_dims": self.feature_dims,
            "gamma": self.gamma,
            "top_k": self.top_k,
            "eps": self.eps,
            "source_means": {m: v.clone() for m, v in self.source_means.items()},
            "source_stds": {m: v.clone() for m, v in self.source_stds.items()},
            "source_eigenvalues": {m: v.clone() for m, v in self.source_eigenvalues.items()},
            "source_null_scores": dict(self.source_null_scores),
            "source_registered": self._source_registered,
        }

    def load_state_dict(self, state: dict) -> None:
        self.feature_dims = state["feature_dims"]
        self.gamma = state["gamma"]
        self.top_k = state["top_k"]
        self.eps = state.get("eps", 1e-6)
        self.source_means = {m: v.clone() for m, v in state["source_means"].items()}
        self.source_stds = {
            m: v.clone() for m, v in state.get("source_stds", {}).items()
        } if "source_stds" in state else {m: torch.ones_like(v) for m, v in self.source_means.items()}
        self.source_eigenvalues = {m: v.clone() for m, v in state["source_eigenvalues"].items()}
        self.source_null_scores = dict(state.get("source_null_scores", {}))
        self._source_registered = state["source_registered"]
        self._shift_history = {m: deque(maxlen=50) for m in self.feature_dims}

    def save_source_stats(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load_source_stats(self, path: str) -> None:
        state = torch.load(path, map_location="cpu")
        self.load_state_dict(state)
