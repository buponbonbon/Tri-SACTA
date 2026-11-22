"""
Per-Modality Spectral Shift Detector for Tri-SACTA with Sample-Size Matched Statistics — V3 (R3-08).

Key scientific improvements in V3:
1. Sample-Size Matched Eigenvalue Rank (R3-08):
   Truncates eigenvalue rank dynamically: k = min(top_k, D, N_valid - 1).
   Strictly prevents rank-collapse bias where taking top-32 eigenvalues on small batches
   (N=8 or 16) forced logs of zero-floored eigenvalues, spuriously inflating shift scores.
2. Valid-Rows-Only Source Statistics:
   Excludes all zero-padded rows corresponding to missing modalities.
3. Holdout Validation Null Estimation:
   Calibrates null distribution on independent holdout validation batches.
4. Stable reliability conversion:
   r_t = max(1e-4, exp(-tau * d_t)) * availability_mask.
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


def _top_k_eigenvalues_from_centered(centered: torch.Tensor, k: int) -> torch.Tensor:
    """Compute top-k eigenvalues using Gram matrix when N < D (Section 6.1, R3-08).

    The non-zero eigenvalues of (X @ X.T) / (N-1) are identical to those of (X.T @ X) / (N-1).
    When N << D (e.g. N <= 32 << 768), eigendecomposition of (N, N) is orders of magnitude faster.
    """
    if k <= 0:
        return torch.empty(0, dtype=centered.dtype, device=centered.device)
    N, D = centered.shape
    scale = max(N - 1, 1)
    if N < D:
        # Gram matrix (N, N)
        G = (centered @ centered.T) / scale
        eigenvalues, _ = torch.linalg.eigh(G)
    else:
        # Covariance matrix (D, D)
        C = (centered.T @ centered) / scale
        eigenvalues, _ = torch.linalg.eigh(C)
    top_k = eigenvalues[-k:].flip(0)
    return top_k.clamp(min=1e-8)


def _top_k_eigenvalues(cov: torch.Tensor, k: int) -> torch.Tensor:
    """Legacy helper for direct PSD covariance matrices."""
    if k <= 0:
        return torch.empty(0, dtype=cov.dtype, device=cov.device)
    eigenvalues, _ = torch.linalg.eigh(cov)
    top_k = eigenvalues[-k:].flip(0)
    return top_k.clamp(min=1e-8)


class SpectralShiftDetector:
    """Per-modality distribution shift detector using sample-size matched spectral statistics."""

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
        self.source_covariances: Dict[str, torch.Tensor] = {}
        self.source_eigenvalues: Dict[str, torch.Tensor] = {}
        self.source_null_scores: Dict[str, float] = {}
        self.null_by_sample_size: Dict[str, Dict[int, float]] = {m: {} for m in feature_dims}
        self._source_registered = False

        # Shift-score history for velocity estimation
        self._shift_history: Dict[str, deque] = {
            m: deque(maxlen=50) for m in feature_dims
        }

    def register_source_stats(
        self,
        features_by_modality: Dict[str, torch.Tensor],
        val_features_by_modality: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Compute source anchor statistics and calibrate sample-size matched clean null baseline (R3-08)."""
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

            max_k = min(self.top_k, D, max(N - 1, 1))
            eig_s = _top_k_eigenvalues_from_centered(centered, max_k)

            self.source_means[modality] = mu_s
            self.source_stds[modality] = std_s
            self.source_covariances[modality] = cov_s
            self.source_eigenvalues[modality] = eig_s

            # Calibrate clean null scores across multiple batch sample sizes (R3-08)
            eval_source = val_features_by_modality.get(modality, feats) if val_features_by_modality else feats
            eval_source = _to_cpu_float(eval_source)
            if eval_source.ndim == 1:
                eval_source = eval_source.unsqueeze(1)
            N_eval = eval_source.shape[0]

            self.null_by_sample_size[modality] = {}
            grid_sizes = [2, 4, 8, 16, 32, 64]

            for s_sz in grid_sizes:
                if N_eval >= s_sz + 2:
                    sub_scores = []
                    n_reps = 15
                    for _ in range(n_reps):
                        perm = torch.randperm(N_eval)[:s_sz]
                        win_feats = eval_source[perm]
                        mu_w = win_feats.mean(dim=0)
                        mean_term = torch.mean(((mu_w - mu_s) / (std_s + self.eps)) ** 2).item()

                        k_w = min(self.top_k, D, s_sz - 1, eig_s.shape[0])
                        if k_w > 0:
                            win_centered = win_feats - mu_w
                            eig_w = _top_k_eigenvalues_from_centered(win_centered, k_w)
                            eig_ref = eig_s[:k_w]
                            eig_term = torch.mean((torch.log(eig_w) - torch.log(eig_ref)) ** 2).item()
                        else:
                            eig_term = 0.0
                        sub_scores.append(mean_term + self.gamma * eig_term)
                    self.null_by_sample_size[modality][s_sz] = float(np.median(sub_scores))

            # Set default 32-sample null score for backward compatibility
            self.source_null_scores[modality] = self.null_by_sample_size[modality].get(
                32, self.null_by_sample_size[modality].get(16, 0.0)
            )

            logger.info(
                f"Registered '{modality}': D={D}, N={N}, k_max={max_k}, "
                f"null_grid={self.null_by_sample_size[modality]}"
            )

        self._source_registered = True

    def _get_matched_null_score(self, modality: str, n_valid: int) -> float:
        """Lookup or interpolate sample-size matched clean null baseline."""
        grid = self.null_by_sample_size.get(modality, {})
        if not grid:
            return self.source_null_scores.get(modality, 0.0)

        if n_valid in grid:
            return grid[n_valid]

        # Nearest neighbor lookup in sample size grid
        sizes = sorted(list(grid.keys()))
        closest = min(sizes, key=lambda s: abs(s - n_valid))
        return grid[closest]

    def compute_shift_scores(
        self,
        features_by_modality: Dict[str, torch.Tensor],
        modality_mask: Optional[Dict[str, torch.Tensor]] = None,
        update_history: bool = True,
    ) -> Dict[str, float]:
        """Compute per-modality shift scores with sample-size matched rank truncation (R3-08)."""
        assert self._source_registered, "Call register_source_stats() before compute_shift_scores()."

        scores: Dict[str, float] = {}

        for modality, feats in features_by_modality.items():
            if modality not in self.source_means:
                continue

            feats = _to_cpu_float(feats)
            N, D = feats.shape

            # Exclude missing samples
            if modality_mask is not None and modality in modality_mask:
                mask = _to_cpu_float(modality_mask[modality]).view(-1)
                valid_idx = torch.nonzero(mask > 0.5, as_tuple=True)[0]
                if len(valid_idx) == 0:
                    scores[modality] = 0.0
                    if update_history:
                        self._shift_history[modality].append(0.0)
                    continue
                valid_feats = feats[valid_idx]
            else:
                valid_feats = feats

            N_valid = valid_feats.shape[0]
            mu_s = self.source_means[modality]
            std_s = self.source_stds[modality]
            eig_s = self.source_eigenvalues[modality]

            # 1. Standardized mean shift
            mu_t = valid_feats.mean(dim=0)
            mean_shift = torch.mean(((mu_t - mu_s) / (std_s + self.eps)) ** 2).item()

            # 2. Sample-size matched log-eigenvalue spectral divergence (R3-08)
            # Rank of cov_t is at most N_valid - 1. Match k to prevent zero-eigenvalue bias.
            k_match = min(self.top_k, D, N_valid - 1, eig_s.shape[0])
            if k_match >= 1:
                centered = valid_feats - mu_t.unsqueeze(0)
                cov_t = (centered.T @ centered) / max(N_valid - 1, 1)
                eig_t = _top_k_eigenvalues(cov_t, k_match)
                eig_ref = eig_s[:k_match]
                spectral_div = torch.mean((torch.log(eig_t) - torch.log(eig_ref)) ** 2).item()
            else:
                spectral_div = 0.0

            raw_d = mean_shift + self.gamma * spectral_div
            null_score = self._get_matched_null_score(modality, N_valid)
            excess_shift = max(0.0, raw_d - null_score)

            scores[modality] = float(excess_shift)
            if update_history:
                self._shift_history[modality].append(float(excess_shift))

        return scores

    def compute_reliability_weights(
        self,
        shift_scores: Dict[str, float],
        modality_mask: Dict[str, torch.Tensor],
        tau: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Compute per-modality reliability weights in [0, 1]."""
        weights: Dict[str, torch.Tensor] = {}

        for modality, d_t in shift_scores.items():
            mask = modality_mask.get(modality, None)
            if mask is None:
                continue

            r_val = float(np.exp(-tau * max(0.0, d_t)))
            r_val = max(1e-4, min(1.0, r_val))

            mask_f = mask.float().view(-1)
            w = mask_f * r_val
            weights[modality] = w

        return weights

    def compute_shift_velocity(self) -> float:
        """Compute average absolute shift velocity across modalities: mean(|d_t - d_{t-1}|)."""
        velocities = []
        for modality, history in self._shift_history.items():
            if len(history) >= 2:
                v = abs(history[-1] - history[-2])
                velocities.append(v)
        return float(np.mean(velocities)) if velocities else 0.0

    def reset_history(self) -> None:
        """Clear rolling shift history."""
        for m in self._shift_history:
            self._shift_history[m].clear()

    def save_source_stats(self, filepath: str) -> None:
        """Persist registered source statistics."""
        payload = {
            "source_means": self.source_means,
            "source_stds": self.source_stds,
            "source_covariances": self.source_covariances,
            "source_eigenvalues": self.source_eigenvalues,
            "source_null_scores": self.source_null_scores,
            "feature_dims": self.feature_dims,
            "gamma": self.gamma,
            "top_k": self.top_k,
        }
        torch.save(payload, filepath)
        logger.info(f"Saved source spectral statistics to {filepath}")

    def load_source_stats(self, filepath: str) -> None:
        """Load registered source statistics."""
        payload = torch.load(filepath, map_location="cpu")
        self.source_means = payload["source_means"]
        self.source_stds = payload["source_stds"]
        self.source_covariances = payload.get("source_covariances", {})
        self.source_eigenvalues = payload["source_eigenvalues"]
        self.source_null_scores = payload["source_null_scores"]
        self.feature_dims = payload["feature_dims"]
        self.gamma = payload.get("gamma", self.gamma)
        self.top_k = payload.get("top_k", self.top_k)
        self._source_registered = True
        logger.info(f"Loaded source spectral statistics from {filepath}")
