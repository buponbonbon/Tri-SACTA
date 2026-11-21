"""
Per-Modality Spectral Shift Detector for Tri-SACTA.

Detects distribution shift per modality using a combination of mean-shift
and spectral (eigenvalue) divergence between source and target distributions.
Computes reliability weights that gate each modality's contribution during
test-time adaptation.

All computations are performed in FP32 on CPU for numerical stability.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import torch
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_cpu_float(t: torch.Tensor) -> torch.Tensor:
    """Move tensor to CPU and cast to float32."""
    return t.detach().cpu().float()


def _top_k_eigenvalues(cov: torch.Tensor, k: int) -> torch.Tensor:
    """Return the top-k eigenvalues of a symmetric PSD matrix.

    Uses ``torch.linalg.eigh`` which returns eigenvalues in ascending order;
    we take the last *k* and reverse so they are in descending order.

    Args:
        cov: Symmetric matrix of shape ``(D, D)``.
        k: Number of top eigenvalues to keep.

    Returns:
        Tensor of shape ``(k,)`` with the largest eigenvalues (descending).
    """
    # eigh is more stable on CPU
    eigenvalues, _ = torch.linalg.eigh(cov)
    # eigenvalues are ascending; take top-k (last k) and reverse
    top_k = eigenvalues[-k:].flip(0)
    # Clamp to zero to avoid tiny negatives from numerical noise
    return torch.clamp(top_k, min=0.0)


# ---------------------------------------------------------------------------
# SpectralShiftDetector
# ---------------------------------------------------------------------------


class SpectralShiftDetector:
    """Detect per-modality distribution shift via spectral statistics.

    For each modality *m* the shift score is:

        d_t^m = ||μ_t − μ_s||₂²  +  γ · ||eig(Σ_t) − eig(Σ_s)||₂²

    Reliability weights are then:

        r_t^m = exp(−τ · d_t^m) · mask^m

    Parameters
    ----------
    feature_dims : dict
        Maps modality name → feature dimensionality,
        e.g. ``{'visual': 512, 'audio': 768, 'text': 768, 'prosody': 88}``.
    gamma : float
        Weight for the eigenvalue divergence term (default ``1.0``).
    top_k : int
        Number of top eigenvalues to retain per modality (default ``32``).
    """

    def __init__(
        self,
        feature_dims: Dict[str, int],
        gamma: float = 1.0,
        top_k: int = 32,
    ) -> None:
        self.feature_dims = dict(feature_dims)
        self.gamma = gamma
        self.top_k = top_k

        # Source statistics (populated by register_source_stats)
        self.source_means: Dict[str, torch.Tensor] = {}
        self.source_eigenvalues: Dict[str, torch.Tensor] = {}
        self._source_registered = False

        # Shift-score history for velocity estimation (per modality)
        self._shift_history: Dict[str, deque] = {
            m: deque(maxlen=50) for m in feature_dims
        }

    # ------------------------------------------------------------------
    # Source registration
    # ------------------------------------------------------------------

    def register_source_stats(
        self, features_by_modality: Dict[str, torch.Tensor]
    ) -> None:
        """Compute and freeze source mean and top-k eigenvalues per modality.

        Args:
            features_by_modality: ``{modality_name: Tensor(N, D)}`` with the
                source-domain features.  Tensors may reside on any device;
                they are moved to CPU/FP32 internally.
        """
        for modality, feats in features_by_modality.items():
            feats = _to_cpu_float(feats)  # (N, D)
            assert feats.ndim == 2, (
                f"Expected 2-D features for '{modality}', got shape {feats.shape}"
            )

            N, D = feats.shape
            assert D == self.feature_dims[modality], (
                f"Dimension mismatch for '{modality}': expected {self.feature_dims[modality]}, "
                f"got {D}"
            )

            # Source mean
            mu_s = feats.mean(dim=0)  # (D,)

            # Covariance (unbiased)
            centered = feats - mu_s.unsqueeze(0)  # (N, D)
            cov_s = (centered.T @ centered) / max(N - 1, 1)  # (D, D)

            # Top-k eigenvalues (clamp k to available dims)
            k = min(self.top_k, D)
            eig_s = _top_k_eigenvalues(cov_s, k)  # (k,)

            self.source_means[modality] = mu_s
            self.source_eigenvalues[modality] = eig_s

            logger.info(
                "Registered source stats for '%s': D=%d, N=%d, top_%d eig range [%.4f, %.4f]",
                modality, D, N, k, eig_s[-1].item(), eig_s[0].item(),
            )

        self._source_registered = True

    # ------------------------------------------------------------------
    # Shift scores
    # ------------------------------------------------------------------

    def compute_shift_scores(
        self, features_by_modality: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute per-modality shift scores w.r.t. registered source stats.

        .. math::

            d_t^m = \\|\\mu_t - \\mu_s\\|_2^2
                    + \\gamma \\cdot \\|\\mathrm{eig}(\\Sigma_t)
                    - \\mathrm{eig}(\\Sigma_s)\\|_2^2

        Args:
            features_by_modality: ``{modality_name: Tensor(N, D)}`` – the
                target/test-time batch features.

        Returns:
            ``{modality_name: d_t}`` where ``d_t`` is a *float* score ≥ 0.
        """
        assert self._source_registered, (
            "Call register_source_stats() before compute_shift_scores()."
        )

        scores: Dict[str, float] = {}

        for modality, feats in features_by_modality.items():
            if modality not in self.source_means:
                logger.warning(
                    "Modality '%s' not in source stats – skipping.", modality
                )
                continue

            feats = _to_cpu_float(feats)  # (N, D)
            N, D = feats.shape

            mu_s = self.source_means[modality]
            eig_s = self.source_eigenvalues[modality]
            k = eig_s.shape[0]

            # Target mean
            mu_t = feats.mean(dim=0)  # (D,)

            # Mean-shift term
            mean_shift = torch.sum((mu_t - mu_s) ** 2).item()

            # Target covariance & eigenvalues
            centered = feats - mu_t.unsqueeze(0)
            cov_t = (centered.T @ centered) / max(N - 1, 1)
            eig_t = _top_k_eigenvalues(cov_t, k)

            # Spectral divergence term
            eig_div = torch.sum((eig_t - eig_s) ** 2).item()

            d_t = mean_shift + self.gamma * eig_div
            scores[modality] = d_t

            # Record for velocity computation
            self._shift_history[modality].append(d_t)

        return scores

    # ------------------------------------------------------------------
    # Reliability weights
    # ------------------------------------------------------------------

    def compute_reliability_weights(
        self,
        shift_scores: Dict[str, float],
        modality_mask: Dict[str, torch.Tensor],
        tau: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Compute soft reliability weights from shift scores and masks.

        .. math::

            r_t^m = \\exp(-\\tau \\cdot d_t^m) \\cdot \\text{mask}^m

        Args:
            shift_scores: Per-modality shift score ``d_t``.
            modality_mask: Per-modality availability mask (broadcastable to
                batch dimension). Values should be 0 or 1.
            tau: Temperature controlling sensitivity to shift.

        Returns:
            ``{modality_name: weight_tensor}`` with values in ``[0, 1]``.
        """
        weights: Dict[str, torch.Tensor] = {}

        for modality, d_t in shift_scores.items():
            # Scalar weight from shift score
            w = float(np.exp(-tau * d_t))

            # Get the mask (could be scalar or per-sample)
            mask = modality_mask.get(modality, torch.tensor(1.0))
            if isinstance(mask, torch.Tensor):
                mask = mask.float()
            else:
                mask = torch.tensor(float(mask))

            weights[modality] = mask * w

        return weights

    # ------------------------------------------------------------------
    # Modality classification
    # ------------------------------------------------------------------

    def classify_modalities(
        self,
        reliability_weights: Dict[str, torch.Tensor],
        theta_high: float = 0.7,
        theta_low: float = 0.3,
    ) -> Tuple[Set[str], Set[str], Set[str]]:
        """Partition modalities into anchor / adaptable / corrupted sets.

        * **Anchor** (``r ≥ θ_high``): reliable, used as-is to guide others.
        * **Adaptable** (``θ_low ≤ r < θ_high``): moderate shift, adapted.
        * **Corrupted** (``r < θ_low``): severe shift, down-weighted or dropped.

        Args:
            reliability_weights: Output of :meth:`compute_reliability_weights`.
            theta_high: Upper threshold.
            theta_low: Lower threshold.

        Returns:
            ``(anchor_set, adaptable_set, corrupted_set)`` – each a set of
            modality name strings.
        """
        anchor_set: Set[str] = set()
        adaptable_set: Set[str] = set()
        corrupted_set: Set[str] = set()

        for modality, w in reliability_weights.items():
            # Collapse to scalar for classification
            w_scalar = w.mean().item() if isinstance(w, torch.Tensor) else float(w)

            if w_scalar >= theta_high:
                anchor_set.add(modality)
            elif w_scalar >= theta_low:
                adaptable_set.add(modality)
            else:
                corrupted_set.add(modality)

        logger.debug(
            "Modality classification: anchor=%s, adaptable=%s, corrupted=%s",
            anchor_set, adaptable_set, corrupted_set,
        )
        return anchor_set, adaptable_set, corrupted_set

    # ------------------------------------------------------------------
    # Shift velocity
    # ------------------------------------------------------------------

    def compute_shift_velocity(self) -> Dict[str, float]:
        """Estimate the first-order velocity of shift scores over time.

        Uses a simple finite-difference on the most recent two entries in the
        per-modality shift-score history.

        Returns:
            ``{modality_name: velocity}`` where velocity > 0 indicates an
            *increasing* shift.  Returns ``0.0`` when fewer than two
            observations are available.
        """
        velocities: Dict[str, float] = {}
        for modality, history in self._shift_history.items():
            if len(history) >= 2:
                velocities[modality] = history[-1] - history[-2]
            else:
                velocities[modality] = 0.0
        return velocities

    def compute_acceleration(self) -> Dict[str, float]:
        """Estimate second-order acceleration of shift scores.

        Uses a central finite-difference on the most recent three entries.

        Returns:
            ``{modality_name: acceleration}`` – positive means the shift is
            *accelerating*.
        """
        accelerations: Dict[str, float] = {}
        for modality, history in self._shift_history.items():
            if len(history) >= 3:
                # Central difference: f''(t) ≈ f(t) - 2f(t-1) + f(t-2)
                accelerations[modality] = (
                    history[-1] - 2 * history[-2] + history[-3]
                )
            else:
                accelerations[modality] = 0.0
        return accelerations

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_history(self) -> None:
        """Clear shift-score history (but keep registered source stats)."""
        for history in self._shift_history.values():
            history.clear()

    def state_dict(self) -> dict:
        """Serialize detector state for checkpointing."""
        return {
            "feature_dims": self.feature_dims,
            "gamma": self.gamma,
            "top_k": self.top_k,
            "source_means": {
                m: v.clone() for m, v in self.source_means.items()
            },
            "source_eigenvalues": {
                m: v.clone() for m, v in self.source_eigenvalues.items()
            },
            "source_registered": self._source_registered,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore detector state from a checkpoint."""
        self.feature_dims = state["feature_dims"]
        self.gamma = state["gamma"]
        self.top_k = state["top_k"]
        self.source_means = {
            m: v.clone() for m, v in state["source_means"].items()
        }
        self.source_eigenvalues = {
            m: v.clone() for m, v in state["source_eigenvalues"].items()
        }
        self._source_registered = state["source_registered"]
        # Re-initialise history deques for all known modalities
        self._shift_history = {
            m: deque(maxlen=50) for m in self.feature_dims
        }

    def save_source_stats(self, path: str) -> None:
        """Save detector source statistics to disk."""
        torch.save(self.state_dict(), path)

    def load_source_stats(self, path: str) -> None:
        """Load detector source statistics from disk."""
        state = torch.load(path, map_location="cpu")
        self.load_state_dict(state)

