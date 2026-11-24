"""
src/service/acquisition.py: Lazy feature acquisition provider for Tri-SACTA V12.1.
Enforces realistic, independent retrieval of text, audio, and visual modalities.
Prevents unbudgeted bulk tensor transfers by tracking simulated wire bytes and failure semantics.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple
import torch


DTYPE_BYTES = 4  # 32-bit float

FEATURE_DIMS = {
    "text": 768,
    "audio": 74,
    "visual": 35,
}


class AcquisitionResult:
    """Telemetry report for a modality retrieval attempt."""
    def __init__(
        self,
        modality: str,
        feature: Optional[torch.Tensor],
        success: bool,
        bytes_transferred: int,
        latency_ms: float,
        error_msg: Optional[str] = None,
    ):
        self.modality = modality
        self.feature = feature
        self.success = success
        self.bytes_transferred = bytes_transferred
        self.latency_ms = latency_ms
        self.error_msg = error_msg


class LazyModalityProvider:
    """Simulates or performs realistic independent on-demand modality acquisition."""

    def __init__(
        self,
        features_pool: Dict[str, Optional[torch.Tensor]],
        modality_mask: Optional[Dict[str, bool]] = None,
        simulated_network_latency_ms: Optional[Dict[str, float]] = None,
    ):
        self.features_pool = features_pool
        self.modality_mask = modality_mask or {m: True for m in ["text", "audio", "visual"]}
        self.simulated_latency = simulated_network_latency_ms or {
            "text": 0.5,
            "audio": 1.5,
            "visual": 2.0,
        }
        self.acquired_modalities: List[str] = []
        self.total_bytes_transferred: int = 0
        self.acquisition_history: List[AcquisitionResult] = []

    def acquire(self, modality: str) -> AcquisitionResult:
        """Independently fetches a single modality tensor, accounting for bandwidth and presence."""
        t0 = time.perf_counter()

        # Check availability
        is_available = self.modality_mask.get(modality, False)
        raw_feat = self.features_pool.get(modality, None)

        if not is_available or raw_feat is None:
            # Modality is missing or unmasked
            lat = (time.perf_counter() - t0) * 1000.0
            res = AcquisitionResult(
                modality=modality,
                feature=None,
                success=False,
                bytes_transferred=0,
                latency_ms=lat,
                error_msg=f"Modality '{modality}' unavailable or unmasked",
            )
            self.acquisition_history.append(res)
            return res

        # Validate tensor
        if not isinstance(raw_feat, torch.Tensor):
            raw_feat = torch.tensor(raw_feat, dtype=torch.float32)
        else:
            raw_feat = raw_feat.clone().detach().float()

        expected_dim = FEATURE_DIMS.get(modality, raw_feat.numel())
        if raw_feat.numel() != expected_dim:
            lat = (time.perf_counter() - t0) * 1000.0
            res = AcquisitionResult(
                modality=modality,
                feature=None,
                success=False,
                bytes_transferred=0,
                latency_ms=lat,
                error_msg=f"Dimension mismatch for {modality}: expected {expected_dim}, got {raw_feat.numel()}",
            )
            self.acquisition_history.append(res)
            return res

        num_bytes = raw_feat.numel() * DTYPE_BYTES
        sim_lat = self.simulated_latency.get(modality, 0.0)
        # Add micro-sleep only if explicitly requested, otherwise compute-only
        lat = ((time.perf_counter() - t0) * 1000.0) + sim_lat

        self.acquired_modalities.append(modality)
        self.total_bytes_transferred += num_bytes

        res = AcquisitionResult(
            modality=modality,
            feature=raw_feat,
            success=True,
            bytes_transferred=num_bytes,
            latency_ms=lat,
        )
        self.acquisition_history.append(res)
        return res
