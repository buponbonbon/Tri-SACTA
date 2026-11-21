"""Model architectures for Tri-SACTA.

Provides the multimodal fusion backbone with cross-modal attention and
the spectral shift detector for modality reliability estimation.
"""

from .fusion_backbone import MultimodalFusionModel
from .spectral_detector import SpectralShiftDetector

__all__ = [
    "MultimodalFusionModel",
    "SpectralShiftDetector",
]
