"""Test-time adaptation methods for Tri-SACTA.

Provides the core Tri-SACTA adaptation algorithm and baseline TTA methods
(TENT, SAR, etc.) for comparison.
"""

from .tri_sacta import TriSACTA
from .baselines_tta import TENT, SAR, MultimodalTTABaseline

__all__ = [
    "TriSACTA",
    "TENT",
    "SAR",
    "MultimodalTTABaseline",
]
