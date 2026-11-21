"""
Conformal prediction module for Tri-SACTA.

Provides online adaptive and static baseline conformal predictors
with mask-conditional coverage guarantees.
"""

from src.conformal.online_aci import OnlineACI
from src.conformal.static_cp import MaskConditionalCP, SplitConformalPredictor

__all__ = [
    "OnlineACI",
    "SplitConformalPredictor",
    "MaskConditionalCP",
]
