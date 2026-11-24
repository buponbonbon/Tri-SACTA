"""
src/service/conditional_utility.py: State-Conditioned Transition Gain and Utility Estimator (Candidate M2).

Estimates the marginal reduction in predictive risk conditioned on the observed pre-acquisition state:
G_hat(z, a) = E[E_S - E_{S U {a}} | z]
where z is composed exclusively of observed state features:
z = [p_max, margin, entropy, p_neg, p_neu, p_pos, rem_cost_ratio, rem_lat_ratio]
"""

from __future__ import annotations

import os
import sys
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.service.utility_router import MODALITY_COSTS, MODALITY_LATENCIES

logger = logging.getLogger(__name__)


def extract_state_features(
    probs: np.ndarray,
    spent_cost: float,
    spent_lat: float,
    cost_budget: float = 3.5,
    deadline_ms: float = 50.0,
) -> np.ndarray:
    """
    Extracts observed state features z legitimately known before acquisition:
    z = [p_max, margin, entropy, p_neg, p_neu, p_pos, rem_cost_ratio, rem_lat_ratio]
    """
    p_max = float(np.max(probs))
    sorted_p = np.sort(probs)
    margin = float(sorted_p[-1] - sorted_p[-2]) if len(sorted_p) > 1 else 1.0
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))

    p_neg = float(probs[0])
    p_neu = float(probs[1])
    p_pos = float(probs[2])

    rem_cost = max(0.0, (cost_budget - spent_cost) / max(1.0, cost_budget))
    rem_lat = max(0.0, (deadline_ms - spent_lat) / max(1.0, deadline_ms))

    return np.array([p_max, margin, entropy, p_neg, p_neu, p_pos, rem_cost, rem_lat], dtype=np.float32)


class ConditionalGainEstimator:
    """Estimates state-dependent expected error reduction G_hat(z, a)."""

    def __init__(self, model_weights_path: Optional[str] = None):
        if model_weights_path is None:
            model_weights_path = os.path.join(
                REPO_ROOT, "results", "v13_1", "priors", "conditional_utility_model.json"
            )

        self.model_path = model_weights_path
        self.feature_names = ["p_max", "margin", "entropy", "p_neg", "p_neu", "p_pos", "rem_cost", "rem_lat"]
        self.models: Dict[str, Dict[str, Any]] = {}

        if os.path.exists(self.model_path):
            with open(self.model_path, "r", encoding="utf-8") as f:
                self.models = json.load(f)
        else:
            logger.warning(f"Conditional model weights not found at {self.model_path}. Using fallback linear priors.")

    def estimate_gain(
        self,
        current_subset: str,
        current_probs: np.ndarray,
        candidate_mod: str,
        spent_cost: float,
        spent_lat: float,
        cost_budget: float = 3.5,
        deadline_ms: float = 50.0,
    ) -> float:
        """
        Estimates G_hat(z, a) = E[E_S - E_{S U {a}} | z].
        """
        key = f"{current_subset}+{candidate_mod}"
        z = extract_state_features(current_probs, spent_cost, spent_lat, cost_budget, deadline_ms)

        if key in self.models:
            m_info = self.models[key]
            intercept = m_info["intercept"]
            coefs = m_info["coefficients"]
            gain = intercept + sum(coefs[fname] * z[idx] for idx, fname in enumerate(self.feature_names))
            return float(gain)
        else:
            # Fallback to corrected empirical prior gain: (1-p)*gamma - p*nu
            p_max = float(np.max(current_probs))
            # Default prior fallback
            gamma, nu = 0.20, 0.15
            gain = (1.0 - p_max) * gamma - p_max * nu
            return float(gain)

    def estimate_utility(
        self,
        current_subset: str,
        current_probs: np.ndarray,
        candidate_mod: str,
        spent_cost: float,
        spent_lat: float,
        cost_budget: float = 3.5,
        deadline_ms: float = 50.0,
    ) -> float:
        """
        Computes conditional utility:
        U(a | z) = G_hat(z, a) / Delta C(a)
        """
        gain = self.estimate_gain(
            current_subset, current_probs, candidate_mod, spent_cost, spent_lat, cost_budget, deadline_ms
        )
        cost = MODALITY_COSTS[candidate_mod]
        return float(gain / cost)
