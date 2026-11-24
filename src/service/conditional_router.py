"""
src/service/conditional_router.py: State-Conditioned Expected Utility Sequential Router (Candidate M2).

Acquisition decisions are governed by state-conditioned gain estimates:
G_hat(z, a) = E[E_S - E_{S U {a}} | z]
and marginal utility:
U(a | z) = G_hat(z, a) / Delta C(a)

Decision Rule:
Acquire a* = argmax_{a in A_feasible} U(a | z) iff U(a* | z) >= lambda_utility
subject to:
- Cumulative cost budget
- Monotonic remaining deadline
- Early-stopping confidence tau_stop_confidence
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from src.service.utility_router import (
    SUBSET_METADATA,
    MODALITY_COSTS,
    MODALITY_LATENCIES,
    mods_to_subset_key,
)
from src.service.baseline_policies import BasePredictorPolicy
from src.service.conditional_utility import ConditionalGainEstimator

logger = logging.getLogger(__name__)


class ConditionalUtilityRouter(BasePredictorPolicy):
    """
    Candidate M2: Conditional Utility Sequential Router.
    Conditions marginal gain and acquisition decisions on the observed pre-acquisition state z.
    """

    def __init__(
        self,
        predictor_bank: Dict[str, nn.Module],
        device: torch.device,
        lambda_utility: float = 0.05,
        tau_stop_confidence: float = 0.75,
        estimator: Optional[ConditionalGainEstimator] = None,
    ):
        super().__init__(predictor_bank, device)
        self.lambda_utility = float(lambda_utility)
        self.tau_stop_confidence = float(tau_stop_confidence)
        self.estimator = estimator if estimator is not None else ConditionalGainEstimator()

    def route_single_query(
        self,
        features: Dict[str, torch.Tensor],
        modality_mask: Dict[str, bool],
        deadline_ms: float = 50.0,
        cost_budget: float = 3.5,
    ) -> Dict[str, Any]:
        """
        Executes sequential acquisition with state-conditioned utility.
        """
        available_mods = [m for m in ["text", "audio", "visual"] if modality_mask.get(m, True)]
        if not available_mods:
            return {
                "prediction": 0,
                "probabilities": np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
                "final_subset": "T",
                "spent_cost": 1.0,
                "spent_latency_ms": 1.0,
                "acquisition_path": ["T"],
                "routing_reason": "blackout_fallback",
            }

        # Step 1: Initialize with base available modality (prefer text)
        if "text" in available_mods:
            current_mods = ["text"]
        elif "audio" in available_mods:
            current_mods = ["audio"]
        else:
            current_mods = ["visual"]

        current_key = mods_to_subset_key(current_mods)
        spent_cost = SUBSET_METADATA[current_key]["cost"]
        spent_latency = SUBSET_METADATA[current_key]["latency_ms"]
        acquisition_path = [current_key]

        probs, pred = self._predict_subset(current_key, features)

        step_traces = []

        # Step 2: Sequential conditional acquisition loop
        while True:
            p_max = float(np.max(probs))
            if p_max >= self.tau_stop_confidence:
                routing_reason = f"confident_stopping_{current_key}_pmax_{p_max:.3f}"
                break

            candidates = [m for m in available_mods if m not in current_mods]
            feasible_candidates = [
                m for m in candidates
                if (spent_cost + MODALITY_COSTS[m] <= cost_budget)
                and (spent_latency + MODALITY_LATENCIES[m] <= deadline_ms)
            ]

            if not feasible_candidates:
                routing_reason = f"budget_or_availability_exhausted_{current_key}"
                break

            # Evaluate conditional marginal utility for all feasible candidates
            candidate_gains = {}
            candidate_utilities = {}
            for m in feasible_candidates:
                g = self.estimator.estimate_gain(
                    current_key, probs, m, spent_cost, spent_latency, cost_budget, deadline_ms
                )
                u = self.estimator.estimate_utility(
                    current_key, probs, m, spent_cost, spent_latency, cost_budget, deadline_ms
                )
                candidate_gains[m] = g
                candidate_utilities[m] = u

            best_mod = max(candidate_utilities, key=candidate_utilities.get)
            best_u = candidate_utilities[best_mod]
            best_g = candidate_gains[best_mod]

            step_traces.append({
                "subset": current_key,
                "p_max": p_max,
                "candidate_gains": candidate_gains,
                "candidate_utilities": candidate_utilities,
                "best_candidate": best_mod,
            })

            # Check threshold: if utility threshold is not met, terminate
            if best_u < self.lambda_utility:
                routing_reason = f"conditional_utility_unmet_{current_key}_best_{best_mod}_u_{best_u:.4f}_g_{best_g:.4f}"
                break

            # Acquire best modality
            current_mods.append(best_mod)
            current_key = mods_to_subset_key(current_mods)
            spent_cost += MODALITY_COSTS[best_mod]
            spent_latency += MODALITY_LATENCIES[best_mod]
            acquisition_path.append(current_key)

            # Re-predict with augmented subset
            probs, pred = self._predict_subset(current_key, features)

            if len(current_mods) == len(available_mods) or len(current_mods) == 3:
                routing_reason = f"all_available_acquired_{current_key}"
                break

        return {
            "prediction": pred,
            "probabilities": probs,
            "final_subset": current_key,
            "spent_cost": float(spent_cost),
            "spent_latency_ms": float(spent_latency),
            "acquisition_path": acquisition_path,
            "routing_reason": routing_reason,
            "step_traces": step_traces,
        }
