"""
src/service/utility_router.py: Expected-Utility Sequential Modality Acquisition Router.

Implements the Tri-SACTA V13 sequential modality acquisition policy governed by
marginal expected utility under explicit cost and latency budgets.

Mathematical Formulation:
-------------------------
Given current acquired subset S and candidate modality m not in S:
Incremental Cost: Delta C(m) = cost(m)
Incremental Latency: Delta L(m) = latency(m)
Estimated Risk: R(S) = 1 - max_c p_S(c) (or Shannon entropy H(p_S))

Marginal Expected Risk Reduction:
Delta R_hat(m | S) = R(S) * gamma(m | S) - (1 - R(S)) * nu(m | S)
where:
  gamma(m | S) = P(S U {m} correct | S incorrect)  [Rescue rate]
  nu(m | S)    = P(S U {m} incorrect | S correct)    [Negative transfer / corruption rate]

Marginal Expected Utility:
Utility(m | S) = Delta R_hat(m | S) / Delta C(m)

Decision Rule:
Acquire m* = argmax_{m feasible} Utility(m | S) iff Utility(m* | S) > lambda_utility
subject to:
  spent_cost + Delta C(m*) <= cost_budget
  spent_latency + Delta L(m*) <= deadline_ms
  modality m* is physically available in modality_mask
"""

from __future__ import annotations

import os
import sys
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Modality metadata and costs per protocol
MODALITY_COSTS = {"text": 1.0, "audio": 1.2, "visual": 1.3}
MODALITY_LATENCIES = {"text": 1.0, "audio": 5.0, "visual": 8.0}

SUBSET_METADATA = {
    "T": {"mods": ["text"], "cost": 1.0, "latency_ms": 1.0},
    "A": {"mods": ["audio"], "cost": 1.2, "latency_ms": 5.0},
    "V": {"mods": ["visual"], "cost": 1.3, "latency_ms": 8.0},
    "TA": {"mods": ["text", "audio"], "cost": 2.2, "latency_ms": 6.0},
    "TV": {"mods": ["text", "visual"], "cost": 2.3, "latency_ms": 9.0},
    "AV": {"mods": ["audio", "visual"], "cost": 2.5, "latency_ms": 13.0},
    "TAV": {"mods": ["text", "audio", "visual"], "cost": 3.5, "latency_ms": 14.0},
}

# Empirical rescue (gamma) and corruption (nu) priors derived from validation headroom
EMPIRICAL_PRIORS = {
    ("T", "audio"): {"gamma": 0.3768, "nu": 0.1160},
    ("T", "visual"): {"gamma": 0.3387, "nu": 0.2245},
    ("TA", "visual"): {"gamma": 0.2437, "nu": 0.1700},
    ("TV", "audio"): {"gamma": 0.2891, "nu": 0.1850},
    ("A", "text"): {"gamma": 0.5495, "nu": 0.1200},
    ("A", "visual"): {"gamma": 0.3781, "nu": 0.2500},
    ("V", "text"): {"gamma": 0.5473, "nu": 0.1200},
    ("V", "audio"): {"gamma": 0.3461, "nu": 0.2600},
    ("AV", "text"): {"gamma": 0.5467, "nu": 0.1100},
}


def mods_to_subset_key(mods: List[str]) -> str:
    """Canonical mapping from modality list to subset key."""
    s = set(mods)
    if s == {"text"}: return "T"
    if s == {"audio"}: return "A"
    if s == {"visual"}: return "V"
    if s == {"text", "audio"}: return "TA"
    if s == {"text", "visual"}: return "TV"
    if s == {"audio", "visual"}: return "AV"
    if s == {"text", "audio", "visual"}: return "TAV"
    raise ValueError(f"Unknown modality combination: {mods}")


class UtilityRouter:
    """Expected-Utility Sequential Modality Acquisition Policy."""

    def __init__(
        self,
        predictor_bank: Dict[str, nn.Module],
        device: torch.device,
        lambda_utility: float = 0.05,
        tau_stop_confidence: float = 0.75,
        conformal_quantile: Optional[float] = None,
    ):
        """
        Args:
            predictor_bank: Dict mapping subset keys ('T', 'A', 'V', 'TA', 'TV', 'AV', 'TAV')
                            to trained PyTorch models.
            device: torch.device (cpu or cuda)
            lambda_utility: Minimum expected risk reduction per cost unit to trigger acquisition.
            tau_stop_confidence: If top probability exceeds this threshold, stop acquisition early.
            conformal_quantile: Quantile (1 - alpha) calibrated on deployed trajectory for prediction sets.
        """
        self.predictor_bank = predictor_bank
        self.device = device
        self.lambda_utility = float(lambda_utility)
        self.tau_stop_confidence = float(tau_stop_confidence)
        self.conformal_quantile = conformal_quantile

        for m in self.predictor_bank.values():
            m.eval()
            m.to(self.device)

    def _predict_subset(self, subset_key: str, features: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, int]:
        """Runs forward inference on the task-matched subset predictor."""
        model = self.predictor_bank[subset_key]
        mods = SUBSET_METADATA[subset_key]["mods"]
        tensors = [features[m] for m in mods]
        concat_feat = torch.cat(tensors, dim=-1)

        with torch.no_grad():
            logits = model(concat_feat)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            pred = int(np.argmax(probs))
        return probs, pred

    def estimate_marginal_utility(
        self,
        current_subset: str,
        current_probs: np.ndarray,
        candidate_mod: str,
    ) -> float:
        """Computes expected marginal utility Delta R_hat / Delta C."""
        p_max = float(np.max(current_probs))
        risk = 1.0 - p_max

        prior = EMPIRICAL_PRIORS.get((current_subset, candidate_mod), {"gamma": 0.20, "nu": 0.20})
        gamma = prior["gamma"]
        nu = prior["nu"]

        # Expected risk reduction
        delta_r_hat = (risk * gamma) - ((1.0 - risk) * nu)
        delta_c = MODALITY_COSTS[candidate_mod]
        utility = delta_r_hat / delta_c
        return utility

    def route_single_query(
        self,
        features: Dict[str, torch.Tensor],
        modality_mask: Dict[str, bool],
        deadline_ms: float = 50.0,
        cost_budget: float = 3.5,
    ) -> Dict[str, Any]:
        """
        Executes sequential modality acquisition for a single incoming query.

        Args:
            features: Dict mapping 'text', 'audio', 'visual' to 1xD tensors.
            modality_mask: Dict mapping modality name to availability bool.
            deadline_ms: Max latency permitted.
            cost_budget: Max cost permitted.

        Returns:
            Dict containing prediction, probabilities, final_subset, cost, latency, path, and trace.
        """
        # 1. Determine Initial Available Modality (prefer text if available)
        available_mods = [m for m in ["text", "audio", "visual"] if modality_mask.get(m, True)]
        if not available_mods:
            # Complete blackout fallback
            return {
                "prediction": 0,
                "probabilities": np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
                "final_subset": "T",
                "spent_cost": 1.0,
                "spent_latency_ms": 1.0,
                "acquisition_path": ["T"],
                "routing_reason": "blackout_fallback",
                "conformal_set": [0, 1, 2],
            }

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

        # 2. Sequential Acquisition Loop
        while True:
            # Check early stopping if prediction is already confident
            p_max = float(np.max(probs))
            if p_max >= self.tau_stop_confidence:
                routing_reason = f"confident_stopping_{current_key}_pmax_{p_max:.3f}"
                break

            # Find unacquired, available candidate modalities that fit cost and deadline
            candidates = [m for m in available_mods if m not in current_mods]
            feasible_candidates = []
            for m in candidates:
                inc_cost = MODALITY_COSTS[m]
                inc_lat = MODALITY_LATENCIES[m]
                if (spent_cost + inc_cost <= cost_budget) and (spent_latency + inc_lat <= deadline_ms):
                    feasible_candidates.append(m)

            if not feasible_candidates:
                routing_reason = f"budget_or_availability_exhausted_{current_key}"
                break

            # Evaluate marginal utility for each feasible candidate
            utilities = {m: self.estimate_marginal_utility(current_key, probs, m) for m in feasible_candidates}
            best_mod = max(utilities, key=utilities.get)
            best_utility = utilities[best_mod]

            # If best utility fails threshold, terminate sequential acquisition
            if best_utility < self.lambda_utility:
                routing_reason = f"utility_threshold_unmet_{current_key}_best_{best_mod}_{best_utility:.4f}"
                break

            # Acquire best modality
            current_mods.append(best_mod)
            current_key = mods_to_subset_key(current_mods)
            spent_cost += MODALITY_COSTS[best_mod]
            spent_latency += MODALITY_LATENCIES[best_mod]
            acquisition_path.append(current_key)

            # Re-evaluate with augmented subset predictor
            probs, pred = self._predict_subset(current_key, features)

            if len(current_mods) == len(available_mods) or len(current_mods) == 3:
                routing_reason = f"all_available_acquired_{current_key}"
                break

        # 3. Form Conformal Prediction Set if Quantile is Calibrated
        if self.conformal_quantile is not None:
            # Score function: s(x, y) = 1 - p(y). Set includes c where 1 - p(c) <= q <=> p(c) >= 1 - q
            threshold = 1.0 - self.conformal_quantile
            conformal_set = [c for c in range(3) if probs[c] >= threshold]
            if len(conformal_set) == 0:
                conformal_set = [int(np.argmax(probs))]
        else:
            conformal_set = [int(np.argmax(probs))]

        return {
            "prediction": pred,
            "probabilities": probs,
            "final_subset": current_key,
            "spent_cost": float(spent_cost),
            "spent_latency_ms": float(spent_latency),
            "acquisition_path": acquisition_path,
            "routing_reason": routing_reason,
            "conformal_set": conformal_set,
        }
