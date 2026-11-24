"""
src/service/baseline_policies.py: Standardized baseline routing policies for Tri-SACTA V13 benchmark.

All baselines implement the identical interface:
route_query(features, modality_mask, deadline_ms, cost_budget) -> Dict[str, Any]
using the exact same task-matched subset predictor bank and device.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from src.service.utility_router import SUBSET_METADATA, MODALITY_COSTS, MODALITY_LATENCIES, mods_to_subset_key

logger = logging.getLogger(__name__)


class BasePredictorPolicy:
    """Base class for all routing policies sharing a predictor bank."""

    def __init__(self, predictor_bank: Dict[str, nn.Module], device: torch.device):
        self.predictor_bank = predictor_bank
        self.device = device
        for m in self.predictor_bank.values():
            m.eval()
            m.to(self.device)

    def _predict_subset(self, subset_key: str, features: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, int]:
        model = self.predictor_bank[subset_key]
        mods = SUBSET_METADATA[subset_key]["mods"]
        tensors = [features[m] for m in mods]
        concat_feat = torch.cat(tensors, dim=-1)

        with torch.no_grad():
            logits = model(concat_feat)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            pred = int(np.argmax(probs))
        return probs, pred


class ConfidenceRoutingPolicy(BasePredictorPolicy):
    """Standard Confidence-Based Routing Baseline."""

    def __init__(self, predictor_bank: Dict[str, nn.Module], device: torch.device, tau_confidence: float = 0.65):
        super().__init__(predictor_bank, device)
        self.tau_confidence = float(tau_confidence)

    def route_single_query(
        self,
        features: Dict[str, torch.Tensor],
        modality_mask: Dict[str, bool],
        deadline_ms: float = 50.0,
        cost_budget: float = 3.5,
    ) -> Dict[str, Any]:
        available_mods = [m for m in ["text", "audio", "visual"] if modality_mask.get(m, True)]
        if not available_mods:
            return {"prediction": 0, "probabilities": np.array([1/3, 1/3, 1/3]), "final_subset": "T",
                    "spent_cost": 1.0, "spent_latency_ms": 1.0, "acquisition_path": ["T"], "routing_reason": "blackout"}

        # Base triage
        base_mod = "text" if "text" in available_mods else available_mods[0]
        base_key = mods_to_subset_key([base_mod])
        probs, pred = self._predict_subset(base_key, features)
        conf = float(np.max(probs))

        spent_cost = SUBSET_METADATA[base_key]["cost"]
        spent_lat = SUBSET_METADATA[base_key]["latency_ms"]
        path = [base_key]

        if conf < self.tau_confidence:
            # Attempt full fusion escalation
            target_mods = [m for m in ["text", "audio", "visual"] if modality_mask.get(m, True)]
            target_key = mods_to_subset_key(target_mods)
            target_cost = SUBSET_METADATA[target_key]["cost"]
            target_lat = SUBSET_METADATA[target_key]["latency_ms"]

            if target_cost <= cost_budget and target_lat <= deadline_ms and target_key != base_key:
                probs, pred = self._predict_subset(target_key, features)
                spent_cost = target_cost
                spent_lat = target_lat
                path.append(target_key)
                reason = f"escalated_to_{target_key}_conf_{conf:.3f}_below_{self.tau_confidence}"
            else:
                reason = f"escalation_budget_constrained_stayed_{base_key}"
        else:
            reason = f"confident_triage_stayed_{base_key}_conf_{conf:.3f}"

        return {
            "prediction": pred,
            "probabilities": probs,
            "final_subset": path[-1],
            "spent_cost": float(spent_cost),
            "spent_latency_ms": float(spent_lat),
            "acquisition_path": path,
            "routing_reason": reason,
        }


class FixedFallbackPolicy(BasePredictorPolicy):
    """Fixed Text-Only (or lowest-cost available) Fallback Policy."""

    def route_single_query(
        self,
        features: Dict[str, torch.Tensor],
        modality_mask: Dict[str, bool],
        deadline_ms: float = 50.0,
        cost_budget: float = 3.5,
    ) -> Dict[str, Any]:
        available_mods = [m for m in ["text", "audio", "visual"] if modality_mask.get(m, True)]
        base_mod = "text" if "text" in available_mods else (available_mods[0] if available_mods else "text")
        key = mods_to_subset_key([base_mod]) if available_mods else "T"
        if not available_mods:
            return {"prediction": 0, "probabilities": np.array([1/3, 1/3, 1/3]), "final_subset": "T",
                    "spent_cost": 1.0, "spent_latency_ms": 1.0, "acquisition_path": ["T"], "routing_reason": "blackout"}

        probs, pred = self._predict_subset(key, features)
        return {
            "prediction": pred,
            "probabilities": probs,
            "final_subset": key,
            "spent_cost": float(SUBSET_METADATA[key]["cost"]),
            "spent_latency_ms": float(SUBSET_METADATA[key]["latency_ms"]),
            "acquisition_path": [key],
            "routing_reason": "fixed_fallback",
        }


class AlwaysOnFusionPolicy(BasePredictorPolicy):
    """Always-On Full Multimodal Fusion Policy."""

    def route_single_query(
        self,
        features: Dict[str, torch.Tensor],
        modality_mask: Dict[str, bool],
        deadline_ms: float = 50.0,
        cost_budget: float = 3.5,
    ) -> Dict[str, Any]:
        available_mods = [m for m in ["text", "audio", "visual"] if modality_mask.get(m, True)]
        if not available_mods:
            return {"prediction": 0, "probabilities": np.array([1/3, 1/3, 1/3]), "final_subset": "T",
                    "spent_cost": 1.0, "spent_latency_ms": 1.0, "acquisition_path": ["T"], "routing_reason": "blackout"}

        key = mods_to_subset_key(available_mods)
        # Check budget constraint
        if SUBSET_METADATA[key]["cost"] > cost_budget or SUBSET_METADATA[key]["latency_ms"] > deadline_ms:
            # Constrained fallback to best subset within budget
            feasible = [k for k in ["TA", "TV", "T", "A", "V"]
                        if set(SUBSET_METADATA[k]["mods"]).issubset(set(available_mods))
                        and SUBSET_METADATA[k]["cost"] <= cost_budget
                        and SUBSET_METADATA[k]["latency_ms"] <= deadline_ms]
            key = feasible[0] if feasible else "T"

        probs, pred = self._predict_subset(key, features)
        return {
            "prediction": pred,
            "probabilities": probs,
            "final_subset": key,
            "spent_cost": float(SUBSET_METADATA[key]["cost"]),
            "spent_latency_ms": float(SUBSET_METADATA[key]["latency_ms"]),
            "acquisition_path": [key],
            "routing_reason": f"always_on_fusion_{key}",
        }


class MatchedBudgetRandomPolicy(BasePredictorPolicy):
    """Matched-Budget Random Acquisition Baseline."""

    def __init__(self, predictor_bank: Dict[str, nn.Module], device: torch.device, p_acquire: float = 0.35, seed: int = 42):
        super().__init__(predictor_bank, device)
        self.p_acquire = float(p_acquire)
        self.rng = np.random.default_rng(seed)

    def route_single_query(
        self,
        features: Dict[str, torch.Tensor],
        modality_mask: Dict[str, bool],
        deadline_ms: float = 50.0,
        cost_budget: float = 3.5,
    ) -> Dict[str, Any]:
        available_mods = [m for m in ["text", "audio", "visual"] if modality_mask.get(m, True)]
        if not available_mods:
            return {"prediction": 0, "probabilities": np.array([1/3, 1/3, 1/3]), "final_subset": "T",
                    "spent_cost": 1.0, "spent_latency_ms": 1.0, "acquisition_path": ["T"], "routing_reason": "blackout"}

        # Base is text
        current_mods = ["text"] if "text" in available_mods else [available_mods[0]]
        # Randomly acquire other available modalities with probability p_acquire
        for m in available_mods:
            if m not in current_mods and self.rng.uniform() < self.p_acquire:
                tentative = current_mods + [m]
                key = mods_to_subset_key(tentative)
                if SUBSET_METADATA[key]["cost"] <= cost_budget and SUBSET_METADATA[key]["latency_ms"] <= deadline_ms:
                    current_mods.append(m)

        final_key = mods_to_subset_key(current_mods)
        probs, pred = self._predict_subset(final_key, features)
        return {
            "prediction": pred,
            "probabilities": probs,
            "final_subset": final_key,
            "spent_cost": float(SUBSET_METADATA[final_key]["cost"]),
            "spent_latency_ms": float(SUBSET_METADATA[final_key]["latency_ms"]),
            "acquisition_path": [final_key],
            "routing_reason": f"random_acquisition_{final_key}",
        }


class DerivedThresholdRouter(BasePredictorPolicy):
    """
    Derived-Threshold Sequential Policy.
    Proves structural equivalence by executing the exact algebraic confidence thresholds:
    tau(S, m, lambda) = (gamma - lambda * Delta C) / (gamma + nu)
    derived from V13's Expected Utility formulation.
    """

    def __init__(
        self,
        predictor_bank: Dict[str, nn.Module],
        device: torch.device,
        lambda_utility: float = 0.05,
        tau_stop_confidence: float = 0.75,
        custom_priors: Optional[Dict[Tuple[str, str], Dict[str, float]]] = None,
    ):
        super().__init__(predictor_bank, device)
        self.lambda_utility = float(lambda_utility)
        self.tau_stop_confidence = float(tau_stop_confidence)

        # Default to historical V13 priors if not specified
        if custom_priors is not None:
            self.priors = custom_priors
        else:
            from src.service.utility_router import EMPIRICAL_PRIORS
            self.priors = EMPIRICAL_PRIORS

        # Precompute algebraic thresholds tau(S, m, lambda)
        self.thresholds: Dict[Tuple[str, str], float] = {}
        for (src, mod), p_dict in self.priors.items():
            g = p_dict["gamma"]
            n = p_dict["nu"]
            c = MODALITY_COSTS[mod]
            if g + n > 0.0:
                tau = (g - self.lambda_utility * c) / (g + n)
            else:
                tau = 0.0
            self.thresholds[(src, mod)] = tau

    def route_single_query(
        self,
        features: Dict[str, torch.Tensor],
        modality_mask: Dict[str, bool],
        deadline_ms: float = 50.0,
        cost_budget: float = 3.5,
    ) -> Dict[str, Any]:
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

            # Rank candidates by linear marginal utility
            candidate_scores = {}
            for m in feasible_candidates:
                p_dict = self.priors.get((current_key, m), {"gamma": 0.20, "nu": 0.20})
                g, n = p_dict["gamma"], p_dict["nu"]
                c = MODALITY_COSTS[m]
                score = (g - (g + n) * p_max) / c
                candidate_scores[m] = score

            best_mod = max(candidate_scores, key=candidate_scores.get)
            tau = self.thresholds.get((current_key, best_mod), 0.0)

            # Equivalence condition: acquire iff p_max <= tau (and tau > 0)
            if p_max > tau or candidate_scores[best_mod] < self.lambda_utility:
                best_u = candidate_scores[best_mod]
                routing_reason = f"utility_threshold_unmet_{current_key}_best_{best_mod}_{best_u:.4f}"
                break

            current_mods.append(best_mod)
            current_key = mods_to_subset_key(current_mods)
            spent_cost += MODALITY_COSTS[best_mod]
            spent_latency += MODALITY_LATENCIES[best_mod]
            acquisition_path.append(current_key)

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
        }


class SequentialConfidenceRouter(BasePredictorPolicy):
    """
    Fair Sequential Confidence Routing Policy.
    Uses the exact same sequential action space and constraints as candidate utility policies:
    Starts at initial available modality (Text preferred), checks confidence threshold tau_stage1.
    If below tau_stage1, acquires next modality (Audio preferred, Visual fallback).
    Re-predicts with augmented subset; if below tau_stage2, acquires remaining modality.
    """

    def __init__(
        self,
        predictor_bank: Dict[str, nn.Module],
        device: torch.device,
        tau_stage1: float = 0.65,
        tau_stage2: float = 0.55,
        audio_first: bool = True,
    ):
        super().__init__(predictor_bank, device)
        self.tau_stage1 = float(tau_stage1)
        self.tau_stage2 = float(tau_stage2)
        self.audio_first = bool(audio_first)

    def route_single_query(
        self,
        features: Dict[str, torch.Tensor],
        modality_mask: Dict[str, bool],
        deadline_ms: float = 50.0,
        cost_budget: float = 3.5,
    ) -> Dict[str, Any]:
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
        p_max = float(np.max(probs))

        # Check stage 1 stopping
        if p_max >= self.tau_stage1:
            return {
                "prediction": pred,
                "probabilities": probs,
                "final_subset": current_key,
                "spent_cost": float(spent_cost),
                "spent_latency_ms": float(spent_latency),
                "acquisition_path": acquisition_path,
                "routing_reason": f"stage1_confident_{current_key}_{p_max:.3f}",
            }

        # Step 2: Acquire secondary modality if budget/deadline permits
        candidates = [m for m in available_mods if m not in current_mods]
        if self.audio_first:
            ordered_candidates = [m for m in ["audio", "visual", "text"] if m in candidates]
        else:
            ordered_candidates = [m for m in ["visual", "audio", "text"] if m in candidates]

        acquired_stage2 = False
        for mod in ordered_candidates:
            if (spent_cost + MODALITY_COSTS[mod] <= cost_budget) and (spent_latency + MODALITY_LATENCIES[mod] <= deadline_ms):
                current_mods.append(mod)
                current_key = mods_to_subset_key(current_mods)
                spent_cost += MODALITY_COSTS[mod]
                spent_latency += MODALITY_LATENCIES[mod]
                acquisition_path.append(current_key)
                probs, pred = self._predict_subset(current_key, features)
                p_max = float(np.max(probs))
                acquired_stage2 = True
                break

        if not acquired_stage2 or p_max >= self.tau_stage2:
            return {
                "prediction": pred,
                "probabilities": probs,
                "final_subset": current_key,
                "spent_cost": float(spent_cost),
                "spent_latency_ms": float(spent_latency),
                "acquisition_path": acquisition_path,
                "routing_reason": f"stage2_stop_{current_key}_{p_max:.3f}",
            }

        # Step 3: Acquire tertiary modality if remaining and affordable
        remaining_candidates = [m for m in available_mods if m not in current_mods]
        for mod in remaining_candidates:
            if (spent_cost + MODALITY_COSTS[mod] <= cost_budget) and (spent_latency + MODALITY_LATENCIES[mod] <= deadline_ms):
                current_mods.append(mod)
                current_key = mods_to_subset_key(current_mods)
                spent_cost += MODALITY_COSTS[mod]
                spent_latency += MODALITY_LATENCIES[mod]
                acquisition_path.append(current_key)
                probs, pred = self._predict_subset(current_key, features)
                break

        return {
            "prediction": pred,
            "probabilities": probs,
            "final_subset": current_key,
            "spent_cost": float(spent_cost),
            "spent_latency_ms": float(spent_latency),
            "acquisition_path": acquisition_path,
            "routing_reason": f"final_exhaustion_{current_key}",
        }


class StaticSubsetPolicy(BasePredictorPolicy):
    """
    Static Subset Policy (e.g. TA, TV, TAV).
    Acquires the target subset directly. Under missingness, falls back to largest available subset.
    """

    def __init__(self, predictor_bank: Dict[str, nn.Module], device: torch.device, target_subset: str = "TA"):
        super().__init__(predictor_bank, device)
        self.target_subset = target_subset
        self.desired_mods = SUBSET_METADATA[target_subset]["mods"]

    def route_single_query(
        self,
        features: Dict[str, torch.Tensor],
        modality_mask: Dict[str, bool],
        deadline_ms: float = 50.0,
        cost_budget: float = 3.5,
    ) -> Dict[str, Any]:
        # Filter desired mods by availability
        feasible_mods = [m for m in self.desired_mods if modality_mask.get(m, True)]
        if not feasible_mods:
            # Fallback to any available
            available = [m for m in ["text", "audio", "visual"] if modality_mask.get(m, True)]
            if not available:
                return {
                    "prediction": 0, "probabilities": np.array([1/3, 1/3, 1/3]),
                    "final_subset": "T", "spent_cost": 1.0, "spent_latency_ms": 1.0,
                    "acquisition_path": ["T"], "routing_reason": "blackout",
                }
            feasible_mods = [available[0]]

        # Respect budget/deadline
        subset_key = mods_to_subset_key(feasible_mods)
        if SUBSET_METADATA[subset_key]["cost"] > cost_budget or SUBSET_METADATA[subset_key]["latency_ms"] > deadline_ms:
            # Reduce to single text if possible
            if "text" in feasible_mods:
                feasible_mods = ["text"]
                subset_key = "T"

        probs, pred = self._predict_subset(subset_key, features)
        return {
            "prediction": pred,
            "probabilities": probs,
            "final_subset": subset_key,
            "spent_cost": float(SUBSET_METADATA[subset_key]["cost"]),
            "spent_latency_ms": float(SUBSET_METADATA[subset_key]["latency_ms"]),
            "acquisition_path": [subset_key],
            "routing_reason": f"static_{self.target_subset}_actual_{subset_key}",
        }

