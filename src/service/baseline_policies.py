"""
src/service/baseline_policies.py: Standardized baseline routing policies for Tri-SACTA V13.2.

All policies implement the standardized interface:
route_single_query(features, modality_mask, deadline_ms, cost_budget) -> Dict[str, Any]
using the exact same task-matched subset predictor bank and PyTorch device.

Enforces Phase I-B Evaluator & Runtime Correctness:
1. Propagates and checks exact per-query cost_budget and deadline_ms.
2. Fails closed on infeasible initial modality (cost_budget < initial cost or deadline < initial latency)
   or total blackout, returning explicit reject status (prediction = -1, spent_cost = 0.0).
3. Traps masked/unacquired feature reads via MaskedFeatureAccessViolation.
4. Separates Unrestricted Always-On from Budget-Admitted Max-Subset.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional, Tuple, Set
import numpy as np
import torch
import torch.nn as nn

from src.service.utility_router import (
    SUBSET_METADATA,
    MODALITY_COSTS,
    MODALITY_LATENCIES,
    mods_to_subset_key,
)

logger = logging.getLogger(__name__)


class MaskedFeatureAccessViolation(RuntimeError):
    """Raised when a policy attempts to read a feature tensor that is masked/unacquired."""
    pass


class BasePredictorPolicy:
    """Base class for all routing policies sharing a common predictor bank."""

    def __init__(self, predictor_bank: Dict[str, nn.Module], device: torch.device):
        self.predictor_bank = predictor_bank
        self.device = device
        for m in self.predictor_bank.values():
            m.eval()
            m.to(self.device)

    def _predict_subset(
        self,
        subset_key: str,
        features: Dict[str, torch.Tensor],
        modality_mask: Optional[Dict[str, bool]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Executes forward inference for the given subset key.
        If modality_mask is supplied, enforces that no masked modality is accessed.
        """
        if subset_key not in self.predictor_bank:
            raise KeyError(f"Subset model '{subset_key}' not present in predictor bank.")

        model = self.predictor_bank[subset_key]
        mods = SUBSET_METADATA[subset_key]["mods"]

        # Enforce masked feature access trap
        if modality_mask is not None:
            for m in mods:
                if not modality_mask.get(m, True):
                    raise MaskedFeatureAccessViolation(
                        f"Attempted to access masked/unavailable modality '{m}' for subset '{subset_key}'."
                    )

        tensors = []
        for m in mods:
            if m not in features:
                raise KeyError(f"Feature tensor for modality '{m}' not found in query features.")
            tensors.append(features[m].to(self.device))

        concat_feat = torch.cat(tensors, dim=-1)

        with torch.no_grad():
            logits = model(concat_feat)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            pred = int(np.argmax(probs))
        return probs, pred


class FixedFallbackPolicy(BasePredictorPolicy):
    """Fixed Text-Only (or lowest-cost available) Fallback Baseline."""

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
                "prediction": -1,
                "probabilities": np.zeros(3, dtype=np.float32),
                "final_subset": "NONE",
                "spent_cost": 0.0,
                "spent_latency_ms": 0.0,
                "acquisition_path": [],
                "routing_reason": "blackout_no_modalities",
                "status": "REJECTED_BLACKOUT",
                "budget_violated": False,
                "deadline_violated": False,
            }

        base_mod = "text" if "text" in available_mods else available_mods[0]
        key = mods_to_subset_key([base_mod])
        c = float(SUBSET_METADATA[key]["cost"])
        l = float(SUBSET_METADATA[key]["latency_ms"])

        if cost_budget < c or deadline_ms < l:
            return {
                "prediction": -1,
                "probabilities": np.zeros(3, dtype=np.float32),
                "final_subset": "NONE",
                "spent_cost": 0.0,
                "spent_latency_ms": 0.0,
                "acquisition_path": [],
                "routing_reason": "infeasible_budget_or_deadline",
                "status": "REJECTED_INFEASIBLE",
                "budget_violated": False,
                "deadline_violated": False,
            }

        probs, pred = self._predict_subset(key, features, modality_mask)
        return {
            "prediction": pred,
            "probabilities": probs,
            "final_subset": key,
            "spent_cost": c,
            "spent_latency_ms": l,
            "acquisition_path": [key],
            "routing_reason": "fixed_fallback",
            "status": "SUCCESS",
            "budget_violated": False,
            "deadline_violated": False,
        }


class AlwaysOnFusionPolicy(BasePredictorPolicy):
    """
    Unrestricted Always-On Full Multimodal Fusion Policy.
    Transmits all available modalities and executes full fusion.
    Reports resource violations rather than imposing an artificial early cutoff.
    """

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
                "prediction": -1,
                "probabilities": np.zeros(3, dtype=np.float32),
                "final_subset": "NONE",
                "spent_cost": 0.0,
                "spent_latency_ms": 0.0,
                "acquisition_path": [],
                "routing_reason": "blackout_no_modalities",
                "status": "REJECTED_BLACKOUT",
                "budget_violated": False,
                "deadline_violated": False,
            }

        target_key = mods_to_subset_key(available_mods)
        spent_cost = float(SUBSET_METADATA[target_key]["cost"])
        spent_lat = float(SUBSET_METADATA[target_key]["latency_ms"])

        probs, pred = self._predict_subset(target_key, features, modality_mask)
        b_violated = spent_cost > cost_budget
        d_violated = spent_lat > deadline_ms

        return {
            "prediction": pred,
            "probabilities": probs,
            "final_subset": target_key,
            "spent_cost": spent_cost,
            "spent_latency_ms": spent_lat,
            "acquisition_path": [target_key],
            "routing_reason": f"always_on_{target_key}",
            "status": "SUCCESS",
            "budget_violated": b_violated,
            "deadline_violated": d_violated,
        }


class BudgetAdmittedMaxSubsetPolicy(BasePredictorPolicy):
    """
    Budget-Admitted Max-Subset Policy.
    Chooses the best feasible static subset within the given cost_budget and
    deadline_ms constraint without sequential multi-stage triage overhead.
    """
    RANKED_SUBSETS = ["TAV", "TA", "TV", "AV", "T", "A", "V"]

    def route_single_query(
        self,
        features: Dict[str, torch.Tensor],
        modality_mask: Dict[str, bool],
        deadline_ms: float = 50.0,
        cost_budget: float = 3.5,
    ) -> Dict[str, Any]:
        available_mods = set(m for m in ["text", "audio", "visual"] if modality_mask.get(m, True))
        if not available_mods:
            return {
                "prediction": -1,
                "probabilities": np.zeros(3, dtype=np.float32),
                "final_subset": "NONE",
                "spent_cost": 0.0,
                "spent_latency_ms": 0.0,
                "acquisition_path": [],
                "routing_reason": "blackout_no_modalities",
                "status": "REJECTED_BLACKOUT",
                "budget_violated": False,
                "deadline_violated": False,
            }

        chosen_key = None
        for sub in self.RANKED_SUBSETS:
            mods = SUBSET_METADATA[sub]["mods"]
            if set(mods).issubset(available_mods):
                c = SUBSET_METADATA[sub]["cost"]
                l = SUBSET_METADATA[sub]["latency_ms"]
                if c <= cost_budget and l <= deadline_ms:
                    chosen_key = sub
                    break

        if chosen_key is None:
            return {
                "prediction": -1,
                "probabilities": np.zeros(3, dtype=np.float32),
                "final_subset": "NONE",
                "spent_cost": 0.0,
                "spent_latency_ms": 0.0,
                "acquisition_path": [],
                "routing_reason": "budget_admission_failed_no_feasible_subset",
                "status": "REJECTED_INFEASIBLE",
                "budget_violated": False,
                "deadline_violated": False,
            }

        probs, pred = self._predict_subset(chosen_key, features, modality_mask)
        return {
            "prediction": pred,
            "probabilities": probs,
            "final_subset": chosen_key,
            "spent_cost": float(SUBSET_METADATA[chosen_key]["cost"]),
            "spent_latency_ms": float(SUBSET_METADATA[chosen_key]["latency_ms"]),
            "acquisition_path": [chosen_key],
            "routing_reason": f"admitted_max_subset_{chosen_key}",
            "status": "SUCCESS",
            "budget_violated": False,
            "deadline_violated": False,
        }


class StaticSubsetPolicy(BasePredictorPolicy):
    """Static Single-Subset Policy (e.g., T, A, V, TA, TV, AV, TAV) with Feasible Fallback."""

    def __init__(self, predictor_bank: Dict[str, nn.Module], device: torch.device, target_subset: str = "TA"):
        super().__init__(predictor_bank, device)
        self.target_subset = str(target_subset).upper()
        if self.target_subset not in SUBSET_METADATA:
            raise ValueError(f"Unknown target subset: {target_subset}")

    def route_single_query(
        self,
        features: Dict[str, torch.Tensor],
        modality_mask: Dict[str, bool],
        deadline_ms: float = 50.0,
        cost_budget: float = 3.5,
    ) -> Dict[str, Any]:
        available_mods = set(m for m in ["text", "audio", "visual"] if modality_mask.get(m, True))
        target_mods = set(SUBSET_METADATA[self.target_subset]["mods"])

        # Check if target subset is available and affordable
        c = SUBSET_METADATA[self.target_subset]["cost"]
        l = SUBSET_METADATA[self.target_subset]["latency_ms"]

        if target_mods.issubset(available_mods) and c <= cost_budget and l <= deadline_ms:
            key = self.target_subset
            reason = f"static_{self.target_subset}"
        else:
            # Fallback to cheapest feasible available subset
            candidates = ["T", "A", "V"]
            feasible = [
                k for k in candidates
                if set(SUBSET_METADATA[k]["mods"]).issubset(available_mods)
                and SUBSET_METADATA[k]["cost"] <= cost_budget
                and SUBSET_METADATA[k]["latency_ms"] <= deadline_ms
            ]
            if not feasible:
                return {
                    "prediction": -1,
                    "probabilities": np.zeros(3, dtype=np.float32),
                    "final_subset": "NONE",
                    "spent_cost": 0.0,
                    "spent_latency_ms": 0.0,
                    "acquisition_path": [],
                    "routing_reason": "static_subset_unaffordable_and_no_fallback",
                    "status": "REJECTED_INFEASIBLE",
                    "budget_violated": False,
                    "deadline_violated": False,
                }
            key = feasible[0]
            reason = f"static_{self.target_subset}_fallback_to_{key}"

        spent_c = float(SUBSET_METADATA[key]["cost"])
        spent_l = float(SUBSET_METADATA[key]["latency_ms"])
        probs, pred = self._predict_subset(key, features, modality_mask)
        return {
            "prediction": pred,
            "probabilities": probs,
            "final_subset": key,
            "spent_cost": spent_c,
            "spent_latency_ms": spent_l,
            "acquisition_path": [key],
            "routing_reason": reason,
            "status": "SUCCESS",
            "budget_violated": False,
            "deadline_violated": False,
        }


class ConfidenceRoutingPolicy(BasePredictorPolicy):
    """Standard One-Shot Confidence-Based Routing Baseline."""

    def __init__(self, predictor_bank: Dict[str, nn.Module], device: torch.device, tau_confidence: float = 0.55):
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
            return {
                "prediction": -1,
                "probabilities": np.zeros(3, dtype=np.float32),
                "final_subset": "NONE",
                "spent_cost": 0.0,
                "spent_latency_ms": 0.0,
                "acquisition_path": [],
                "routing_reason": "blackout_no_modalities",
                "status": "REJECTED_BLACKOUT",
                "budget_violated": False,
                "deadline_violated": False,
            }

        base_mod = "text" if "text" in available_mods else available_mods[0]
        base_key = mods_to_subset_key([base_mod])
        spent_cost = float(SUBSET_METADATA[base_key]["cost"])
        spent_lat = float(SUBSET_METADATA[base_key]["latency_ms"])

        if cost_budget < spent_cost or deadline_ms < spent_lat:
            return {
                "prediction": -1,
                "probabilities": np.zeros(3, dtype=np.float32),
                "final_subset": "NONE",
                "spent_cost": 0.0,
                "spent_latency_ms": 0.0,
                "acquisition_path": [],
                "routing_reason": "infeasible_budget_or_deadline",
                "status": "REJECTED_INFEASIBLE",
                "budget_violated": False,
                "deadline_violated": False,
            }

        probs, pred = self._predict_subset(base_key, features, modality_mask)
        conf = float(np.max(probs))
        path = [base_key]

        if conf < self.tau_confidence:
            target_key = mods_to_subset_key(available_mods)
            target_cost = float(SUBSET_METADATA[target_key]["cost"])
            target_lat = float(SUBSET_METADATA[target_key]["latency_ms"])

            if target_cost <= cost_budget and target_lat <= deadline_ms and target_key != base_key:
                probs, pred = self._predict_subset(target_key, features, modality_mask)
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
            "spent_cost": spent_cost,
            "spent_latency_ms": spent_lat,
            "acquisition_path": path,
            "routing_reason": reason,
            "status": "SUCCESS",
            "budget_violated": False,
            "deadline_violated": False,
        }


class SequentialConfidenceRouter(BasePredictorPolicy):
    """
    Candidate Release Router: Multi-Stage Sequential Confidence Routing.
    Uses exact sequential action space and constraints:
    Stage 1: Evaluates base modality (prefer Text). If p_max >= tau_stage1, stop.
    Stage 2: Acquires secondary modality (Audio preferred if audio_first, else Visual) if budget allows.
             Re-predicts. If p_max >= tau_stage2, stop.
    Stage 3: Acquires tertiary modality if remaining and affordable.
    """

    def __init__(
        self,
        predictor_bank: Dict[str, nn.Module],
        device: torch.device,
        tau_stage1: float = 0.55,
        tau_stage2: float = 0.45,
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
                "prediction": -1,
                "probabilities": np.zeros(3, dtype=np.float32),
                "final_subset": "NONE",
                "spent_cost": 0.0,
                "spent_latency_ms": 0.0,
                "acquisition_path": [],
                "routing_reason": "blackout_no_modalities",
                "status": "REJECTED_BLACKOUT",
                "budget_violated": False,
                "deadline_violated": False,
            }

        base_mod = "text" if "text" in available_mods else available_mods[0]
        current_mods = [base_mod]
        current_key = mods_to_subset_key(current_mods)
        spent_cost = float(SUBSET_METADATA[current_key]["cost"])
        spent_lat = float(SUBSET_METADATA[current_key]["latency_ms"])

        if cost_budget < spent_cost or deadline_ms < spent_lat:
            return {
                "prediction": -1,
                "probabilities": np.zeros(3, dtype=np.float32),
                "final_subset": "NONE",
                "spent_cost": 0.0,
                "spent_latency_ms": 0.0,
                "acquisition_path": [],
                "routing_reason": "infeasible_budget_or_deadline",
                "status": "REJECTED_INFEASIBLE",
                "budget_violated": False,
                "deadline_violated": False,
            }

        acquisition_path = [current_key]
        probs, pred = self._predict_subset(current_key, features, modality_mask)
        p_max = float(np.max(probs))

        # Check stage 1 stopping
        if p_max >= self.tau_stage1:
            return {
                "prediction": pred,
                "probabilities": probs,
                "final_subset": current_key,
                "spent_cost": spent_cost,
                "spent_latency_ms": spent_lat,
                "acquisition_path": acquisition_path,
                "routing_reason": f"stage1_confident_{current_key}_{p_max:.3f}",
                "status": "SUCCESS",
                "budget_violated": False,
                "deadline_violated": False,
            }

        # Step 2: Acquire secondary modality if affordable
        candidates = [m for m in available_mods if m not in current_mods]
        if self.audio_first:
            ordered_candidates = [m for m in ["audio", "visual", "text"] if m in candidates]
        else:
            ordered_candidates = [m for m in ["visual", "audio", "text"] if m in candidates]

        acquired_stage2 = False
        for mod in ordered_candidates:
            c_add = MODALITY_COSTS[mod]
            l_add = MODALITY_LATENCIES[mod]
            if (spent_cost + c_add <= cost_budget) and (spent_lat + l_add <= deadline_ms):
                current_mods.append(mod)
                current_key = mods_to_subset_key(current_mods)
                spent_cost += c_add
                spent_lat += l_add
                acquisition_path.append(current_key)
                probs, pred = self._predict_subset(current_key, features, modality_mask)
                p_max = float(np.max(probs))
                acquired_stage2 = True
                break

        if not acquired_stage2 or p_max >= self.tau_stage2:
            return {
                "prediction": pred,
                "probabilities": probs,
                "final_subset": current_key,
                "spent_cost": spent_cost,
                "spent_latency_ms": spent_lat,
                "acquisition_path": acquisition_path,
                "routing_reason": f"stage2_stop_{current_key}_{p_max:.3f}",
                "status": "SUCCESS",
                "budget_violated": False,
                "deadline_violated": False,
            }

        # Step 3: Acquire tertiary modality if remaining and affordable
        remaining_candidates = [m for m in available_mods if m not in current_mods]
        for mod in remaining_candidates:
            c_add = MODALITY_COSTS[mod]
            l_add = MODALITY_LATENCIES[mod]
            if (spent_cost + c_add <= cost_budget) and (spent_lat + l_add <= deadline_ms):
                current_mods.append(mod)
                current_key = mods_to_subset_key(current_mods)
                spent_cost += c_add
                spent_lat += l_add
                acquisition_path.append(current_key)
                probs, pred = self._predict_subset(current_key, features, modality_mask)
                break

        return {
            "prediction": pred,
            "probabilities": probs,
            "final_subset": current_key,
            "spent_cost": spent_cost,
            "spent_latency_ms": spent_lat,
            "acquisition_path": acquisition_path,
            "routing_reason": f"stage3_stop_{current_key}_{float(np.max(probs)):.3f}",
            "status": "SUCCESS",
            "budget_violated": False,
            "deadline_violated": False,
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

        if custom_priors is not None:
            self.priors = custom_priors
        else:
            from src.service.utility_router import EMPIRICAL_PRIORS
            self.priors = EMPIRICAL_PRIORS

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
                "prediction": -1,
                "probabilities": np.zeros(3, dtype=np.float32),
                "final_subset": "NONE",
                "spent_cost": 0.0,
                "spent_latency_ms": 0.0,
                "acquisition_path": [],
                "routing_reason": "blackout_no_modalities",
                "status": "REJECTED_BLACKOUT",
                "budget_violated": False,
                "deadline_violated": False,
            }

        base_mod = "text" if "text" in available_mods else available_mods[0]
        current_mods = [base_mod]
        current_key = mods_to_subset_key(current_mods)
        spent_cost = float(SUBSET_METADATA[current_key]["cost"])
        spent_lat = float(SUBSET_METADATA[current_key]["latency_ms"])

        if cost_budget < spent_cost or deadline_ms < spent_lat:
            return {
                "prediction": -1,
                "probabilities": np.zeros(3, dtype=np.float32),
                "final_subset": "NONE",
                "spent_cost": 0.0,
                "spent_latency_ms": 0.0,
                "acquisition_path": [],
                "routing_reason": "infeasible_budget_or_deadline",
                "status": "REJECTED_INFEASIBLE",
                "budget_violated": False,
                "deadline_violated": False,
            }

        acquisition_path = [current_key]
        probs, pred = self._predict_subset(current_key, features, modality_mask)

        while True:
            p_max = float(np.max(probs))
            if p_max >= self.tau_stop_confidence:
                reason = f"confident_stopping_{current_key}_pmax_{p_max:.3f}"
                break

            candidates = [m for m in available_mods if m not in current_mods]
            feasible = [
                m for m in candidates
                if (spent_cost + MODALITY_COSTS[m] <= cost_budget)
                and (spent_lat + MODALITY_LATENCIES[m] <= deadline_ms)
            ]
            if not feasible:
                reason = f"budget_exhausted_{current_key}"
                break

            # Evaluate algebraic thresholds
            qualifying = []
            for m in feasible:
                tau = self.thresholds.get((current_key, m), 0.0)
                if p_max <= tau:
                    qualifying.append((m, tau))

            if not qualifying:
                reason = f"threshold_unmet_{current_key}_pmax_{p_max:.3f}"
                break

            # Choose mod with highest priority (lowest cost or highest tau)
            best_mod = max(qualifying, key=lambda x: x[1])[0]
            current_mods.append(best_mod)
            current_key = mods_to_subset_key(current_mods)
            spent_cost += MODALITY_COSTS[best_mod]
            spent_lat += MODALITY_LATENCIES[best_mod]
            acquisition_path.append(current_key)
            probs, pred = self._predict_subset(current_key, features, modality_mask)

        return {
            "prediction": pred,
            "probabilities": probs,
            "final_subset": current_key,
            "spent_cost": spent_cost,
            "spent_latency_ms": spent_lat,
            "acquisition_path": acquisition_path,
            "routing_reason": reason,
            "status": "SUCCESS",
            "budget_violated": False,
            "deadline_violated": False,
        }


class MatchedBudgetRandomPolicy(BasePredictorPolicy):
    """
    Matched-Budget Random Acquisition Baseline.
    Parameterizes acquisition probability p_acquire calibrated on validation
    to match target operational resource budgets.
    """

    def __init__(
        self,
        predictor_bank: Dict[str, nn.Module],
        device: torch.device,
        p_acquire: float = 0.35,
        seed: int = 42,
    ):
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
            return {
                "prediction": -1,
                "probabilities": np.zeros(3, dtype=np.float32),
                "final_subset": "NONE",
                "spent_cost": 0.0,
                "spent_latency_ms": 0.0,
                "acquisition_path": [],
                "routing_reason": "blackout_no_modalities",
                "status": "REJECTED_BLACKOUT",
                "budget_violated": False,
                "deadline_violated": False,
            }

        base_mod = "text" if "text" in available_mods else available_mods[0]
        current_mods = [base_mod]
        current_key = mods_to_subset_key(current_mods)
        spent_cost = float(SUBSET_METADATA[current_key]["cost"])
        spent_lat = float(SUBSET_METADATA[current_key]["latency_ms"])

        if cost_budget < spent_cost or deadline_ms < spent_lat:
            return {
                "prediction": -1,
                "probabilities": np.zeros(3, dtype=np.float32),
                "final_subset": "NONE",
                "spent_cost": 0.0,
                "spent_latency_ms": 0.0,
                "acquisition_path": [],
                "routing_reason": "infeasible_budget_or_deadline",
                "status": "REJECTED_INFEASIBLE",
                "budget_violated": False,
                "deadline_violated": False,
            }

        acquisition_path = [current_key]
        for m in available_mods:
            if m not in current_mods and self.rng.uniform() < self.p_acquire:
                c_add = MODALITY_COSTS[m]
                l_add = MODALITY_LATENCIES[m]
                if (spent_cost + c_add <= cost_budget) and (spent_lat + l_add <= deadline_ms):
                    current_mods.append(m)
                    spent_cost += c_add
                    spent_lat += l_add
                    current_key = mods_to_subset_key(current_mods)
                    acquisition_path.append(current_key)

        probs, pred = self._predict_subset(current_key, features, modality_mask)
        return {
            "prediction": pred,
            "probabilities": probs,
            "final_subset": current_key,
            "spent_cost": spent_cost,
            "spent_latency_ms": spent_lat,
            "acquisition_path": acquisition_path,
            "routing_reason": f"random_acquisition_{current_key}",
            "status": "SUCCESS",
            "budget_violated": False,
            "deadline_violated": False,
        }
