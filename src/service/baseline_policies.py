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
