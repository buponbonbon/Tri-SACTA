"""
src/service/budget.py: Latency and cost admission controller for Tri-SACTA V12.1.
Enforces realistic sensor acquisition costs and deadline guarantees before dispatching
additional modalities or neural networks.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple


DEFAULT_COST_WEIGHTS: Dict[str, float] = {
    "text": 1.0,
    "audio": 1.2,
    "visual": 1.3,
}

# Empirical latency estimates (in milliseconds) used for predictive admission control
DEFAULT_LATENCY_ESTIMATES_MS: Dict[str, float] = {
    "text_triage": 2.5,
    "audio_acquisition_and_eval": 5.0,
    "visual_acquisition_and_eval": 8.0,
    "multimodal_fusion": 12.0,
}


class BudgetController:
    """Manages runtime admission and budget compliance for multimodal sensor requests."""

    def __init__(
        self,
        cost_weights: Optional[Dict[str, float]] = None,
        latency_estimates_ms: Optional[Dict[str, float]] = None,
    ):
        self.cost_weights = cost_weights or dict(DEFAULT_COST_WEIGHTS)
        self.latency_estimates_ms = latency_estimates_ms or dict(DEFAULT_LATENCY_ESTIMATES_MS)

    def can_acquire(
        self,
        modality: str,
        remaining_time_ms: Optional[float],
        remaining_cost: Optional[float],
    ) -> bool:
        """Determines if acquiring a modality is admissible within remaining time and cost budgets."""
        # 1. Cost check
        req_cost = self.cost_weights.get(modality, 1.0)
        if remaining_cost is not None and (remaining_cost < req_cost - 1e-6):
            return False

        # 2. Latency check
        est_lat_key = f"{modality}_acquisition_and_eval"
        est_lat = self.latency_estimates_ms.get(est_lat_key, 6.0)
        if remaining_time_ms is not None and (remaining_time_ms < est_lat):
            return False

        return True

    def can_escalate_multimodal(
        self,
        modalities: List[str],
        remaining_time_ms: Optional[float],
        remaining_cost: Optional[float],
    ) -> bool:
        """Checks whether full multimodal fusion across specified modalities can meet deadlines and cost."""
        total_req_cost = sum(self.cost_weights.get(m, 1.0) for m in modalities)
        if remaining_cost is not None and (remaining_cost < total_req_cost - 1e-6):
            return False

        total_est_lat = self.latency_estimates_ms.get("multimodal_fusion", 12.0)
        if remaining_time_ms is not None and (remaining_time_ms < total_est_lat):
            return False

        return True

    def compute_cost(self, invoked_modalities: List[str]) -> float:
        """Computes total compute and sensor cost for a set of invoked modalities."""
        return float(sum(self.cost_weights.get(m, 1.0) for m in invoked_modalities))

    def check_compliance(
        self,
        elapsed_ms: float,
        total_cost: float,
        deadline_ms: Optional[float],
        cost_budget: Optional[float],
    ) -> Tuple[bool, bool]:
        """Returns (deadline_met, budget_met) boolean flags."""
        deadline_met = True if deadline_ms is None else (elapsed_ms <= deadline_ms + 1e-6)
        budget_met = True if cost_budget is None else (total_cost <= cost_budget + 1e-6)
        return deadline_met, budget_met
