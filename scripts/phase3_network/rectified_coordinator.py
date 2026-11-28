"""
scripts/phase3_network/rectified_coordinator.py: Rectified Network-Aware Coordinator
extending DistributedCoordinator WITHOUT modifying frozen src/service/distributed_coordinator.py.

Injects controlled application-layer RTT/bandwidth/jitter DURING fetch_remote_modality()
and records monotonic per-request CommunicationEvent traces before returning to routing logic.
"""

from __future__ import annotations

import os
import sys
import time
import uuid
import zlib
import random
from typing import Dict, Any, List, Optional
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.service.distributed_coordinator import DistributedCoordinator, ModalityFetchResult
from scripts.phase3_network.network_emulator import NetworkCondition, emulate_rtt_and_bandwidth


class RectifiedNetworkCoordinator(DistributedCoordinator):
    """
    Subclass of DistributedCoordinator that injects network emulation INSIDE
    fetch_remote_modality() and records monotonic communication events.
    """

    def __init__(
        self,
        predictor_bank: Dict[str, torch.nn.Module],
        device: torch.device,
        network_condition: NetworkCondition,
        audio_host: str = "127.0.0.1",
        audio_port: int = 8765,
        visual_host: str = "127.0.0.1",
        visual_port: int = 8766,
        persistent_connections: bool = True,
        inference_reserve_ms: float = 0.2,
        jitter_seed: int = 1320,
    ):
        super().__init__(
            predictor_bank=predictor_bank,
            device=device,
            audio_host=audio_host,
            audio_port=audio_port,
            visual_host=visual_host,
            visual_port=visual_port,
            persistent_connections=persistent_connections,
            inference_reserve_ms=inference_reserve_ms,
        )
        self.network_condition = network_condition
        self._jitter_rng = random.Random(jitter_seed)
        self._parent_request_id: str = ""
        self._base_jitter_seed: int = jitter_seed
        self.current_events: List[Dict[str, Any]] = []

    def try_fetch_remote_modality(
        self,
        modality: str,
        sample_idx: int,
        timeout_s: float = 2.0,
    ) -> ModalityFetchResult:
        req_id = str(uuid.uuid4())[:12]
        dispatch_ts = time.perf_counter()
        rem_start_ms = timeout_s * 1000.0

        res = super().try_fetch_remote_modality(modality, sample_idx, timeout_s=timeout_s)
        elapsed_socket_s = time.perf_counter() - dispatch_ts
        rem_after_socket_s = max(0.0, timeout_s - elapsed_socket_s)

        app_b = int(res.sent_bytes + res.recv_bytes)
        em = emulate_rtt_and_bandwidth(
            total_bytes=app_b,
            cond=self.network_condition,
            rng=self._jitter_rng,
            deadline_remaining_s=rem_after_socket_s,
        )
        complete_ts = time.perf_counter()
        total_elapsed_ms = (complete_ts - dispatch_ts) * 1000.0
        rem_end_ms = max(0.0, rem_start_ms - total_elapsed_ms)

        if em["timed_out"] and res.ok:
            final_res = ModalityFetchResult(
                None,
                total_elapsed_ms,
                app_b,
                res.payload_bytes,
                ok=False,
                error_code="TIMEOUT_DEADLINE_EXCEEDED",
                sent_bytes=res.sent_bytes,
                recv_bytes=res.recv_bytes,
            )
            outcome = "TIMEOUT_AFTER_SOCKET"
        else:
            final_res = res
            outcome = "SUCCESS" if res.ok else f"FAILED_{res.error_code}"

        self.current_events.append({
            "request_id": req_id,
            "parent_request_id": self._parent_request_id,
            "modality": modality,
            "sample_idx": int(sample_idx),
            "dispatch_ts": dispatch_ts,
            "complete_ts": complete_ts,
            "elapsed_ms": total_elapsed_ms,
            "remaining_deadline_start_ms": rem_start_ms,
            "remaining_deadline_end_ms": rem_end_ms,
            "outcome": outcome,
            "app_bytes": app_b,
            "payload_bytes": int(final_res.payload_bytes),
            "target_rtt_ms": float(self.network_condition.target_rtt_ms),
        })
        return final_res

    def process_query_with_events(
        self,
        sample_idx: int,
        text_feature: torch.Tensor,
        policy_mode: str = "sequential_confidence",
        tau_stage1: float = 0.55,
        tau_stage2: float = 0.45,
        deadline_ms: float = 50.0,
        cost_budget: float = 3.5,
        modality_mask: Optional[Dict[str, bool]] = None,
        rng: Optional[Any] = None,
        p_acquire: float = 0.28,
        parent_request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._parent_request_id = parent_request_id or str(uuid.uuid4())[:12]
        # Paired exogenous jitter keyed by deterministic scenario/sample identity, independent of policy call count
        det_seed = zlib.crc32(f"{self._base_jitter_seed}:{sample_idx}:{self._parent_request_id}".encode("utf-8"))
        self._jitter_rng = random.Random(det_seed)
        self.current_events = []
        res = super().process_query(
            sample_idx=sample_idx,
            text_feature=text_feature,
            policy_mode=policy_mode,
            tau_stage1=tau_stage1,
            tau_stage2=tau_stage2,
            deadline_ms=deadline_ms,
            cost_budget=cost_budget,
            modality_mask=modality_mask,
            rng=rng,
            p_acquire=p_acquire,
        )
        events = list(self.current_events)
        res["events"] = events
        res["remote_attempted"] = len(events)
        res["remote_succeeded"] = sum(1 for e in events if e["outcome"] == "SUCCESS")
        res["remote_timeouts"] = sum(1 for e in events if "TIMEOUT" in e["outcome"])
        return res
