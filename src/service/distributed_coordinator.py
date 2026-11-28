"""
src/service/distributed_coordinator.py: Unified Distributed Orchestrator for Tri-SACTA V13.2.

Shares the exact same decision semantics, action thresholds, budget admission rules,
and masked feature guards across offline simulation and real TCP socket execution:
1. always_on / always_on_serial: Direct fetch Audio + Visual serially -> 1 TAV forward pass.
2. always_on_parallel: Optimized parallel ThreadPoolExecutor fetch Audio + Visual -> 1 TAV pass.
3. static_ta: Direct TA (fetch Audio -> 1 TA pass), with feasible fallback when Audio blocked/budgeted.
4. budget_admitted_max: Best feasible subset in [TAV, TA, TV, AV, T, A, V] within budget/deadline.
5. confidence_routing_055: One-shot confidence routing (Text local -> parallel escalation if p_max < 0.55).
6. sequential_confidence: Candidate SequentialConfidence_0.55_0.45 (Text -> Audio -> Visual).
7. unimodal_text: Local Text (0 remote wire bytes) with feasible fallback/rejection.
8. matched_budget_random: Calibrated random acquisition at matched budget probability.

Supports BOTH persistent keep-alive TCP connections and per-request TCP connections,
strict absolute monotonic deadline reserves, exact client-side protocol byte accounting
(including error frames & failed/partial attempts), and fail-closed fallback on provider faults.
"""

from __future__ import annotations

import os
import sys
import socket
import struct
import json
import time
import threading
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.loader import SUBSET_METADATA
from src.service.utility_router import MODALITY_COSTS, MODALITY_LATENCIES, mods_to_subset_key
from src.service.baseline_policies import MaskedFeatureAccessViolation
from src.service.distributed_providers import (
    recv_exact,
    MAX_CONTROL_FRAME_BYTES,
    MAX_PAYLOAD_BYTES,
    ProtocolFrameError,
)

logger = logging.getLogger(__name__)

# Reserve 0.2 ms for final PyTorch forward pass so network read does not eat 100% of deadline
INFERENCE_TIME_RESERVE_S = 0.0002


class ModalityFetchResult(tuple):
    """
    3-tuple containing (feat_tensor, rtt_ms, total_app_bytes) for legacy unpacking parity,
    carrying .payload_bytes, .ok, .error_code, and .sent_bytes attributes.
    """
    def __new__(
        cls,
        feat_tensor: Optional[torch.Tensor],
        rtt_ms: float,
        total_app_bytes: int,
        payload_bytes: int = 0,
        ok: bool = True,
        error_code: Optional[str] = None,
        sent_bytes: int = 0,
        recv_bytes: int = 0,
    ):
        obj = super().__new__(cls, (feat_tensor, rtt_ms, total_app_bytes))
        obj.payload_bytes = payload_bytes
        obj.ok = ok
        obj.error_code = error_code
        obj.sent_bytes = sent_bytes
        obj.recv_bytes = recv_bytes
        return obj


class DistributedCoordinator:
    """Orchestrates edge-cloud multimodal acquisition over real TCP sockets."""

    def __init__(
        self,
        predictor_bank: Dict[str, nn.Module],
        device: torch.device,
        audio_host: str = "127.0.0.1",
        audio_port: int = 8765,
        visual_host: str = "127.0.0.1",
        visual_port: int = 8766,
        persistent_connections: bool = False,
        inference_reserve_ms: float = 0.2,
    ):
        self.predictor_bank = predictor_bank
        self.device = device
        self.provider_endpoints = {
            "audio": (audio_host, audio_port),
            "visual": (visual_host, visual_port),
        }
        self.persistent_connections = bool(persistent_connections)
        self.inference_reserve_s = float(inference_reserve_ms) / 1000.0
        self._conn_pool: Dict[str, Optional[socket.socket]] = {"audio": None, "visual": None}
        self._conn_locks: Dict[str, threading.Lock] = {"audio": threading.Lock(), "visual": threading.Lock()}
        self._pool_executor = ThreadPoolExecutor(max_workers=2)

        for m in self.predictor_bank.values():
            m.eval()
            m.to(self.device)

    def close(self) -> None:
        for mod in ("audio", "visual"):
            with self._conn_locks[mod]:
                if self._conn_pool[mod] is not None:
                    try:
                        self._conn_pool[mod].close()
                    except Exception:
                        pass
                    self._conn_pool[mod] = None
        self._pool_executor.shutdown(wait=False)

    def _get_or_connect(self, modality: str, timeout_s: float) -> socket.socket:
        host, port = self.provider_endpoints[modality]
        if self.persistent_connections and self._conn_pool[modality] is not None:
            return self._conn_pool[modality]
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(max(0.0005, timeout_s))
        sock.connect((host, port))
        if self.persistent_connections:
            self._conn_pool[modality] = sock
        return sock

    def _drop_persistent_conn(self, modality: str) -> None:
        if self._conn_pool[modality] is not None:
            try:
                self._conn_pool[modality].close()
            except Exception:
                pass
            self._conn_pool[modality] = None

    def fetch_remote_modality(
        self,
        modality: str,
        sample_idx: int,
        timeout_s: float = 2.0,
    ) -> ModalityFetchResult:
        """
        Backward-compatible fetch method: raises RuntimeError/OSError on hard failure,
        returning ModalityFetchResult(tensor, rtt_ms, app_bytes, payload_bytes).
        """
        res = self.try_fetch_remote_modality(modality, sample_idx, timeout_s=timeout_s)
        if not res.ok:
            raise RuntimeError(f"Remote {modality} fetch failed ({res.error_code})")
        return res

    def try_fetch_remote_modality(
        self,
        modality: str,
        sample_idx: int,
        timeout_s: float = 2.0,
    ) -> ModalityFetchResult:
        """
        Safe bounded TCP socket retrieval returning ModalityFetchResult with exact sent/recv
        application-protocol byte accounting even when disconnects, timeouts, or errors occur.
        """
        if timeout_s <= 0.0:
            return ModalityFetchResult(None, 0.0, 0, 0, ok=False, error_code="DEADLINE_EXPIRED_BEFORE_CONNECT")

        req_obj = {"sample_idx": int(sample_idx), "modality": modality}
        req_bytes = json.dumps(req_obj).encode("utf-8")
        req_hdr = struct.pack("!I", len(req_bytes))
        full_req = req_hdr + req_bytes

        t_start = time.perf_counter()
        deadline_abs = t_start + timeout_s
        sent_bytes = 0
        recv_bytes = 0
        sock: Optional[socket.socket] = None

        with self._conn_locks[modality]:
            try:
                rem = deadline_abs - time.perf_counter()
                if rem <= 0.0:
                    return ModalityFetchResult(None, 0.0, 0, 0, ok=False, error_code="DEADLINE_EXPIRED")

                sock = self._get_or_connect(modality, rem)
                sock.settimeout(max(0.0005, deadline_abs - time.perf_counter()))
                sock.sendall(full_req)
                sent_bytes += len(full_req)

                raw_hdr = recv_exact(sock, 12, timeout_s=max(0.0, deadline_abs - time.perf_counter()))
                recv_bytes += len(raw_hdr)
                status_code, meta_len, payload_len = struct.unpack("!III", raw_hdr)

                if meta_len > MAX_CONTROL_FRAME_BYTES or payload_len > MAX_PAYLOAD_BYTES:
                    raise ProtocolFrameError(
                        f"Oversized frame reported: meta_len={meta_len}, payload_len={payload_len}"
                    )

                raw_meta = recv_exact(sock, meta_len, timeout_s=max(0.0, deadline_abs - time.perf_counter()))
                recv_bytes += len(raw_meta)
                meta = json.loads(raw_meta.decode("utf-8"))

                if status_code != 200 or meta.get("status") != "OK":
                    err_code = meta.get("error_code", f"HTTP_{status_code}")
                    if not self.persistent_connections:
                        sock.close()
                    rtt_ms = (time.perf_counter() - t_start) * 1000.0
                    return ModalityFetchResult(
                        None, rtt_ms, sent_bytes + recv_bytes, 0,
                        ok=False, error_code=err_code, sent_bytes=sent_bytes, recv_bytes=recv_bytes,
                    )

                raw_payload = recv_exact(sock, payload_len, timeout_s=max(0.0, deadline_abs - time.perf_counter()))
                recv_bytes += len(raw_payload)
                rtt_ms = (time.perf_counter() - t_start) * 1000.0

                feat_arr = np.frombuffer(raw_payload, dtype=np.float32).copy()
                if not np.all(np.isfinite(feat_arr)):
                    if not self.persistent_connections:
                        sock.close()
                    return ModalityFetchResult(
                        None, rtt_ms, sent_bytes + recv_bytes, 0,
                        ok=False, error_code="NON_FINITE_CLIENT_PAYLOAD",
                        sent_bytes=sent_bytes, recv_bytes=recv_bytes,
                    )

                expected_dim = 74 if modality == "audio" else 35
                if len(feat_arr) != expected_dim:
                    if not self.persistent_connections:
                        sock.close()
                    return ModalityFetchResult(
                        None, rtt_ms, sent_bytes + recv_bytes, 0,
                        ok=False, error_code="DIMENSION_MISMATCH",
                        sent_bytes=sent_bytes, recv_bytes=recv_bytes,
                    )

                if not self.persistent_connections:
                    sock.close()

                feat_tensor = torch.from_numpy(feat_arr).unsqueeze(0).to(self.device)
                return ModalityFetchResult(
                    feat_tensor, rtt_ms, sent_bytes + recv_bytes, payload_len,
                    ok=True, error_code=None, sent_bytes=sent_bytes, recv_bytes=recv_bytes,
                )

            except ConnectionRefusedError:
                self._drop_persistent_conn(modality)
                rtt_ms = (time.perf_counter() - t_start) * 1000.0
                return ModalityFetchResult(
                    None, rtt_ms, sent_bytes + recv_bytes, 0,
                    ok=False, error_code="CONNECTION_REFUSED", sent_bytes=sent_bytes, recv_bytes=recv_bytes,
                )
            except (socket.timeout, TimeoutError):
                if sock is not None:
                    try:
                        sock.close()
                    except Exception:
                        pass
                self._drop_persistent_conn(modality)
                rtt_ms = (time.perf_counter() - t_start) * 1000.0
                return ModalityFetchResult(
                    None, rtt_ms, sent_bytes + recv_bytes, 0,
                    ok=False, error_code="SOCKET_TIMEOUT", sent_bytes=sent_bytes, recv_bytes=recv_bytes,
                )
            except (EOFError, ConnectionResetError, BrokenPipeError, ProtocolFrameError, json.JSONDecodeError, ValueError) as e:
                if sock is not None:
                    try:
                        sock.close()
                    except Exception:
                        pass
                self._drop_persistent_conn(modality)
                rtt_ms = (time.perf_counter() - t_start) * 1000.0
                return ModalityFetchResult(
                    None, rtt_ms, sent_bytes + recv_bytes, 0,
                    ok=False, error_code=f"PROTOCOL_OR_RESET:{type(e).__name__}",
                    sent_bytes=sent_bytes, recv_bytes=recv_bytes,
                )
            except OSError as e:
                if sock is not None:
                    try:
                        sock.close()
                    except Exception:
                        pass
                self._drop_persistent_conn(modality)
                rtt_ms = (time.perf_counter() - t_start) * 1000.0
                return ModalityFetchResult(
                    None, rtt_ms, sent_bytes + recv_bytes, 0,
                    ok=False, error_code=f"OS_ERROR:{e.errno}",
                    sent_bytes=sent_bytes, recv_bytes=recv_bytes,
                )

    def _fetch_parallel(self, modalities: List[str], sample_idx: int, timeout_s: float) -> Dict[str, ModalityFetchResult]:
        """Executes concurrent TCP socket fetches using bounded ThreadPoolExecutor."""
        if len(modalities) == 1:
            m = modalities[0]
            return {m: self.try_fetch_remote_modality(m, sample_idx, timeout_s=timeout_s)}
        futs = {m: self._pool_executor.submit(self.try_fetch_remote_modality, m, sample_idx, timeout_s) for m in modalities}
        results = {}
        for m, fut in futs.items():
            results[m] = fut.result()
        return results

    def _predict(
        self,
        subset_key: str,
        features: Dict[str, torch.Tensor],
        modality_mask: Optional[Dict[str, bool]] = None,
    ) -> Tuple[np.ndarray, int]:
        model = self.predictor_bank[subset_key]
        mods = SUBSET_METADATA[subset_key]["mods"]
        if modality_mask is not None:
            for m in mods:
                if not modality_mask.get(m, True):
                    raise MaskedFeatureAccessViolation(f"Attempted masked read of '{m}' in distributed predict.")
        tensors = [features[m] for m in mods]
        concat_feat = torch.cat(tensors, dim=-1)
        with torch.no_grad():
            logits = model(concat_feat)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            pred = int(np.argmax(probs))
        return probs, pred

    def _acquire_mod(
        self,
        mod: str,
        sample_idx: int,
        features: Dict[str, torch.Tensor],
        t_deadline_monotonic: float,
        tx_log: List[Dict[str, Any]],
    ) -> Tuple[bool, int, int, float, Optional[str]]:
        if mod == "text":
            return True, 0, 0, 0.0, None
        rem_s = (t_deadline_monotonic - time.perf_counter()) - self.inference_reserve_s
        res = self.try_fetch_remote_modality(mod, sample_idx, timeout_s=rem_s)
        tx_log.append({
            "modality": mod,
            "ok": res.ok,
            "rtt_ms": res[1],
            "app_bytes": res[2],
            "payload_bytes": res.payload_bytes,
            "sent_bytes": res.sent_bytes,
            "recv_bytes": res.recv_bytes,
            "error_code": res.error_code,
        })
        if res.ok and res[0] is not None:
            features[mod] = res[0]
        return res.ok, res[2], res.payload_bytes, res[1], res.error_code

    def process_query(
        self,
        sample_idx: int,
        text_feature: torch.Tensor,
        policy_mode: str = "sequential_confidence",
        tau_stage1: float = 0.55,
        tau_stage2: float = 0.45,
        deadline_ms: float = 50.0,
        cost_budget: float = 3.5,
        modality_mask: Optional[Dict[str, bool]] = None,
        rng: Optional[np.random.Generator] = None,
        p_acquire: float = 0.28,
    ) -> Dict[str, Any]:
        """
        Unified execution path for distributed TCP socket routing policies.
        Propagates exact per-query modality_mask, cost_budget, and monotonic deadline_ms.
        """
        t_wall_start = time.perf_counter()
        t_deadline_monotonic = t_wall_start + (deadline_ms / 1000.0)

        if modality_mask is None:
            modality_mask = {"text": True, "audio": True, "visual": True}

        available_mods = [m for m in ["text", "audio", "visual"] if modality_mask.get(m, True)]
        total_app_bytes = 0
        total_payload_bytes = 0
        total_network_rtt_ms = 0.0
        total_inference_ms = 0.0
        nominal_cost = 0.0
        failed_attempts = 0
        network_transactions: List[Dict[str, Any]] = []
        acquisition_path: List[str] = []

        features: Dict[str, torch.Tensor] = {}
        if modality_mask.get("text", True):
            features["text"] = text_feature.to(self.device)

        def _finalize(
            pred: int,
            probs: np.ndarray,
            final_subset: str,
            status: str,
            reason: str,
        ) -> Dict[str, Any]:
            t_wall_end = time.perf_counter()
            wall_ms = (t_wall_end - t_wall_start) * 1000.0
            on_time = bool(wall_ms <= deadline_ms) if status == "SUCCESS" else False
            b_viol = bool(nominal_cost > cost_budget)
            sq_pred = int(pred) if (status == "SUCCESS" and on_time and pred in (0, 1, 2)) else -1
            return {
                "sample_idx": sample_idx,
                "policy": policy_mode,
                "prediction": int(pred),
                "service_quality_prediction": sq_pred,
                "probabilities": probs,
                "confidence": float(np.max(probs)) if np.sum(probs) > 0 else 0.0,
                "final_subset": final_subset,
                "acquisition_path": acquisition_path,
                "routing_reason": reason,
                "status": status,
                "nominal_cost": float(nominal_cost),
                "wire_bytes_transmitted": int(total_app_bytes),
                "application_protocol_bytes": int(total_app_bytes),
                "payload_bytes": int(total_payload_bytes),
                "network_rtt_ms": float(total_network_rtt_ms),
                "inference_ms": float(total_inference_ms),
                "wall_ms": float(wall_ms),
                "deadline_ms": float(deadline_ms),
                "cost_budget": float(cost_budget),
                "on_time": on_time,
                "budget_violated": b_viol,
                "accepted": bool(pred in (0, 1, 2)),
                "useful_on_time": bool(sq_pred in (0, 1, 2)),
                "failed_attempts": int(failed_attempts),
                "network_transactions": network_transactions,
                "num_network_hops": len(network_transactions),
            }

        # Blackout check
        if not available_mods:
            return _finalize(-1, np.zeros(3, dtype=np.float32), "NONE", "REJECTED_BLACKOUT", "blackout_no_modalities")

        # Initial feasible subset check for constrained policies
        base_mod = "text" if "text" in available_mods else available_mods[0]
        base_key = mods_to_subset_key([base_mod])
        base_nom_cost = float(SUBSET_METADATA[base_key]["cost"])
        base_nom_lat = float(SUBSET_METADATA[base_key]["latency_ms"])

        # Normalize legacy alias always_on -> always_on_serial
        mode = "always_on_serial" if policy_mode == "always_on" else policy_mode

        # =====================================================================
        # 1. Always-On Direct Serial & Always-On Direct Parallel (Unrestricted)
        # =====================================================================
        if mode in ("always_on_serial", "always_on_parallel"):
            remote_mods = [m for m in available_mods if m != "text"]
            acquired_mods = ["text"] if "text" in available_mods else []
            nominal_cost = float(SUBSET_METADATA[mods_to_subset_key(available_mods)]["cost"])

            if mode == "always_on_serial":
                for rm in remote_mods:
                    ok, app_b, pay_b, rtt_ms, err = self._acquire_mod(
                        rm, sample_idx, features, t_deadline_monotonic, network_transactions
                    )
                    total_app_bytes += app_b
                    total_payload_bytes += pay_b
                    total_network_rtt_ms += rtt_ms
                    if ok:
                        acquired_mods.append(rm)
                    else:
                        failed_attempts += 1
            else:
                if remote_mods:
                    rem_s = (t_deadline_monotonic - time.perf_counter()) - self.inference_reserve_s
                    res_map = self._fetch_parallel(remote_mods, sample_idx, timeout_s=rem_s)
                    for rm, res in res_map.items():
                        network_transactions.append({
                            "modality": rm, "ok": res.ok, "rtt_ms": res[1],
                            "app_bytes": res[2], "payload_bytes": res.payload_bytes,
                            "sent_bytes": res.sent_bytes, "recv_bytes": res.recv_bytes, "error_code": res.error_code,
                        })
                        total_app_bytes += res[2]
                        total_payload_bytes += res.payload_bytes
                        total_network_rtt_ms = max(total_network_rtt_ms, res[1])
                        if res.ok and res[0] is not None:
                            features[rm] = res[0]
                            acquired_mods.append(rm)
                        else:
                            failed_attempts += 1

            if not acquired_mods or (t_deadline_monotonic - time.perf_counter()) <= 0.0:
                return _finalize(
                    -1, np.zeros(3, dtype=np.float32), "NONE", "FAILED_OR_TIMEOUT",
                    f"{mode}_failed_or_deadline_expired"
                )
            final_subset = mods_to_subset_key(acquired_mods)
            acquisition_path.append(final_subset)
            t_inf = time.perf_counter()
            probs, pred = self._predict(final_subset, features, modality_mask)
            total_inference_ms += (time.perf_counter() - t_inf) * 1000.0
            return _finalize(pred, probs, final_subset, "SUCCESS", mode)

        # =====================================================================
        # 2. Direct Static TA (with feasible fallback)
        # =====================================================================
        elif mode == "static_ta":
            target_mods = ["text", "audio"]
            if set(target_mods).issubset(available_mods) and cost_budget >= 2.2 and (t_deadline_monotonic - time.perf_counter()) > self.inference_reserve_s:
                nominal_cost += MODALITY_COSTS["text"] + MODALITY_COSTS["audio"]
                ok, app_b, pay_b, rtt_ms, err = self._acquire_mod(
                    "audio", sample_idx, features, t_deadline_monotonic, network_transactions
                )
                total_app_bytes += app_b
                total_payload_bytes += pay_b
                total_network_rtt_ms += rtt_ms
                if ok and (t_deadline_monotonic - time.perf_counter()) > 0.0:
                    final_subset = "TA"
                    acquisition_path.append("TA")
                    t_inf = time.perf_counter()
                    probs, pred = self._predict("TA", features, modality_mask)
                    total_inference_ms += (time.perf_counter() - t_inf) * 1000.0
                    return _finalize(pred, probs, "TA", "SUCCESS", "static_ta_direct")
                failed_attempts += 1
                # Fallback to local Text if deadline still valid
                if (t_deadline_monotonic - time.perf_counter()) > 0.0 and cost_budget >= 1.0:
                    acquisition_path.append("T")
                    t_inf = time.perf_counter()
                    probs, pred = self._predict("T", features, modality_mask)
                    total_inference_ms += (time.perf_counter() - t_inf) * 1000.0
                    return _finalize(pred, probs, "T", "SUCCESS", f"static_ta_provider_fault_fallback_T_{err}")
                return _finalize(-1, np.zeros(3, dtype=np.float32), "NONE", "FAILED_OR_TIMEOUT", f"static_ta_failed_{err}")

            # Budget or mask constrained fallback
            for cand in ["T", "A", "V"]:
                c_mods = SUBSET_METADATA[cand]["mods"]
                c_cost = float(SUBSET_METADATA[cand]["cost"])
                c_lat = float(SUBSET_METADATA[cand]["latency_ms"])
                if set(c_mods).issubset(available_mods) and c_cost <= cost_budget and deadline_ms >= c_lat:
                    nominal_cost = c_cost
                    if c_mods[0] != "text":
                        ok, app_b, pay_b, rtt_ms, err = self._acquire_mod(
                            c_mods[0], sample_idx, features, t_deadline_monotonic, network_transactions
                        )
                        total_app_bytes += app_b
                        total_payload_bytes += pay_b
                        total_network_rtt_ms += rtt_ms
                        if not ok:
                            failed_attempts += 1
                            return _finalize(-1, np.zeros(3, dtype=np.float32), "NONE", "FAILED_OR_TIMEOUT", f"static_ta_fallback_err_{err}")
                    acquisition_path.append(cand)
                    t_inf = time.perf_counter()
                    probs, pred = self._predict(cand, features, modality_mask)
                    total_inference_ms += (time.perf_counter() - t_inf) * 1000.0
                    return _finalize(pred, probs, cand, "SUCCESS", f"static_TA_fallback_to_{cand}")

            return _finalize(-1, np.zeros(3, dtype=np.float32), "NONE", "REJECTED_INFEASIBLE", "static_ta_infeasible")

        # =====================================================================
        # 3. Budget-Admitted Max-Subset (Parallel Fetch)
        # =====================================================================
        elif mode == "budget_admitted_max":
            chosen = None
            for sub in ["TAV", "TA", "TV", "AV", "T", "A", "V"]:
                s_mods = SUBSET_METADATA[sub]["mods"]
                if set(s_mods).issubset(available_mods) and SUBSET_METADATA[sub]["cost"] <= cost_budget and SUBSET_METADATA[sub]["latency_ms"] <= deadline_ms:
                    chosen = sub
                    break
            if chosen is None:
                return _finalize(-1, np.zeros(3, dtype=np.float32), "NONE", "REJECTED_INFEASIBLE", "budget_admission_failed")

            c_mods = SUBSET_METADATA[chosen]["mods"]
            nominal_cost = float(SUBSET_METADATA[chosen]["cost"])
            remote_mods = [m for m in c_mods if m != "text"]
            acq = ["text"] if "text" in c_mods else []
            if remote_mods:
                rem_s = (t_deadline_monotonic - time.perf_counter()) - self.inference_reserve_s
                res_map = self._fetch_parallel(remote_mods, sample_idx, timeout_s=rem_s)
                for rm, res in res_map.items():
                    network_transactions.append({
                        "modality": rm, "ok": res.ok, "rtt_ms": res[1], "app_bytes": res[2],
                        "payload_bytes": res.payload_bytes, "sent_bytes": res.sent_bytes,
                        "recv_bytes": res.recv_bytes, "error_code": res.error_code,
                    })
                    total_app_bytes += res[2]
                    total_payload_bytes += res.payload_bytes
                    total_network_rtt_ms = max(total_network_rtt_ms, res[1])
                    if res.ok and res[0] is not None:
                        features[rm] = res[0]
                        acq.append(rm)
                    else:
                        failed_attempts += 1

            if not acq or (t_deadline_monotonic - time.perf_counter()) <= 0.0:
                return _finalize(-1, np.zeros(3, dtype=np.float32), "NONE", "FAILED_OR_TIMEOUT", "budget_admitted_failed_or_timeout")
            final_sub = mods_to_subset_key(acq)
            acquisition_path.append(final_sub)
            t_inf = time.perf_counter()
            probs, pred = self._predict(final_sub, features, modality_mask)
            total_inference_ms += (time.perf_counter() - t_inf) * 1000.0
            return _finalize(pred, probs, final_sub, "SUCCESS", f"admitted_max_subset_{chosen}")

        # =====================================================================
        # 4. Unimodal Text (or cheapest feasible fallback)
        # =====================================================================
        elif mode == "unimodal_text":
            if cost_budget < base_nom_cost or deadline_ms < base_nom_lat:
                return _finalize(-1, np.zeros(3, dtype=np.float32), "NONE", "REJECTED_INFEASIBLE", "infeasible_budget_or_deadline")
            nominal_cost = base_nom_cost
            if base_mod != "text":
                ok, app_b, pay_b, rtt_ms, err = self._acquire_mod(
                    base_mod, sample_idx, features, t_deadline_monotonic, network_transactions
                )
                total_app_bytes += app_b
                total_payload_bytes += pay_b
                total_network_rtt_ms += rtt_ms
                if not ok:
                    failed_attempts += 1
                    return _finalize(-1, np.zeros(3, dtype=np.float32), "NONE", "FAILED_OR_TIMEOUT", f"unimodal_err_{err}")
            acquisition_path.append(base_key)
            t_inf = time.perf_counter()
            probs, pred = self._predict(base_key, features, modality_mask)
            total_inference_ms += (time.perf_counter() - t_inf) * 1000.0
            return _finalize(pred, probs, base_key, "SUCCESS", "unimodal_text_local")

        # =====================================================================
        # 5. One-Shot Confidence Routing (tau=0.55) with Parallel Escalation
        # =====================================================================
        elif mode == "confidence_routing_055":
            if cost_budget < base_nom_cost or deadline_ms < base_nom_lat:
                return _finalize(-1, np.zeros(3, dtype=np.float32), "NONE", "REJECTED_INFEASIBLE", "infeasible_budget_or_deadline")
            nominal_cost = base_nom_cost
            if base_mod != "text":
                ok, app_b, pay_b, rtt_ms, err = self._acquire_mod(
                    base_mod, sample_idx, features, t_deadline_monotonic, network_transactions
                )
                total_app_bytes += app_b
                total_payload_bytes += pay_b
                total_network_rtt_ms += rtt_ms
                if not ok:
                    failed_attempts += 1
                    return _finalize(-1, np.zeros(3, dtype=np.float32), "NONE", "FAILED_OR_TIMEOUT", f"cr_init_err_{err}")

            acquisition_path.append(base_key)
            t_inf = time.perf_counter()
            probs, pred = self._predict(base_key, features, modality_mask)
            total_inference_ms += (time.perf_counter() - t_inf) * 1000.0
            p_max = float(np.max(probs))

            if p_max >= tau_stage1:
                return _finalize(pred, probs, base_key, "SUCCESS", f"confident_triage_stayed_{base_key}_{p_max:.3f}")

            target_key = mods_to_subset_key(available_mods)
            target_cost = float(SUBSET_METADATA[target_key]["cost"])
            target_lat = float(SUBSET_METADATA[target_key]["latency_ms"])
            if target_key != base_key and target_cost <= cost_budget and target_lat <= deadline_ms:
                missing_mods = [m for m in available_mods if m != base_mod]
                remote_missing = [m for m in missing_mods if m != "text"]
                rem_s = (t_deadline_monotonic - time.perf_counter()) - self.inference_reserve_s
                if rem_s <= 0.0:
                    return _finalize(pred, probs, base_key, "SUCCESS", "cr_deadline_exhausted_kept_base")

                # Charge nominal escalation cost for attempted acquisition
                nominal_cost = target_cost
                acq = [base_mod]
                if "text" in missing_mods:
                    acq.append("text")
                if remote_missing:
                    res_map = self._fetch_parallel(remote_missing, sample_idx, timeout_s=rem_s)
                    for rm, res in res_map.items():
                        network_transactions.append({
                            "modality": rm, "ok": res.ok, "rtt_ms": res[1], "app_bytes": res[2],
                            "payload_bytes": res.payload_bytes, "sent_bytes": res.sent_bytes,
                            "recv_bytes": res.recv_bytes, "error_code": res.error_code,
                        })
                        total_app_bytes += res[2]
                        total_payload_bytes += res.payload_bytes
                        total_network_rtt_ms = max(total_network_rtt_ms, res[1])
                        if res.ok and res[0] is not None:
                            features[rm] = res[0]
                            acq.append(rm)
                        else:
                            failed_attempts += 1

                final_sub = mods_to_subset_key(acq)
                if final_sub != base_key and (t_deadline_monotonic - time.perf_counter()) > 0.0:
                    acquisition_path.append(final_sub)
                    t_inf2 = time.perf_counter()
                    probs, pred = self._predict(final_sub, features, modality_mask)
                    total_inference_ms += (time.perf_counter() - t_inf2) * 1000.0
                    return _finalize(pred, probs, final_sub, "SUCCESS", f"escalated_parallel_to_{final_sub}")
                elif (t_deadline_monotonic - time.perf_counter()) > 0.0:
                    return _finalize(pred, probs, base_key, "SUCCESS", f"cr_parallel_fault_kept_{base_key}")
                return _finalize(-1, np.zeros(3, dtype=np.float32), base_key, "FAILED_OR_TIMEOUT", "cr_escalation_timeout")

            return _finalize(pred, probs, base_key, "SUCCESS", f"escalation_budget_constrained_stayed_{base_key}")

        # =====================================================================
        # 6. Candidate: Sequential Confidence Router (tau1=0.55, tau2=0.45)
        # =====================================================================
        elif mode == "sequential_confidence":
            if cost_budget < base_nom_cost or deadline_ms < base_nom_lat:
                return _finalize(-1, np.zeros(3, dtype=np.float32), "NONE", "REJECTED_INFEASIBLE", "infeasible_budget_or_deadline")

            current_mods = [base_mod]
            current_key = base_key
            nominal_cost = base_nom_cost
            nom_lat = base_nom_lat

            if base_mod != "text":
                ok, app_b, pay_b, rtt_ms, err = self._acquire_mod(
                    base_mod, sample_idx, features, t_deadline_monotonic, network_transactions
                )
                total_app_bytes += app_b
                total_payload_bytes += pay_b
                total_network_rtt_ms += rtt_ms
                if not ok:
                    failed_attempts += 1
                    return _finalize(-1, np.zeros(3, dtype=np.float32), "NONE", "FAILED_OR_TIMEOUT", f"seq_init_err_{err}")

            acquisition_path.append(current_key)
            t_inf1 = time.perf_counter()
            probs, pred = self._predict(current_key, features, modality_mask)
            total_inference_ms += (time.perf_counter() - t_inf1) * 1000.0
            p_max = float(np.max(probs))

            if p_max >= tau_stage1:
                return _finalize(pred, probs, current_key, "SUCCESS", f"stage1_confident_{current_key}_{p_max:.3f}")

            # Stage 2: secondary modality (audio-first)
            candidates = [m for m in available_mods if m not in current_mods]
            ordered_cand = [m for m in ["audio", "visual", "text"] if m in candidates]
            acquired_stage2 = False
            for mod in ordered_cand:
                c_add = MODALITY_COSTS[mod]
                l_add = MODALITY_LATENCIES[mod]
                if (nominal_cost + c_add <= cost_budget) and (nom_lat + l_add <= deadline_ms):
                    rem_s = (t_deadline_monotonic - time.perf_counter()) - self.inference_reserve_s
                    if rem_s <= 0.0:
                        return _finalize(pred, probs, current_key, "SUCCESS", f"stage1_deadline_reserve_kept_{current_key}")
                    nominal_cost += c_add
                    nom_lat += l_add
                    ok, app_b, pay_b, rtt_ms, err = self._acquire_mod(
                        mod, sample_idx, features, t_deadline_monotonic, network_transactions
                    )
                    total_app_bytes += app_b
                    total_payload_bytes += pay_b
                    total_network_rtt_ms += rtt_ms
                    if not ok:
                        failed_attempts += 1
                        if (t_deadline_monotonic - time.perf_counter()) > 0.0:
                            return _finalize(pred, probs, current_key, "SUCCESS", f"stage2_fault_{err}_kept_{current_key}")
                        return _finalize(-1, np.zeros(3, dtype=np.float32), current_key, "FAILED_OR_TIMEOUT", f"stage2_timeout_{err}")

                    current_mods.append(mod)
                    current_key = mods_to_subset_key(current_mods)
                    acquisition_path.append(current_key)
                    t_inf2 = time.perf_counter()
                    probs, pred = self._predict(current_key, features, modality_mask)
                    total_inference_ms += (time.perf_counter() - t_inf2) * 1000.0
                    p_max = float(np.max(probs))
                    acquired_stage2 = True
                    break

            if not acquired_stage2 or p_max >= tau_stage2:
                return _finalize(pred, probs, current_key, "SUCCESS", f"stage2_stop_{current_key}_{p_max:.3f}")

            # Stage 3: tertiary modality
            rem_cand = [m for m in available_mods if m not in current_mods]
            for mod in rem_cand:
                c_add = MODALITY_COSTS[mod]
                l_add = MODALITY_LATENCIES[mod]
                if (nominal_cost + c_add <= cost_budget) and (nom_lat + l_add <= deadline_ms):
                    rem_s2 = (t_deadline_monotonic - time.perf_counter()) - self.inference_reserve_s
                    if rem_s2 <= 0.0:
                        return _finalize(pred, probs, current_key, "SUCCESS", f"stage2_deadline_reserve_kept_{current_key}")
                    nominal_cost += c_add
                    nom_lat += l_add
                    ok, app_b, pay_b, rtt_ms, err = self._acquire_mod(
                        mod, sample_idx, features, t_deadline_monotonic, network_transactions
                    )
                    total_app_bytes += app_b
                    total_payload_bytes += pay_b
                    total_network_rtt_ms += rtt_ms
                    if not ok:
                        failed_attempts += 1
                        if (t_deadline_monotonic - time.perf_counter()) > 0.0:
                            return _finalize(pred, probs, current_key, "SUCCESS", f"stage3_fault_{err}_kept_{current_key}")
                        return _finalize(-1, np.zeros(3, dtype=np.float32), current_key, "FAILED_OR_TIMEOUT", f"stage3_timeout_{err}")

                    current_mods.append(mod)
                    current_key = mods_to_subset_key(current_mods)
                    acquisition_path.append(current_key)
                    t_inf3 = time.perf_counter()
                    probs, pred = self._predict(current_key, features, modality_mask)
                    total_inference_ms += (time.perf_counter() - t_inf3) * 1000.0
                    break

            return _finalize(pred, probs, current_key, "SUCCESS", f"stage3_stop_{current_key}_{float(np.max(probs)):.3f}")

        # =====================================================================
        # 7. Matched-Budget Random Acquisition
        # =====================================================================
        elif mode == "matched_budget_random":
            if cost_budget < base_nom_cost or deadline_ms < base_nom_lat:
                return _finalize(-1, np.zeros(3, dtype=np.float32), "NONE", "REJECTED_INFEASIBLE", "infeasible_budget_or_deadline")

            local_rng = rng if rng is not None else np.random.default_rng(42 + sample_idx)
            chosen_mods = [base_mod]
            nominal_cost = base_nom_cost
            nom_lat = base_nom_lat
            for m in available_mods:
                if m not in chosen_mods and local_rng.uniform() < p_acquire:
                    c_add = MODALITY_COSTS[m]
                    l_add = MODALITY_LATENCIES[m]
                    if (nominal_cost + c_add <= cost_budget) and (nom_lat + l_add <= deadline_ms):
                        chosen_mods.append(m)
                        nominal_cost += c_add
                        nom_lat += l_add

            acq = ["text"] if "text" in chosen_mods else []
            rem_mods = [m for m in chosen_mods if m != "text"]
            if rem_mods:
                rem_s = (t_deadline_monotonic - time.perf_counter()) - self.inference_reserve_s
                res_map = self._fetch_parallel(rem_mods, sample_idx, timeout_s=rem_s)
                for rm, res in res_map.items():
                    network_transactions.append({
                        "modality": rm, "ok": res.ok, "rtt_ms": res[1], "app_bytes": res[2],
                        "payload_bytes": res.payload_bytes, "sent_bytes": res.sent_bytes,
                        "recv_bytes": res.recv_bytes, "error_code": res.error_code,
                    })
                    total_app_bytes += res[2]
                    total_payload_bytes += res.payload_bytes
                    total_network_rtt_ms = max(total_network_rtt_ms, res[1])
                    if res.ok and res[0] is not None:
                        features[rm] = res[0]
                        acq.append(rm)
                    else:
                        failed_attempts += 1

            if not acq or (t_deadline_monotonic - time.perf_counter()) <= 0.0:
                return _finalize(-1, np.zeros(3, dtype=np.float32), "NONE", "FAILED_OR_TIMEOUT", "random_failed_or_timeout")
            final_sub = mods_to_subset_key(acq)
            acquisition_path.append(final_sub)
            t_inf = time.perf_counter()
            probs, pred = self._predict(final_sub, features, modality_mask)
            total_inference_ms += (time.perf_counter() - t_inf) * 1000.0
            return _finalize(pred, probs, final_sub, "SUCCESS", f"random_acquisition_{final_sub}")

        else:
            raise ValueError(f"Unknown distributed policy mode: '{policy_mode}'")
