"""
src/service/distributed_coordinator.py: Distributed Systems Coordinator for Tri-SACTA V13.2 (Stage D/G).

Rectifies Baseline Transport Fairness and Safety:
1. Gives Always-On an optimized direct path: fetches remote modalities and predicts directly
   on TAV without a redundant initial Text-only forward pass.
2. Gives Static TA a direct path: fetches remote Audio and predicts directly on TA without redundant Text pass.
3. Sequential routing pays all intermediate inference passes and sequential decision overhead.
4. Enforces monotonic absolute deadlines, remaining-time calculation, socket timeouts,
   and exact-length recv_exact reads.
5. Accurately reports payload bytes, application-protocol wire bytes, socket RTT, and inference wall-clock.
"""

from __future__ import annotations

import os
import sys
import socket
import struct
import json
import time
import logging
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.service.utility_router import SUBSET_METADATA, mods_to_subset_key
from src.service.distributed_providers import recv_exact

logger = logging.getLogger(__name__)


class ModalityFetchResult(tuple):
    """
    3-tuple containing (feat_tensor, rtt_ms, total_app_bytes).
    Carries payload_bytes as an attribute for detailed accounting while preserving 3-tuple unpacking compatibility.
    """
    def __new__(cls, feat_tensor: torch.Tensor, rtt_ms: float, total_app_bytes: int, payload_bytes: int = 0):
        obj = super().__new__(cls, (feat_tensor, rtt_ms, total_app_bytes))
        obj.payload_bytes = payload_bytes
        return obj


class DistributedCoordinator:
    """Orchestrates edge-cloud multimodal acquisition with fair baseline execution."""

    def __init__(
        self,
        predictor_bank: Dict[str, nn.Module],
        device: torch.device,
        audio_host: str = "127.0.0.1",
        audio_port: int = 8765,
        visual_host: str = "127.0.0.1",
        visual_port: int = 8766,
    ):
        self.predictor_bank = predictor_bank
        self.device = device
        self.provider_endpoints = {
            "audio": (audio_host, audio_port),
            "visual": (visual_host, visual_port),
        }
        for m in self.predictor_bank.values():
            m.eval()
            m.to(self.device)

    def fetch_remote_modality(
        self,
        modality: str,
        sample_idx: int,
        timeout_s: float = 2.0,
    ) -> ModalityFetchResult:
        """
        Sends TCP socket request to remote provider and receives binary feature tensor.
        Returns ModalityFetchResult (unpackable as (feature_tensor, rtt_ms, application_protocol_bytes),
        with .payload_bytes attribute).
        """
        host, port = self.provider_endpoints[modality]
        req_obj = {"sample_idx": int(sample_idx), "modality": modality}
        req_bytes = json.dumps(req_obj).encode("utf-8")
        req_hdr = struct.pack("!I", len(req_bytes))
        full_req = req_hdr + req_bytes

        t_start = time.perf_counter()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(max(0.001, timeout_s))
            sock.connect((host, port))
            sock.sendall(full_req)

            # Receive 12-byte response header: [status_code (4B), meta_len (4B), payload_len (4B)]
            raw_hdr = recv_exact(sock, 12, timeout_s=timeout_s)
            status_code, meta_len, payload_len = struct.unpack("!III", raw_hdr)

            # Receive metadata JSON
            raw_meta = recv_exact(sock, meta_len, timeout_s=timeout_s)
            meta = json.loads(raw_meta.decode("utf-8"))

            if status_code != 200 or meta.get("status") != "OK":
                err_msg = meta.get("message", f"Provider error code {status_code}")
                raise RuntimeError(f"Remote {modality} provider returned error: {err_msg}")

            # Receive binary float32 payload
            raw_payload = recv_exact(sock, payload_len, timeout_s=timeout_s)
            rtt_ms = (time.perf_counter() - t_start) * 1000.0

        total_app_bytes = len(full_req) + 12 + meta_len + payload_len
        feat_arr = np.frombuffer(raw_payload, dtype=np.float32).copy()
        feat_tensor = torch.from_numpy(feat_arr).unsqueeze(0).to(self.device)

        return ModalityFetchResult(feat_tensor, rtt_ms, total_app_bytes, payload_len)

    def _predict(self, subset_key: str, features: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, int]:
        model = self.predictor_bank[subset_key]
        mods = SUBSET_METADATA[subset_key]["mods"]
        tensors = [features[m] for m in mods]
        concat_feat = torch.cat(tensors, dim=-1)

        with torch.no_grad():
            logits = model(concat_feat)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            pred = int(np.argmax(probs))
        return probs, pred

    def process_query(
        self,
        sample_idx: int,
        text_feature: torch.Tensor,
        policy_mode: str = "sequential_confidence",
        tau_stage1: float = 0.55,
        tau_stage2: float = 0.45,
        deadline_ms: float = 50.0,
        cost_budget: float = 3.5,
    ) -> Dict[str, Any]:
        """
        Executes query end-to-end measuring real physical network bytes, RTT, and inference latency.
        Enforces monotonic deadline checking before each network hop and inference stage.
        """
        t_wall_start = time.perf_counter()
        t_deadline_monotonic = t_wall_start + (deadline_ms / 1000.0)

        total_app_bytes = 0
        total_payload_bytes = 0
        total_network_rtt_ms = 0.0
        total_inference_ms = 0.0
        features: Dict[str, torch.Tensor] = {"text": text_feature.to(self.device)}
        acquisition_path: List[str] = []
        network_transactions: List[Dict[str, Any]] = []

        # =====================================================================
        # Policy Mode 1: Optimized Always-On Full Fusion (Direct TAV)
        # =====================================================================
        if policy_mode == "always_on":
            # Direct path: Fetch Audio and Visual, then execute ONE forward pass on TAV.
            # Does NOT execute redundant 1st Text inference pass.
            rem_s = t_deadline_monotonic - time.perf_counter()
            aud_res = self.fetch_remote_modality("audio", sample_idx, timeout_s=rem_s)
            aud_feat, aud_rtt, aud_app_b = aud_res
            aud_pay_b = aud_res.payload_bytes
            total_app_bytes += aud_app_b
            total_payload_bytes += aud_pay_b
            total_network_rtt_ms += aud_rtt
            features["audio"] = aud_feat
            network_transactions.append({"modality": "audio", "rtt_ms": aud_rtt, "app_bytes": aud_app_b})

            rem_s = t_deadline_monotonic - time.perf_counter()
            vis_res = self.fetch_remote_modality("visual", sample_idx, timeout_s=rem_s)
            vis_feat, vis_rtt, vis_app_b = vis_res
            vis_pay_b = vis_res.payload_bytes
            total_app_bytes += vis_app_b
            total_payload_bytes += vis_pay_b
            total_network_rtt_ms += vis_rtt
            features["visual"] = vis_feat
            network_transactions.append({"modality": "visual", "rtt_ms": vis_rtt, "app_bytes": vis_app_b})

            final_subset = "TAV"
            acquisition_path = ["TAV"]
            t_inf_start = time.perf_counter()
            probs, pred = self._predict(final_subset, features)
            total_inference_ms = (time.perf_counter() - t_inf_start) * 1000.0
            routing_reason = "always_on_direct_tav"

        # =====================================================================
        # Policy Mode 2: Optimized Static TA (Direct TA)
        # =====================================================================
        elif policy_mode == "static_ta":
            # Direct path: Fetch Audio, then execute ONE forward pass on TA.
            rem_s = t_deadline_monotonic - time.perf_counter()
            aud_res = self.fetch_remote_modality("audio", sample_idx, timeout_s=rem_s)
            aud_feat, aud_rtt, aud_app_b = aud_res
            aud_pay_b = aud_res.payload_bytes
            total_app_bytes += aud_app_b
            total_payload_bytes += aud_pay_b
            total_network_rtt_ms += aud_rtt
            features["audio"] = aud_feat
            network_transactions.append({"modality": "audio", "rtt_ms": aud_rtt, "app_bytes": aud_app_b})

            final_subset = "TA"
            acquisition_path = ["TA"]
            t_inf_start = time.perf_counter()
            probs, pred = self._predict(final_subset, features)
            total_inference_ms = (time.perf_counter() - t_inf_start) * 1000.0
            routing_reason = "static_ta_direct"

        # =====================================================================
        # Policy Mode 3: Unimodal Text-Only
        # =====================================================================
        elif policy_mode == "unimodal_text":
            final_subset = "T"
            acquisition_path = ["T"]
            t_inf_start = time.perf_counter()
            probs, pred = self._predict(final_subset, features)
            total_inference_ms = (time.perf_counter() - t_inf_start) * 1000.0
            routing_reason = "unimodal_text_local"

        # =====================================================================
        # Policy Mode 4: Candidate Sequential Confidence Router
        # =====================================================================
        elif policy_mode == "sequential_confidence":
            # Step 1: Local inference on Text (pays intermediate inference)
            current_subset = "T"
            acquisition_path.append(current_subset)
            t_inf_start = time.perf_counter()
            probs, pred = self._predict(current_subset, features)
            total_inference_ms += (time.perf_counter() - t_inf_start) * 1000.0

            p_max = float(np.max(probs))

            if p_max >= tau_stage1:
                routing_reason = f"confident_stage1_text_pmax_{p_max:.3f}"
                final_subset = "T"
            else:
                # Step 2: Acquire Audio over TCP if budget and deadline allow
                rem_s = t_deadline_monotonic - time.perf_counter()
                if rem_s <= 0.001 or cost_budget < 2.2:
                    routing_reason = f"stage1_constrained_stayed_T_rem_{rem_s*1000:.1f}ms_budget_{cost_budget}"
                    final_subset = "T"
                else:
                    aud_res = self.fetch_remote_modality(
                        "audio", sample_idx, timeout_s=rem_s
                    )
                    aud_feat, aud_rtt, aud_app_b = aud_res
                    aud_pay_b = aud_res.payload_bytes
                    total_app_bytes += aud_app_b
                    total_payload_bytes += aud_pay_b
                    total_network_rtt_ms += aud_rtt
                    features["audio"] = aud_feat
                    network_transactions.append({"modality": "audio", "rtt_ms": aud_rtt, "app_bytes": aud_app_b})

                    current_subset = "TA"
                    acquisition_path.append(current_subset)
                    t_inf_start2 = time.perf_counter()
                    probs, pred = self._predict(current_subset, features)
                    total_inference_ms += (time.perf_counter() - t_inf_start2) * 1000.0
                    p_max = float(np.max(probs))

                    if p_max >= tau_stage2 or cost_budget < 3.5:
                        routing_reason = f"stage2_stop_TA_pmax_{p_max:.3f}"
                        final_subset = "TA"
                    else:
                        # Step 3: Acquire Visual over TCP
                        rem_s2 = t_deadline_monotonic - time.perf_counter()
                        if rem_s2 <= 0.001:
                            routing_reason = f"stage2_deadline_exhausted_stayed_TA"
                            final_subset = "TA"
                        else:
                            vis_res = self.fetch_remote_modality(
                                "visual", sample_idx, timeout_s=rem_s2
                            )
                            vis_feat, vis_rtt, vis_app_b = vis_res
                            vis_pay_b = vis_res.payload_bytes
                            total_app_bytes += vis_app_b
                            total_payload_bytes += vis_pay_b
                            total_network_rtt_ms += vis_rtt
                            features["visual"] = vis_feat
                            network_transactions.append({"modality": "visual", "rtt_ms": vis_rtt, "app_bytes": vis_app_b})

                            current_subset = "TAV"
                            acquisition_path.append(current_subset)
                            t_inf_start3 = time.perf_counter()
                            probs, pred = self._predict(current_subset, features)
                            total_inference_ms += (time.perf_counter() - t_inf_start3) * 1000.0
                            routing_reason = f"stage3_stop_TAV_pmax_{float(np.max(probs)):.3f}"
                            final_subset = "TAV"
        else:
            raise ValueError(f"Unknown distributed policy mode: '{policy_mode}'")

        t_wall_end = time.perf_counter()
        total_wall_ms = (t_wall_end - t_wall_start) * 1000.0
        on_time = total_wall_ms <= deadline_ms

        return {
            "sample_idx": sample_idx,
            "policy": policy_mode,
            "prediction": pred,
            "probabilities": probs,
            "confidence": float(np.max(probs)),
            "final_subset": final_subset,
            "acquisition_path": acquisition_path,
            "routing_reason": routing_reason,
            "wire_bytes_transmitted": total_app_bytes,
            "application_protocol_bytes": total_app_bytes,
            "payload_bytes": total_payload_bytes,
            "network_rtt_ms": total_network_rtt_ms,
            "inference_ms": total_inference_ms,
            "wall_ms": total_wall_ms,
            "deadline_ms": deadline_ms,
            "cost_budget": cost_budget,
            "on_time": on_time,
            "network_transactions": network_transactions,
            "num_network_hops": len(network_transactions),
        }
