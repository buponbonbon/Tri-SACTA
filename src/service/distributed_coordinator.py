"""
src/service/distributed_coordinator.py: Distributed Systems Coordinator (Stage G).

Coordinates edge-cloud multimodal acquisition:
- Retains Text locally.
- Selectively queries remote Audio and Visual socket providers over TCP only when the policy demands.
- Accurately measures wire bytes, socket RTT, serialization overhead, and end-to-end latency.
"""

from __future__ import annotations

import os
import sys
import socket
import struct
import json
import time
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.service.utility_router import SUBSET_METADATA, mods_to_subset_key

logger = logging.getLogger(__name__)


class DistributedCoordinator:
    """Orchestrates sequential acquisition over real TCP sockets."""

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

    def fetch_remote_modality(self, modality: str, sample_idx: int) -> Tuple[torch.Tensor, float, int]:
        """
        Sends TCP socket request to remote provider and receives binary feature tensor.
        Returns: (feature_tensor, rtt_ms, wire_bytes)
        """
        host, port = self.provider_endpoints[modality]
        req_obj = {"sample_idx": int(sample_idx), "modality": modality}
        req_bytes = json.dumps(req_obj).encode("utf-8")
        req_msg = struct.pack("!I", len(req_bytes)) + req_bytes

        t_start = time.perf_counter()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host, port))
            sock.sendall(req_msg)

            # Receive 4-byte payload length header
            resp_header = sock.recv(4)
            if not resp_header or len(resp_header) < 4:
                raise RuntimeError(f"Connection closed by {modality} provider without payload header.")
            payload_len = struct.unpack("!I", resp_header)[0]

            # Receive payload
            chunks = []
            bytes_received = 0
            while bytes_received < payload_len:
                chunk = sock.recv(min(4096, payload_len - bytes_received))
                if not chunk:
                    break
                chunks.append(chunk)
                bytes_received += len(chunk)

            raw_payload = b"".join(chunks)
        t_end = time.perf_counter()

        rtt_ms = (t_end - t_start) * 1000.0
        total_wire_bytes = len(req_msg) + 4 + len(raw_payload)

        # Deserialize float32 numpy array
        feat_arr = np.frombuffer(raw_payload, dtype=np.float32)
        feat_tensor = torch.from_numpy(feat_arr).unsqueeze(0).to(self.device)
        return feat_tensor, rtt_ms, total_wire_bytes

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
    ) -> Dict[str, Any]:
        """
        Processes query end-to-end measuring real network bytes, RTT, and model inference latency.
        """
        t_wall_start = time.perf_counter()
        total_wire_bytes = 0
        total_network_rtt_ms = 0.0
        features: Dict[str, torch.Tensor] = {"text": text_feature.to(self.device)}
        current_mods = ["text"]
        current_key = "T"
        acquisition_path = ["T"]

        # Step 1: Local inference on Text
        t_inf_start = time.perf_counter()
        probs, pred = self._predict(current_key, features)
        t_inf_ms = (time.perf_counter() - t_inf_start) * 1000.0

        p_max = float(np.max(probs))

        if policy_mode == "always_on":
            # Always fetch Audio and Visual over network
            aud_feat, aud_rtt, aud_bytes = self.fetch_remote_modality("audio", sample_idx)
            total_wire_bytes += aud_bytes
            total_network_rtt_ms += aud_rtt
            features["audio"] = aud_feat

            vis_feat, vis_rtt, vis_bytes = self.fetch_remote_modality("visual", sample_idx)
            total_wire_bytes += vis_bytes
            total_network_rtt_ms += vis_rtt
            features["visual"] = vis_feat

            current_key = "TAV"
            acquisition_path.append("TAV")
            t_inf_start2 = time.perf_counter()
            probs, pred = self._predict(current_key, features)
            t_inf_ms += (time.perf_counter() - t_inf_start2) * 1000.0

        elif policy_mode == "static_ta":
            # Always fetch Audio
            aud_feat, aud_rtt, aud_bytes = self.fetch_remote_modality("audio", sample_idx)
            total_wire_bytes += aud_bytes
            total_network_rtt_ms += aud_rtt
            features["audio"] = aud_feat

            current_key = "TA"
            acquisition_path.append("TA")
            t_inf_start2 = time.perf_counter()
            probs, pred = self._predict(current_key, features)
            t_inf_ms += (time.perf_counter() - t_inf_start2) * 1000.0

        elif policy_mode == "unimodal_text":
            # Never fetch over network
            pass

        elif policy_mode == "sequential_confidence":
            # Sequential Confidence Routing: acquire only if p_max < tau_stage1
            if p_max < tau_stage1:
                # Acquire Audio over socket
                aud_feat, aud_rtt, aud_bytes = self.fetch_remote_modality("audio", sample_idx)
                total_wire_bytes += aud_bytes
                total_network_rtt_ms += aud_rtt
                features["audio"] = aud_feat

                current_mods.append("audio")
                current_key = mods_to_subset_key(current_mods)
                acquisition_path.append(current_key)

                t_inf_start2 = time.perf_counter()
                probs, pred = self._predict(current_key, features)
                t_inf_ms += (time.perf_counter() - t_inf_start2) * 1000.0
                p_max2 = float(np.max(probs))

                # Check stage 2: acquire Visual if still uncertain
                if p_max2 < tau_stage2:
                    vis_feat, vis_rtt, vis_bytes = self.fetch_remote_modality("visual", sample_idx)
                    total_wire_bytes += vis_bytes
                    total_network_rtt_ms += vis_rtt
                    features["visual"] = vis_feat

                    current_mods.append("visual")
                    current_key = mods_to_subset_key(current_mods)
                    acquisition_path.append(current_key)

                    t_inf_start3 = time.perf_counter()
                    probs, pred = self._predict(current_key, features)
                    t_inf_ms += (time.perf_counter() - t_inf_start3) * 1000.0

        total_wall_ms = (time.perf_counter() - t_wall_start) * 1000.0
        on_time = int(total_wall_ms <= deadline_ms)

        return {
            "prediction": pred,
            "probabilities": probs,
            "final_subset": current_key,
            "acquisition_path": acquisition_path,
            "wire_bytes_transmitted": total_wire_bytes,
            "network_rtt_ms": total_network_rtt_ms,
            "inference_latency_ms": t_inf_ms,
            "total_wall_clock_ms": total_wall_ms,
            "on_time": on_time,
        }
