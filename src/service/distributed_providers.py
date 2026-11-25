"""
src/service/distributed_providers.py: Hardened Distributed Modality Socket Providers for Tri-SACTA V13.2.

Features:
- Standalone TCP socket servers for isolated modality providers (Audio: 8765, Visual: 8766).
- Supports BOTH persistent keep-alive TCP framing and per-request short-lived connections.
- Strict sample ID verification: Rejects invalid/missing sample IDs, wrong modalities,
  or out-of-bounds indices (NO modulo indexing, NO silent fallback).
- Bounded recv_exact reads with maximum frame limits (64KB header/meta, 16MB payload).
- Finite-float validation on serialized float32 feature arrays.
- Controlled fault injection support for runtime resilience testing (delay, disconnect/truncate,
  malformed/oversized header, non-finite tensor injection).
- Symmetric client/server application-protocol byte telemetry.
"""

from __future__ import annotations

import os
import sys
import time
import socket
import struct
import json
import threading
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

logger = logging.getLogger(__name__)

MAX_CONTROL_FRAME_BYTES = 65536          # 64 KB max JSON request / error / metadata
MAX_PAYLOAD_BYTES = 16 * 1024 * 1024     # 16 MB max float32 payload


class ProtocolFrameError(RuntimeError):
    """Raised when a socket frame is malformed, oversized, truncated, or violates protocol."""


def recv_exact(sock: socket.socket, n_bytes: int, timeout_s: Optional[float] = None) -> bytes:
    """
    Reads exactly n_bytes from socket, handling TCP fragmentation and enforcing remaining timeout.
    Raises EOFError if cleanly closed at 0 bytes, ConnectionResetError on mid-frame truncation,
    or TimeoutError if deadline expires.
    """
    if n_bytes < 0:
        raise ProtocolFrameError(f"Negative frame read size: {n_bytes}")
    if n_bytes == 0:
        return b""

    deadline = (time.perf_counter() + timeout_s) if timeout_s is not None else None
    chunks = []
    received = 0
    while received < n_bytes:
        if deadline is not None:
            rem = deadline - time.perf_counter()
            if rem <= 0.0:
                raise socket.timeout(f"Absolute deadline expired while reading {received}/{n_bytes} bytes.")
            sock.settimeout(max(0.0005, rem))
        chunk = sock.recv(min(n_bytes - received, 65536))
        if not chunk:
            if received == 0:
                raise EOFError("Peer cleanly closed connection.")
            raise ConnectionResetError(f"Truncated TCP frame: closed after {received}/{n_bytes} bytes.")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


class ModalitySocketServer:
    """
    Standalone TCP socket server for an isolated modality provider.
    Supports persistent keep-alive requests per TCP connection as well as single-shot requests.
    """

    def __init__(
        self,
        modality: str,
        host: str = "127.0.0.1",
        port: int = 8765,
        split_name: str = "val",
        features_tensor: Optional[torch.Tensor] = None,
    ):
        self.modality = modality.lower()
        self.host = host
        self.port = port
        self.split_name = split_name
        self.running = False
        self.server_sock: Optional[socket.socket] = None
        self.thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        if features_tensor is not None:
            self.features = features_tensor.float().cpu()
        else:
            if split_name in ("holdout", "eligible_holdout", "confirmatory"):
                raise PermissionError(
                    "PHASE I GUARDRAIL: Socket provider cannot auto-load holdout without Phase II authorization."
                )
            fpath = os.path.join(REPO_ROOT, "data", "cmu_mosei", "features_cache", split_name, f"{self.modality}.pt")
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Provider cannot locate feature cache: {fpath}")
            self.features = torch.load(fpath, map_location="cpu", weights_only=False).float()
        self.total_samples = len(self.features)

        # Controlled fault injection state (for runtime stress tests)
        self.fault_config: Dict[str, Any] = {
            "delay_ms": 0.0,
            "disconnect_mid_payload": False,
            "send_oversize_header": False,
            "inject_nan_sample": None,
        }

        # Telemetry
        self.requests_served = 0
        self.requests_rejected = 0
        self.total_app_bytes_sent = 0
        self.total_app_bytes_recv = 0
        self.total_payload_bytes_sent = 0

    def update_features(self, features_tensor: torch.Tensor) -> None:
        """Swaps the served workload tensor cleanly under lock."""
        with self._lock:
            self.features = features_tensor.float().cpu()
            self.total_samples = len(self.features)

    def set_fault(self, **kwargs: Any) -> None:
        with self._lock:
            self.fault_config.update(kwargs)

    def clear_faults(self) -> None:
        with self._lock:
            self.fault_config = {
                "delay_ms": 0.0,
                "disconnect_mid_payload": False,
                "send_oversize_header": False,
                "inject_nan_sample": None,
            }

    def reset_telemetry(self) -> None:
        with self._lock:
            self.requests_served = 0
            self.requests_rejected = 0
            self.total_app_bytes_sent = 0
            self.total_app_bytes_recv = 0
            self.total_payload_bytes_sent = 0

    def start(self):
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((self.host, self.port))
        self.server_sock.listen(128)
        self.running = True

        self.thread = threading.Thread(target=self._serve_loop, daemon=True)
        self.thread.start()
        logger.info(
            f"[{self.modality.upper()} Provider] Listening on {self.host}:{self.port} "
            f"({self.total_samples} samples loaded from '{self.split_name}')."
        )

    def _serve_loop(self):
        while self.running:
            try:
                self.server_sock.settimeout(0.2)
                conn, addr = self.server_sock.accept()
            except socket.timeout:
                continue
            except Exception:
                break

            t = threading.Thread(target=self._handle_connection, args=(conn,), daemon=True)
            t.start()

    def _handle_connection(self, conn: socket.socket):
        try:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass

        with conn:
            while self.running:
                try:
                    raw_len = recv_exact(conn, 4, timeout_s=5.0)
                except (EOFError, socket.timeout, ConnectionResetError, OSError):
                    break

                try:
                    msg_len = struct.unpack("!I", raw_len)[0]
                    if msg_len == 0 or msg_len > MAX_CONTROL_FRAME_BYTES:
                        with self._lock:
                            self.total_app_bytes_recv += 4
                            self.requests_rejected += 1
                        self._send_error(conn, "MESSAGE_TOO_LARGE", f"Header length {msg_len} exceeds bounds [1, {MAX_CONTROL_FRAME_BYTES}].")
                        break

                    req_bytes = recv_exact(conn, msg_len, timeout_s=2.0)
                    with self._lock:
                        self.total_app_bytes_recv += 4 + len(req_bytes)

                    req = json.loads(req_bytes.decode("utf-8"))

                    if req.get("action") == "ping":
                        resp_data = json.dumps({
                            "status": "OK",
                            "modality": self.modality,
                            "samples": self.total_samples,
                        }).encode("utf-8")
                        header = struct.pack("!II", 200, len(resp_data))
                        pkt = header + resp_data
                        conn.sendall(pkt)
                        with self._lock:
                            self.total_app_bytes_sent += len(pkt)
                        continue

                    with self._lock:
                        f_cfg = dict(self.fault_config)
                        cur_features = self.features
                        cur_total = self.total_samples

                    if f_cfg["delay_ms"] > 0.0:
                        time.sleep(f_cfg["delay_ms"] / 1000.0)

                    if f_cfg["send_oversize_header"]:
                        bad_hdr = struct.pack("!III", 200, MAX_CONTROL_FRAME_BYTES + 1024, 0)
                        conn.sendall(bad_hdr)
                        with self._lock:
                            self.total_app_bytes_sent += len(bad_hdr)
                            self.requests_rejected += 1
                        break

                    req_mod = str(req.get("modality", "")).lower()
                    if req_mod != self.modality:
                        self._send_error(conn, "WRONG_MODALITY", f"Provider serves '{self.modality}', requested '{req_mod}'.")
                        with self._lock:
                            self.requests_rejected += 1
                        continue

                    if "sample_idx" not in req:
                        self._send_error(conn, "MISSING_SAMPLE_ID", "Request must include 'sample_idx'.")
                        with self._lock:
                            self.requests_rejected += 1
                        continue

                    try:
                        s_idx = int(req["sample_idx"])
                    except (ValueError, TypeError):
                        self._send_error(conn, "INVALID_SAMPLE_FORMAT", f"Cannot parse sample_idx: {req.get('sample_idx')}")
                        with self._lock:
                            self.requests_rejected += 1
                        continue

                    if s_idx < 0 or s_idx >= cur_total:
                        self._send_error(
                            conn, "OUT_OF_BOUNDS_SAMPLE_ID",
                            f"sample_idx {s_idx} is out of bounds [0, {cur_total - 1}]."
                        )
                        with self._lock:
                            self.requests_rejected += 1
                        continue

                    feat_np = cur_features[s_idx].numpy().astype("float32").copy()
                    if f_cfg["inject_nan_sample"] == s_idx:
                        feat_np[0] = np.nan

                    if not np.all(np.isfinite(feat_np)):
                        self._send_error(conn, "NON_FINITE_FEATURES", f"Sample {s_idx} contains NaN or Inf features.")
                        with self._lock:
                            self.requests_rejected += 1
                        continue

                    raw_payload = feat_np.tobytes()
                    payload_len = len(raw_payload)
                    meta_dict = {
                        "status": "OK",
                        "sample_idx": s_idx,
                        "modality": self.modality,
                        "dim": int(len(feat_np)),
                        "payload_bytes": payload_len,
                    }
                    meta_bytes = json.dumps(meta_dict).encode("utf-8")
                    meta_len = len(meta_bytes)
                    frame_hdr = struct.pack("!III", 200, meta_len, payload_len)

                    if f_cfg["disconnect_mid_payload"]:
                        partial = frame_hdr + meta_bytes + raw_payload[: max(1, payload_len // 2)]
                        conn.sendall(partial)
                        with self._lock:
                            self.total_app_bytes_sent += len(partial)
                            self.requests_rejected += 1
                        break

                    full_resp = frame_hdr + meta_bytes + raw_payload
                    conn.sendall(full_resp)

                    with self._lock:
                        self.requests_served += 1
                        self.total_payload_bytes_sent += payload_len
                        self.total_app_bytes_sent += len(full_resp)

                except (EOFError, ConnectionResetError, BrokenPipeError, socket.timeout):
                    break
                except Exception as e:
                    logger.debug(f"[{self.modality.upper()} Provider] Socket error: {e}")
                    with self._lock:
                        self.requests_rejected += 1
                    break

    def _send_error(self, conn: socket.socket, code: str, msg: str):
        try:
            err_dict = {"status": "ERROR", "error_code": code, "message": msg}
            err_bytes = json.dumps(err_dict).encode("utf-8")
            hdr = struct.pack("!III", 400, len(err_bytes), 0)
            pkt = hdr + err_bytes
            conn.sendall(pkt)
            with self._lock:
                self.total_app_bytes_sent += len(pkt)
        except Exception:
            pass

    def stop(self):
        self.running = False
        if self.server_sock:
            try:
                self.server_sock.close()
            except Exception:
                pass
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        logger.info(
            f"[{self.modality.upper()} Provider] Stopped. Served {self.requests_served} queries "
            f"({self.total_app_bytes_sent:,} application bytes sent, {self.total_app_bytes_recv:,} bytes recv)."
        )
