"""
src/service/distributed_providers.py: Distributed Modality Socket Providers for Tri-SACTA V13.2.

Features:
- Standalone TCP socket servers for isolated modality providers (Audio: 8765, Visual: 8766).
- Strict sample ID verification: Rejects invalid sample IDs, wrong modalities, or out-of-bounds indices.
  (NO modulo indexing; NO fallback to index 0).
- Robust framed protocol with exact-length reads (recv_exact loop for headers and binary payloads).
- Finite-float validation on serialized feature arrays.
- Telemetry accounting distinguishing payload bytes from application-protocol framing bytes.
- Clean process-safe shutdown and health-check ping support.
"""

from __future__ import annotations

import os
import sys
import socket
import struct
import json
import threading
import logging
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

logger = logging.getLogger(__name__)


def recv_exact(sock: socket.socket, n_bytes: int, timeout_s: Optional[float] = None) -> bytes:
    """
    Reads exactly n_bytes from socket, handling TCP fragmentation and enforcing socket timeouts.
    Raises ConnectionResetError if the socket closes before n_bytes are received.
    """
    if timeout_s is not None:
        sock.settimeout(max(0.001, timeout_s))
    chunks = []
    received = 0
    while received < n_bytes:
        chunk = sock.recv(min(n_bytes - received, 65536))
        if not chunk:
            raise ConnectionResetError(f"Connection closed prematurely after {received}/{n_bytes} bytes.")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


class ModalitySocketServer:
    """
    Standalone TCP socket server for an isolated modality provider (Stage D/G).
    Supports either in-process daemon thread execution or separate OS process execution.
    """

    def __init__(
        self,
        modality: str,
        host: str = "127.0.0.1",
        port: int = 8765,
        split_name: str = "val",
    ):
        self.modality = modality.lower()
        self.host = host
        self.port = port
        self.split_name = split_name
        self.running = False
        self.server_sock: Optional[socket.socket] = None
        self.thread: Optional[threading.Thread] = None

        # Load feature cache into provider memory
        fpath = os.path.join(REPO_ROOT, "data", "cmu_mosei", "features_cache", split_name, f"{self.modality}.pt")
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Provider cannot locate feature cache: {fpath}")
        self.features = torch.load(fpath, map_location="cpu", weights_only=False).float()
        self.total_samples = len(self.features)

        # Telemetry
        self.requests_served = 0
        self.requests_rejected = 0
        self.total_app_bytes_sent = 0
        self.total_payload_bytes_sent = 0

    def start(self):
        """Binds socket and starts listening."""
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
                self.server_sock.settimeout(0.5)
                conn, addr = self.server_sock.accept()
            except socket.timeout:
                continue
            except Exception:
                break

            with conn:
                try:
                    conn.settimeout(2.0)
                    # 1. Read 4-byte message header length
                    raw_len = recv_exact(conn, 4, timeout_s=2.0)
                    msg_len = struct.unpack("!I", raw_len)[0]

                    if msg_len > 65536:
                        # Reject oversized framing
                        self._send_error(conn, "MESSAGE_TOO_LARGE", f"Header length {msg_len} exceeds 64KB.")
                        self.requests_rejected += 1
                        continue

                    # 2. Read request JSON bytes
                    req_bytes = recv_exact(conn, msg_len, timeout_s=2.0)
                    req = json.loads(req_bytes.decode("utf-8"))

                    # Handle health-check ping
                    if req.get("action") == "ping":
                        resp_data = json.dumps({
                            "status": "OK",
                            "modality": self.modality,
                            "samples": self.total_samples,
                        }).encode("utf-8")
                        header = struct.pack("!II", 200, len(resp_data))
                        conn.sendall(header + resp_data)
                        continue

                    # 3. Validate modality
                    req_mod = req.get("modality", "").lower()
                    if req_mod != self.modality:
                        self._send_error(conn, "WRONG_MODALITY", f"Provider serves '{self.modality}', requested '{req_mod}'.")
                        self.requests_rejected += 1
                        continue

                    # 4. Strict sample ID / index validation (NO MODULO INDEXING!)
                    if "sample_idx" not in req:
                        self._send_error(conn, "MISSING_SAMPLE_ID", "Request must include 'sample_idx'.")
                        self.requests_rejected += 1
                        continue

                    try:
                        s_idx = int(req["sample_idx"])
                    except (ValueError, TypeError):
                        self._send_error(conn, "INVALID_SAMPLE_FORMAT", f"Cannot parse sample_idx: {req.get('sample_idx')}")
                        self.requests_rejected += 1
                        continue

                    if s_idx < 0 or s_idx >= self.total_samples:
                        self._send_error(
                            conn, "OUT_OF_BOUNDS_SAMPLE_ID",
                            f"sample_idx {s_idx} is out of bounds [0, {self.total_samples - 1}]."
                        )
                        self.requests_rejected += 1
                        continue

                    # 5. Retrieve feature tensor and validate finite floats
                    feat_np = self.features[s_idx].numpy().astype("float32")
                    if not np.all(np.isfinite(feat_np)):
                        self._send_error(conn, "NON_FINITE_FEATURES", f"Sample {s_idx} contains NaN or Inf features.")
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

                    # Framing: [4 bytes status code: 200] + [4 bytes meta JSON len] + [meta bytes] + [4 bytes payload len] + [payload bytes]
                    frame_hdr = struct.pack("!III", 200, meta_len, payload_len)
                    full_resp = frame_hdr + meta_bytes + raw_payload
                    conn.sendall(full_resp)

                    self.requests_served += 1
                    self.total_payload_bytes_sent += payload_len
                    self.total_app_bytes_sent += len(full_resp)

                except Exception as e:
                    logger.debug(f"[{self.modality.upper()} Provider] Socket error: {e}")
                    self.requests_rejected += 1

    def _send_error(self, conn: socket.socket, code: str, msg: str):
        """Sends framed error response over socket."""
        try:
            err_dict = {"status": "ERROR", "error_code": code, "message": msg}
            err_bytes = json.dumps(err_dict).encode("utf-8")
            hdr = struct.pack("!III", 400, len(err_bytes), 0)
            conn.sendall(hdr + err_bytes)
            self.total_app_bytes_sent += len(hdr) + len(err_bytes)
        except Exception:
            pass

    def stop(self):
        """Cleanly stops the socket server."""
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
            f"({self.total_app_bytes_sent:,} application bytes sent)."
        )
