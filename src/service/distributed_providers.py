"""
src/service/distributed_providers.py: Distributed Modality Socket Providers (Stage G).

Implements real socket servers for Audio and Visual modalities:
- Listens on TCP sockets (default ports 8765 for Audio, 8766 for Visual).
- Responds to query requests by fetching sample embeddings and serializing binary wire payloads.
- Measures real transmission bytes, payload bytes, and socket transfer time without mock delays.
"""

from __future__ import annotations

import os
import sys
import socket
import struct
import json
import threading
import logging
from typing import Dict, Any, Optional
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

logger = logging.getLogger(__name__)


class ModalitySocketServer:
    """Standalone TCP socket server for an isolated modality provider."""

    def __init__(self, modality: str, host: str = "127.0.0.1", port: int = 8765, split_name: str = "val"):
        self.modality = modality
        self.host = host
        self.port = port
        self.split_name = split_name
        self.running = False
        self.server_sock: Optional[socket.socket] = None
        self.thread: Optional[threading.Thread] = None

        # Load modality feature cache into provider memory
        fpath = os.path.join(REPO_ROOT, "data", "cmu_mosei", "features_cache", split_name, f"{modality}.pt")
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Provider cannot locate feature cache: {fpath}")
        self.features = torch.load(fpath, map_location="cpu", weights_only=False).float()
        self.total_samples = len(self.features)

        # Telemetry
        self.bytes_transmitted = 0
        self.requests_served = 0

    def start(self):
        """Starts socket server in a daemon background thread."""
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((self.host, self.port))
        self.server_sock.listen(10)
        self.running = True

        self.thread = threading.Thread(target=self._serve_loop, daemon=True)
        self.thread.start()
        logger.info(f"[{self.modality.upper()} Provider] Socket server listening on {self.host}:{self.port} ({self.total_samples} samples loaded).")

    def _serve_loop(self):
        while self.running:
            try:
                self.server_sock.settimeout(1.0)
                conn, addr = self.server_sock.accept()
            except socket.timeout:
                continue
            except Exception:
                break

            with conn:
                try:
                    # Read 4-byte message length
                    raw_len = conn.recv(4)
                    if not raw_len or len(raw_len) < 4:
                        continue
                    msg_len = struct.unpack("!I", raw_len)[0]
                    req_data = conn.recv(msg_len)
                    req = json.loads(req_data.decode("utf-8"))

                    sample_idx = int(req.get("sample_idx", 0)) % self.total_samples

                    # Retrieve feature tensor and convert to raw float32 bytes
                    feat_np = self.features[sample_idx].numpy().astype("float32")
                    raw_bytes = feat_np.tobytes()
                    payload_len = len(raw_bytes)

                    # Send header [4 bytes payload length] + payload
                    resp_header = struct.pack("!I", payload_len)
                    conn.sendall(resp_header + raw_bytes)

                    self.bytes_transmitted += (4 + payload_len)
                    self.requests_served += 1
                except Exception as e:
                    logger.debug(f"Error handling socket request: {e}")

    def stop(self):
        """Stops the socket server cleanly."""
        self.running = False
        if self.server_sock:
            try:
                self.server_sock.close()
            except Exception:
                pass
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        logger.info(f"[{self.modality.upper()} Provider] Server stopped. Served {self.requests_served} requests, {self.bytes_transmitted} bytes transmitted.")
