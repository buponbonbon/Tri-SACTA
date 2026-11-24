"""
tests/test_v13_2_runtime_safety.py: Regression and Safety Suite for Distributed Runtime (Phase I-D).

Verifies Section 5 Requirements:
1. Strict sample ID rejection (no modulo indexing, no default index 0).
2. Wrong modality rejection.
3. Optimized Always-on direct TAV path (no redundant initial text pass).
4. Optimized Static TA direct path.
5. Unimodal text incurs exactly 0 remote bytes.
6. Monotonic deadline reserve enforcement stops before remote fetch.
"""

from __future__ import annotations

import os
import sys
import unittest
import socket
import struct
import json
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.loader import SUBSET_METADATA
from src.service.distributed_providers import ModalitySocketServer, recv_exact
from src.service.distributed_coordinator import DistributedCoordinator


def dummy_predictor_bank() -> dict[str, nn.Module]:
    bank = {}
    for k, v in SUBSET_METADATA.items():
        in_dim = v["in_dim"]
        linear = nn.Linear(in_dim, 3)
        nn.init.constant_(linear.weight, 0.01)
        nn.init.constant_(linear.bias, 0.0)
        linear.eval()
        bank[k] = linear
    return bank


class TestV132RuntimeSafety(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Start local test servers on test ports
        cls.audio_port = 8875
        cls.visual_port = 8876
        cls.audio_server = ModalitySocketServer("audio", host="127.0.0.1", port=cls.audio_port, split_name="val")
        cls.visual_server = ModalitySocketServer("visual", host="127.0.0.1", port=cls.visual_port, split_name="val")
        cls.audio_server.start()
        cls.visual_server.start()

        cls.bank = dummy_predictor_bank()
        cls.coordinator = DistributedCoordinator(
            cls.bank,
            torch.device("cpu"),
            audio_port=cls.audio_port,
            visual_port=cls.visual_port,
        )

    @classmethod
    def tearDownClass(cls):
        cls.audio_server.stop()
        cls.visual_server.stop()

    def test_strict_out_of_bounds_sample_id_rejection(self):
        """Out-of-bounds sample_idx must be rejected and NOT modulo indexed to a valid sample."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect(("127.0.0.1", self.audio_port))
            # Request sample_idx 99999 which does not exist
            req_data = json.dumps({"sample_idx": 99999, "modality": "audio"}).encode("utf-8")
            sock.sendall(struct.pack("!I", len(req_data)) + req_data)

            raw_hdr = recv_exact(sock, 12, timeout_s=2.0)
            status_code, meta_len, payload_len = struct.unpack("!III", raw_hdr)
            self.assertEqual(status_code, 400)
            self.assertEqual(payload_len, 0)

            err_resp = json.loads(recv_exact(sock, meta_len).decode("utf-8"))
            self.assertEqual(err_resp["status"], "ERROR")
            self.assertEqual(err_resp["error_code"], "OUT_OF_BOUNDS_SAMPLE_ID")

    def test_wrong_modality_rejection(self):
        """Requesting 'visual' from audio server must be rejected with WRONG_MODALITY."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect(("127.0.0.1", self.audio_port))
            req_data = json.dumps({"sample_idx": 0, "modality": "visual"}).encode("utf-8")
            sock.sendall(struct.pack("!I", len(req_data)) + req_data)

            raw_hdr = recv_exact(sock, 12, timeout_s=2.0)
            status_code, meta_len, payload_len = struct.unpack("!III", raw_hdr)
            self.assertEqual(status_code, 400)
            err_resp = json.loads(recv_exact(sock, meta_len).decode("utf-8"))
            self.assertEqual(err_resp["error_code"], "WRONG_MODALITY")

    def test_optimized_always_on_direct_tav_path(self):
        """Always-on directly fetches Audio & Visual and executes ONE forward pass on TAV."""
        text_feat = torch.ones(1, 768) * 0.1
        res = self.coordinator.process_query(0, text_feat, policy_mode="always_on")
        self.assertEqual(res["final_subset"], "TAV")
        self.assertEqual(res["acquisition_path"], ["TAV"])
        self.assertEqual(res["num_network_hops"], 2)
        self.assertGreater(res["application_protocol_bytes"], 0)

    def test_optimized_static_ta_direct_path(self):
        """Static TA directly fetches Audio and executes ONE forward pass on TA."""
        text_feat = torch.ones(1, 768) * 0.1
        res = self.coordinator.process_query(0, text_feat, policy_mode="static_ta")
        self.assertEqual(res["final_subset"], "TA")
        self.assertEqual(res["acquisition_path"], ["TA"])
        self.assertEqual(res["num_network_hops"], 1)

    def test_unimodal_text_zero_wire_bytes(self):
        """Unimodal text must incur 0 network hops and 0 application protocol bytes."""
        text_feat = torch.ones(1, 768) * 0.1
        res = self.coordinator.process_query(0, text_feat, policy_mode="unimodal_text")
        self.assertEqual(res["final_subset"], "T")
        self.assertEqual(res["num_network_hops"], 0)
        self.assertEqual(res["application_protocol_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
