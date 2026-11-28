"""
tests/test_v13_1_accounting.py: Unit tests for distributed socket communication and wire byte accounting.
"""

import os
import sys
import time
import socket
import struct
import unittest
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.service.distributed_providers import ModalitySocketServer
from src.service.distributed_coordinator import DistributedCoordinator
from src.models.fusion_backbone import SimpleMLP


class TestV131DistributedAccounting(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cpu")
        cls.audio_port = 8795
        cls.visual_port = 8796

        cls.audio_server = ModalitySocketServer("audio", host="127.0.0.1", port=cls.audio_port, split_name="val")
        cls.visual_server = ModalitySocketServer("visual", host="127.0.0.1", port=cls.visual_port, split_name="val")
        cls.audio_server.start()
        cls.visual_server.start()
        time.sleep(0.3)

        # Mock predictor bank
        configs = {"T": 768, "A": 74, "V": 35, "TA": 842, "TV": 803, "AV": 109, "TAV": 877}
        cls.bank = {sub: SimpleMLP(in_dim=in_dim, hidden_dim=16, num_classes=3) for sub, in_dim in configs.items()}

        cls.coordinator = DistributedCoordinator(
            cls.bank, cls.device,
            audio_host="127.0.0.1", audio_port=cls.audio_port,
            visual_host="127.0.0.1", visual_port=cls.visual_port,
        )

    @classmethod
    def tearDownClass(cls):
        cls.audio_server.stop()
        cls.visual_server.stop()

    def test_remote_socket_fetch(self):
        """Tests physical TCP socket retrieval of modality features."""
        aud_tensor, rtt_ms, wire_bytes = self.coordinator.fetch_remote_modality("audio", 0)
        self.assertEqual(aud_tensor.shape, (1, 74))
        self.assertGreater(wire_bytes, 0)
        self.assertGreater(rtt_ms, 0.0)

        vis_tensor, vis_rtt, vis_bytes = self.coordinator.fetch_remote_modality("visual", 0)
        self.assertEqual(vis_tensor.shape, (1, 35))
        self.assertGreater(vis_bytes, 0)

    def test_unimodal_text_zero_wire_bytes(self):
        """Verifies that unimodal text incurs exactly 0 remote wire bytes."""
        text_feat = torch.randn(1, 768)
        res = self.coordinator.process_query(0, text_feat, policy_mode="unimodal_text")
        self.assertEqual(res["wire_bytes_transmitted"], 0)
        self.assertEqual(res["final_subset"], "T")

    def test_always_on_full_wire_bytes(self):
        """Verifies always_on mode fetches both audio and visual."""
        text_feat = torch.randn(1, 768)
        res = self.coordinator.process_query(0, text_feat, policy_mode="always_on")
        self.assertGreater(res["wire_bytes_transmitted"], 400)
        self.assertEqual(res["final_subset"], "TAV")


if __name__ == "__main__":
    unittest.main()
