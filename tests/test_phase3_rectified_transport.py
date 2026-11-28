"""
tests/test_phase3_rectified_transport.py: Stage D Synthetic Verification Suite (D1-D8)
for RectifiedNetworkCoordinator before running the Phase III development benchmark.
"""

import os
import sys
import time
import unittest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.loader import SUBSET_METADATA
from src.service.distributed_providers import ModalitySocketServer
from scripts.v13_2_run_phase2_holdout import _get_free_port
from scripts.phase3_network.network_emulator import NetworkCondition
from scripts.phase3_network.rectified_coordinator import RectifiedNetworkCoordinator


class TestPhase3RectifiedTransport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a_port = _get_free_port()
        cls.v_port = _get_free_port()
        cls.a_srv = ModalitySocketServer("audio", host="127.0.0.1", port=cls.a_port, features_tensor=torch.ones(4, 74))
        cls.v_srv = ModalitySocketServer("visual", host="127.0.0.1", port=cls.v_port, features_tensor=torch.ones(4, 35))
        cls.a_srv.start()
        cls.v_srv.start()
        time.sleep(0.15)
        cls.bank = {k: torch.nn.Linear(v["in_dim"], 3).eval() for k, v in SUBSET_METADATA.items()}
        for m in cls.bank.values():
            torch.nn.init.zeros_(m.weight)
            torch.nn.init.zeros_(m.bias)  # Uniform 1/3 probs -> confidence 0.333 < 0.45 -> triggers full cascade!

    @classmethod
    def tearDownClass(cls):
        cls.a_srv.stop()
        cls.v_srv.stop()

    def test_d1_zero_injection_equivalence(self):
        cond = NetworkCondition("ZERO", 0.0, None, 0.0, "B")
        coord = RectifiedNetworkCoordinator(self.bank, torch.device("cpu"), cond, audio_port=self.a_port, visual_port=self.v_port)
        try:
            res = coord.process_query_with_events(0, torch.ones(1, 768), policy_mode="sequential_confidence", deadline_ms=250.0)
            self.assertEqual(res["final_subset"], "TAV")
            self.assertEqual(res["remote_attempted"], 2)
            self.assertEqual(res["remote_succeeded"], 2)
        finally:
            coord.close()

    def test_d2_delay_affects_subsequent_routing_decision(self):
        # 25ms RTT under 35ms deadline: Audio succeeds (~25ms), Visual cannot complete in remaining ~10ms -> stops at TA!
        cond = NetworkCondition("RTT25", 25.0, None, 0.0, "A")
        coord = RectifiedNetworkCoordinator(self.bank, torch.device("cpu"), cond, audio_port=self.a_port, visual_port=self.v_port)
        try:
            res = coord.process_query_with_events(0, torch.ones(1, 768), policy_mode="sequential_confidence", deadline_ms=35.0)
            self.assertIn(res["final_subset"], ("TA", "T"))
            self.assertGreaterEqual(res["remote_attempted"], 1)
            self.assertGreater(res["events"][0]["elapsed_ms"], 15.0)
        finally:
            coord.close()

    def test_d4_parallel_overlap_under_delay(self):
        cond = NetworkCondition("RTT30", 30.0, None, 0.0, "B")
        coord = RectifiedNetworkCoordinator(self.bank, torch.device("cpu"), cond, audio_port=self.a_port, visual_port=self.v_port)
        try:
            res = coord.process_query_with_events(0, torch.ones(1, 768), policy_mode="always_on_parallel", deadline_ms=250.0)
            self.assertEqual(res["remote_attempted"], 2)
            ev0, ev1 = res["events"][0], res["events"][1]
            # Intervals must overlap: max(start0, start1) < min(end0, end1)
            self.assertLess(max(ev0["dispatch_ts"], ev1["dispatch_ts"]), min(ev0["complete_ts"], ev1["complete_ts"]))
        finally:
            coord.close()

    def test_event_based_critical_path_rounds_for_oneshot(self):
        cond = NetworkCondition("RTT20", 20.0, None, 0.0, "B")
        coord = RectifiedNetworkCoordinator(self.bank, torch.device("cpu"), cond, audio_port=self.a_port, visual_port=self.v_port)
        try:
            res = coord.process_query_with_events(0, torch.ones(1, 768), policy_mode="confidence_routing_055", deadline_ms=250.0, parent_request_id="paired_0")
            evs = sorted(res["events"], key=lambda e: e["dispatch_ts"])
            rounds, cur_end = 0, -1.0
            for ev in evs:
                if ev["dispatch_ts"] >= cur_end:
                    rounds += 1
                    cur_end = ev["complete_ts"]
                else:
                    cur_end = max(cur_end, ev["complete_ts"])
            self.assertEqual(res["remote_attempted"], 2)
            self.assertEqual(rounds, 1)
        finally:
            coord.close()

    def test_d5_deadline_exhausted_timeout(self):
        cond = NetworkCondition("RTT80", 80.0, None, 0.0, "A")
        coord = RectifiedNetworkCoordinator(self.bank, torch.device("cpu"), cond, audio_port=self.a_port, visual_port=self.v_port)
        try:
            res = coord.process_query_with_events(0, torch.ones(1, 768), policy_mode="sequential_confidence", deadline_ms=15.0)
            self.assertGreaterEqual(res["remote_timeouts"], 1)
            self.assertGreater(res["application_protocol_bytes"], 0)
        finally:
            coord.close()


if __name__ == "__main__":
    unittest.main()
