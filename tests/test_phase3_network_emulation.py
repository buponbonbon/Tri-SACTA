"""
tests/test_phase3_network_emulation.py: Unit tests for Phase III network emulation & analytical model.
"""
import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.phase3_network.network_emulator import NetworkCondition, compute_serialization_delay_s, emulate_rtt_and_bandwidth


class TestPhase3NetworkEmulation(unittest.TestCase):
    def test_serialization_delay_units(self):
        # 1 MB (1,000,000 bytes) at 8 Mbps = 1.0 second
        self.assertAlmostEqual(compute_serialization_delay_s(1_000_000, 8.0), 1.0, places=6)
        self.assertEqual(compute_serialization_delay_s(1000, None), 0.0)

    def test_deadline_cancellation_caps_sleep(self):
        cond = NetworkCondition("TEST_80", 80.0, None, 0.0, "A")
        res = emulate_rtt_and_bandwidth(512, cond, deadline_remaining_s=0.005)
        self.assertTrue(res["timed_out"])
        self.assertLessEqual(res["slept_ms"], 10.0)


if __name__ == "__main__":
    unittest.main()
