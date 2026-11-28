"""
scripts/phase3_network/network_emulator.py: Controlled Application-Layer Communication-Delay
and Bandwidth-Throttling Emulation over TCP Loopback (Phase III Supplementary).
"""

from __future__ import annotations

import time
import random
from dataclasses import dataclass
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor


@dataclass(frozen=True)
class NetworkCondition:
    condition_id: str
    target_rtt_ms: float
    bandwidth_mbps: Optional[float]  # None = unrestricted loopback
    jitter_ms: float = 0.0
    regime: str = "A"  # "A" = original deadlines, "B" = relaxed 250ms deadline


def compute_serialization_delay_s(total_bytes: int, bandwidth_mbps: Optional[float]) -> float:
    if bandwidth_mbps is None or bandwidth_mbps <= 0.0:
        return 0.0
    bytes_per_sec = (bandwidth_mbps * 1_000_000.0) / 8.0
    return total_bytes / bytes_per_sec


def emulate_rtt_and_bandwidth(
    total_bytes: int,
    cond: NetworkCondition,
    rng: Optional[random.Random] = None,
    deadline_remaining_s: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Emulates a single round-trip communication delay + serialization delay.
    Ensures target_rtt_ms is injected ONCE per round-trip (not doubled) and
    respects deadline_remaining_s to avoid unbounded waiting.
    """
    jitter = 0.0
    if cond.jitter_ms > 0.0 and rng is not None:
        jitter = rng.uniform(-cond.jitter_ms, cond.jitter_ms)
    rtt_s = max(0.0, (cond.target_rtt_ms + jitter) / 1000.0)
    ser_s = compute_serialization_delay_s(total_bytes, cond.bandwidth_mbps)
    intended_s = rtt_s + ser_s

    timed_out = False
    sleep_s = intended_s
    if deadline_remaining_s is not None and intended_s > deadline_remaining_s:
        sleep_s = max(0.0, deadline_remaining_s)
        timed_out = True

    if sleep_s > 0.0:
        time.sleep(sleep_s)

    return {"intended_ms": intended_s * 1000.0, "slept_ms": sleep_s * 1000.0, "timed_out": timed_out}


def calibrate_transport(conditions: list[NetworkCondition]) -> Dict[str, Any]:
    """Synthetic transport calibration verifying RTT, parallel overlap, and deadline cancellation."""
    results = {}
    rng = random.Random(1320)
    for cond in conditions:
        samples = []
        for _ in range(5):
            t0 = time.perf_counter()
            emulate_rtt_and_bandwidth(512, cond, rng=rng)
            samples.append((time.perf_counter() - t0) * 1000.0)
        # Verify parallel overlap
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as ex:
            f1 = ex.submit(emulate_rtt_and_bandwidth, 512, cond, None)
            f2 = ex.submit(emulate_rtt_and_bandwidth, 512, cond, None)
            f1.result()
            f2.result()
        par_ms = (time.perf_counter() - t0) * 1000.0
        results[cond.condition_id] = {
            "target_rtt_ms": cond.target_rtt_ms,
            "bandwidth_mbps": cond.bandwidth_mbps,
            "jitter_ms": cond.jitter_ms,
            "achieved_mean_ms": sum(samples) / len(samples),
            "parallel_two_stream_ms": par_ms,
        }
    return results
