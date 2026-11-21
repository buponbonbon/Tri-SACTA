"""
04_run_experiment_matrix.py: Multi-Seed Experiment Matrix Runner for Tri-SACTA.

Executes paired benchmark matrix across fixed seeds [42, 100, 2026, 777, 999]
for methods:
- Source-Only + Static CP
- TENT + Online ACI
- SAR + Online ACI
- READ / CTTA + Online ACI
- Tri-SACTA + Online ACI (Ours)
- Delayed feedback variants (delay=10, 50)

Outputs immutable results to results/runs/ with detailed provenance.
"""

from __future__ import annotations

import os
import sys
import copy
import argparse
import logging
import json
import yaml
import subprocess
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SEEDS = [42, 100, 2026, 777, 999]


def run_single_experiment(
    config_path: str,
    method: str,
    conformal: str,
    seed: int,
    data_mode: str,
    delay: int = 0,
    output_dir: str = "results/runs",
) -> str:
    """Execute a single stream benchmark evaluation and return result JSON path."""
    os.makedirs(output_dir, exist_ok=True)
    out_filename = f"{method}_{conformal}_{data_mode}_delay{delay}_seed{seed}.json"
    target_path = os.path.join(output_dir, out_filename)

    if os.path.exists(target_path):
        logger.info(f"Skipping already completed experiment: {out_filename}")
        return target_path

    cmd = [
        sys.executable,
        "scripts/02_run_stream_benchmark.py",
        "--config", config_path,
        "--method", method,
        "--conformal", conformal,
        "--seed", str(seed),
        "--delay", str(delay),
        "--data-mode", data_mode,
        "--output_dir", output_dir,
    ]

    logger.info(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Experiment failed! STDERR:\n{result.stderr}")
        raise RuntimeError(f"Experiment failed for {method} seed {seed}")

    logger.info(f"Completed experiment: {out_filename}")
    return target_path


def main():
    parser = argparse.ArgumentParser(description="Run Multi-Seed Experiment Matrix")
    parser.add_argument("--config", type=str, default="configs/debug_synthetic.yaml")
    parser.add_argument("--data-mode", type=str, default="synthetic", choices=["synthetic", "real"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 100, 2026])  # Default 3 seeds for laptop efficiency, full 5 configurable
    parser.add_argument("--methods", type=str, nargs="+", default=["source_only", "tent", "sar", "read", "tri_sacta"])
    parser.add_argument("--output_dir", type=str, default="results/runs")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    matrix_jobs = []
    for seed in args.seeds:
        for method in args.methods:
            conformal = "static_cp" if method == "source_only" else "online_aci"
            matrix_jobs.append((method, conformal, seed, 0))

        # Delayed feedback ablations on Tri-SACTA
        if "tri_sacta" in args.methods:
            matrix_jobs.append(("tri_sacta", "online_aci", seed, 10))
            matrix_jobs.append(("tri_sacta", "online_aci", seed, 50))

    logger.info(f"Generated Matrix Plan: {len(matrix_jobs)} total runs across seeds {args.seeds}.")

    if args.dry_run:
        logger.info("Dry-run requested. Exiting without execution.")
        for j in matrix_jobs:
            logger.info(f"  Plan: Method={j[0]}, Conformal={j[1]}, Seed={j[2]}, Delay={j[3]}")
        return

    # Execute all jobs
    completed = []
    for method, conformal, seed, delay in matrix_jobs:
        res_file = run_single_experiment(
            config_path=args.config,
            method=method,
            conformal=conformal,
            seed=seed,
            data_mode=args.data_mode,
            delay=delay,
            output_dir=args.output_dir,
        )
        completed.append(res_file)

    logger.info(f"All {len(completed)} matrix experiments completed successfully.")


if __name__ == "__main__":
    main()
