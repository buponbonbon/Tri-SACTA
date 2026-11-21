"""
04_run_experiment_matrix.py: Multi-Seed Experiment Matrix Runner for Tri-SACTA — V3 (R3-10).

Executes paired benchmark matrix across fixed seeds [42, 100, 2026, 777, 999] for:
- Source-Only + Static CP
- TENT + Online ACI
- SAR + Online ACI
- READ / CTTA + Online ACI
- Tri-SACTA + Online ACI (Ours)
- Delayed feedback variants (delay=10, 50)

Guarantees (R3-10):
1. Pre-trains and validates source checkpoint for every seed before benchmarking.
2. Collision-free canonical namespaced run outputs matching 02_run_stream_benchmark.py.
3. Explicit protocol flag (--protocol prequential | transductive).
"""

from __future__ import annotations

import os
import sys
import argparse
import logging
import json
import yaml
import subprocess
from typing import Dict, List, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.naming import get_checkpoint_path, get_run_output_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SEEDS = [42, 100, 2026, 777, 999]


def ensure_source_checkpoint(config_path: str, data_mode: str, seed: int, output_dir: str = "checkpoints") -> str:
    """Ensure source checkpoint and anchor statistics exist for this seed (R3-10)."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    dataset = config.get("dataset", {}).get("name", "mosei")

    canonical_ckpt = get_checkpoint_path(output_dir, dataset, data_mode, seed)
    legacy_ckpt = os.path.join(output_dir, f"source_model_{data_mode}_seed{seed}.pt")

    if os.path.exists(canonical_ckpt) or os.path.exists(legacy_ckpt):
        ckpt_to_use = canonical_ckpt if os.path.exists(canonical_ckpt) else legacy_ckpt
        logger.info(f"Verified existing source checkpoint for seed {seed}: {ckpt_to_use}")
        return ckpt_to_use

    logger.info(f"No existing source checkpoint found for seed {seed}. Scheduling source pre-training...")
    cmd = [
        sys.executable,
        "scripts/01_train_source.py",
        "--config", config_path,
        "--seed", str(seed),
        "--data-mode", data_mode,
        "--output_dir", output_dir,
    ]
    logger.info(f"Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error(f"Source pre-training failed for seed {seed}! STDERR:\n{res.stderr}")
        raise RuntimeError(f"Source pre-training failed for seed {seed}")

    logger.info(f"Source pre-training completed successfully for seed {seed}.")
    return canonical_ckpt if os.path.exists(canonical_ckpt) else legacy_ckpt


def run_single_experiment(
    config_path: str,
    method: str,
    conformal: str,
    seed: int,
    data_mode: str,
    protocol: str = "prequential",
    delay: int = 0,
    output_dir: str = "results/runs",
) -> str:
    """Execute a single stream benchmark evaluation and return result JSON path."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    dataset = config.get("dataset", {}).get("name", "mosei")

    canonical_target = get_run_output_path(output_dir, dataset, data_mode, protocol, method, conformal, delay, seed)
    delay_tag = f"_delay{delay}" if delay > 0 else ""
    flat_target = os.path.join(output_dir, f"{method}_{conformal}_{dataset}_{data_mode}{delay_tag}_seed{seed}.json")

    if os.path.exists(canonical_target) or os.path.exists(flat_target):
        existing = canonical_target if os.path.exists(canonical_target) else flat_target
        logger.info(f"Skipping already completed experiment: {os.path.basename(existing)}")
        return existing

    cmd = [
        sys.executable,
        "scripts/02_run_stream_benchmark.py",
        "--config", config_path,
        "--method", method,
        "--conformal", conformal,
        "--protocol", protocol,
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

    logger.info(f"Completed experiment: {os.path.basename(canonical_target)}")
    return canonical_target


def main():
    parser = argparse.ArgumentParser(description="Run Multi-Seed Experiment Matrix (V3)")
    parser.add_argument("--config", type=str, default="configs/debug_synthetic.yaml")
    parser.add_argument("--data-mode", type=str, default="synthetic", choices=["synthetic", "real"])
    parser.add_argument("--protocol", type=str, default="prequential", choices=["prequential", "transductive"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 100, 2026])
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

    logger.info(
        f"Generated Matrix Plan: {len(matrix_jobs)} total runs across seeds {args.seeds} "
        f"| protocol: {args.protocol} | mode: {args.data_mode}."
    )

    if args.dry_run:
        logger.info("[DRY-RUN] Planned execution sequence:")
        for job in matrix_jobs:
            m, c, s, d = job
            logger.info(f"  - Method: {m:<12} Conformal: {c:<12} Seed: {s} Delay: {d}")
        return

    # Phase 1: Pre-train source checkpoints for all seeds (R3-10)
    for seed in args.seeds:
        ensure_source_checkpoint(args.config, args.data_mode, seed)

    # Phase 2: Execute benchmark stream for each job
    completed = []
    for job in matrix_jobs:
        m, c, s, d = job
        res_path = run_single_experiment(
            config_path=args.config,
            method=m,
            conformal=c,
            seed=s,
            data_mode=args.data_mode,
            protocol=args.protocol,
            delay=d,
            output_dir=args.output_dir,
        )
        completed.append(res_path)

    logger.info(f"Successfully finished all {len(completed)} matrix experiments.")


if __name__ == "__main__":
    main()
