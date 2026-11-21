"""
Run and Checkpoint Naming Convention Helper for Tri-SACTA — V3 (R3-05, R3-10, R3-11).

Provides standardized, collision-free naming and parsing across all scripts:
- Checkpoints: checkpoints/{dataset}/{data_mode}/seed{seed}/source_model.pt
- Spectral Stats: checkpoints/{dataset}/{data_mode}/seed{seed}/source_spectral_stats.pt
- Benchmark Runs: results/runs/{dataset}/{data_mode}/{protocol}/{method}_{conformal}_delay{delay}_seed{seed}.json
"""

from __future__ import annotations

import os
from typing import Dict, Any, Optional


def get_checkpoint_dir(base_dir: str, dataset: str, data_mode: str, seed: int) -> str:
    """Return canonical namespaced directory for checkpoint and stats."""
    return os.path.join(base_dir, dataset, data_mode, f"seed{seed}")


def get_checkpoint_path(base_dir: str, dataset: str, data_mode: str, seed: int) -> str:
    """Return canonical path to source model checkpoint."""
    return os.path.join(get_checkpoint_dir(base_dir, dataset, data_mode, seed), "source_model.pt")


def get_spectral_stats_path(base_dir: str, dataset: str, data_mode: str, seed: int) -> str:
    """Return canonical path to spectral anchor statistics."""
    return os.path.join(get_checkpoint_dir(base_dir, dataset, data_mode, seed), "source_spectral_stats.pt")


def get_run_output_path(
    output_dir: str,
    dataset: str,
    data_mode: str,
    protocol: str,
    method: str,
    conformal: str,
    delay: int,
    seed: int,
) -> str:
    """Return canonical namespaced output JSON path for a benchmark run."""
    run_dir = os.path.join(output_dir, dataset, data_mode, protocol)
    os.makedirs(run_dir, exist_ok=True)
    filename = f"{method}_{conformal}_delay{delay}_seed{seed}.json"
    return os.path.join(run_dir, filename)


def parse_run_filename(fpath: str) -> Optional[Dict[str, Any]]:
    """Parse components from a standardized run filepath."""
    parts = os.path.normpath(fpath).split(os.sep)
    filename = parts[-1]
    if not filename.endswith(".json"):
        return None

    # Handle legacy flat structure or namespaced structure
    name = filename[:-5]
    tokens = name.split("_")
    # Expected format: {method}_{conformal}_delay{delay}_seed{seed}
    # or {method}_{conformal}_{dataset}_{data_mode}[_delayX]_seed{seed}
    return {
        "filename": filename,
        "path": fpath,
    }
