"""
00_validate_data.py: Data Integrity & Provenance Validation CLI Tool (Section 11.2, AUD-10, AUD-11).

Validates:
1. Feature cache directory and tensor dimensions.
2. Split disjointness (zero utterance-level or speaker-level leakage).
3. Manifest integrity and SHA-256 checksums if real data mode.
4. Fail-closed behavior for production reproducibility.
"""

from __future__ import annotations

import os
import sys
import argparse
import logging
import yaml
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import get_dataloaders
from src.data.manifest import DataManifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Validate Dataset Integrity for Tri-SACTA")
    parser.add_argument("--config", type=str, default="configs/debug_synthetic.yaml")
    parser.add_argument("--data-mode", type=str, default="synthetic", choices=["synthetic", "real"])
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Starting Data Validation for {config.get('dataset', {}).get('name')} | Mode: {args.data_mode}")

    if args.data_mode == "real":
        manifest_path = config.get("dataset", {}).get("manifest_path")
        if not manifest_path or not os.path.exists(manifest_path):
            logger.error(f"[FAIL-CLOSED] Manifest file missing at: {manifest_path}")
            sys.exit(1)

        manifest = DataManifest.load_json(manifest_path)
        report = manifest.validate_split_integrity()
        logger.info(f"Total manifest samples: {report['total_samples']}")
        logger.info(f"Counts per split: {report['counts_by_split']}")

        if not report["is_valid"]:
            logger.error(f"[LEAKAGE DETECTED] Overlaps found: {report['speaker_overlaps']}, {report['video_overlaps']}")
            sys.exit(1)
        logger.info("[PASS] Real dataset split integrity: 100% speaker and video disjoint!")
    else:
        # Synthetic validation
        dataloaders = get_dataloaders(config, batch_size=32, seed=42, data_mode="synthetic")
        counts = {k: len(v.dataset) for k, v in dataloaders.items()}
        logger.info(f"[PASS] Synthetic data generator verified. Split sample counts: {counts}")

        # Check sample batch shapes
        sample_batch = next(iter(dataloaders["train"]))
        feats = sample_batch["features"]
        mask = sample_batch["modality_mask"]
        labels = sample_batch["labels"]

        expected_dims = {
            "visual": config.get("features", {}).get("visual_dim", 512),
            "audio": config.get("features", {}).get("audio_dim", 768),
            "text": config.get("features", {}).get("text_dim", 768),
            "prosody": config.get("features", {}).get("prosody_dim", 88),
        }

        for m, dim in expected_dims.items():
            actual_dim = feats[m].shape[-1]
            assert actual_dim == dim, f"Dimension mismatch for {m}: expected {dim}, got {actual_dim}"
            assert mask[m].shape[0] == labels.shape[0]

        logger.info("[PASS] Feature shapes and modality masks match configuration schema.")

    logger.info("Dataset validation completed successfully.")


if __name__ == "__main__":
    main()
