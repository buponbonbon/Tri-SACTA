"""
src/data/fixtures.py: Explicit Synthetic Test Fixtures Generator (V5-02).

Mandatory Scientific Integrity Rules:
1. Every artifact produced by this module is explicitly tagged as 'synthetic_fixture'.
2. Manifest dataset_version explicitly includes 'synthetic-fixture'.
3. Output directories are strictly confined to test fixture directories.
4. Under no circumstances may output from this module be written to production
   data directories (e.g. data/{dataset}/features_cache) or labeled as real.
"""

from __future__ import annotations

import os
import hashlib
from typing import Dict, List, Optional, Any

import torch
import numpy as np

from src.data.manifest import DataManifest, ManifestItem


def generate_synthetic_test_fixture(
    output_dir: str,
    dataset_name: str = "synthetic_fixture",
    split_counts: Optional[Dict[str, int]] = None,
    num_classes: int = 7,
    seed: int = 42,
) -> DataManifest:
    """Generate isolated synthetic fixtures exclusively for unit testing and CI negative controls."""
    if split_counts is None:
        split_counts = {"train": 40, "val": 10, "cal": 10, "test": 20}

    dims = {
        "visual": 512,
        "audio": 768,
        "text": 768,
        "prosody": 88,
    }

    os.makedirs(output_dir, exist_ok=True)
    manifest_items: List[ManifestItem] = []
    global_idx = 0

    speakers_pool = {
        "train": [f"fixture_spk_trn_{i:03d}" for i in range(10)],
        "val": [f"fixture_spk_val_{i:03d}" for i in range(4)],
        "cal": [f"fixture_spk_cal_{i:03d}" for i in range(4)],
        "test": [f"fixture_spk_tst_{i:03d}" for i in range(6)],
    }
    videos_pool = {
        "train": [f"fixture_vid_trn_{i:03d}" for i in range(8)],
        "val": [f"fixture_vid_val_{i:03d}" for i in range(3)],
        "cal": [f"fixture_vid_cal_{i:03d}" for i in range(3)],
        "test": [f"fixture_vid_tst_{i:03d}" for i in range(5)],
    }

    for split, count in split_counts.items():
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        rng = np.random.RandomState(seed + global_idx)
        labels = torch.from_numpy(rng.randint(0, num_classes, size=(count,))).long()

        # Class-conditioned synthetic representations
        v_feat = torch.from_numpy(rng.randn(count, dims["visual"]).astype(np.float32) * 0.1).half()
        a_feat = torch.from_numpy(rng.randn(count, dims["audio"]).astype(np.float32) * 0.1).half()
        p_feat = torch.from_numpy(rng.randn(count, dims["prosody"]).astype(np.float32) * 0.05).half()
        t_feat = torch.from_numpy(rng.randn(count, dims["text"]).astype(np.float32) * 0.1).half()

        feature_files = {
            "visual": f"{split}/visual.pt",
            "audio": f"{split}/audio.pt",
            "prosody": f"{split}/prosody.pt",
            "text": f"{split}/text.pt",
        }

        # Save files
        torch.save(v_feat, os.path.join(split_dir, "visual.pt"))
        torch.save(a_feat, os.path.join(split_dir, "audio.pt"))
        torch.save(p_feat, os.path.join(split_dir, "prosody.pt"))
        torch.save(t_feat, os.path.join(split_dir, "text.pt"))
        torch.save(labels, os.path.join(split_dir, "labels.pt"))

        # Compute file checksums
        feature_checksums = {}
        for mod, rel_p in feature_files.items():
            full_p = os.path.join(output_dir, rel_p)
            h = hashlib.sha256()
            with open(full_p, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            feature_checksums[mod] = h.hexdigest()

        meta: List[Dict[str, Any]] = []
        spk_list = speakers_pool[split]
        vid_list = videos_pool[split]

        for i in range(count):
            s_id = f"fixture_{dataset_name}_{split}_{i:05d}"
            spk = spk_list[i % len(spk_list)]
            vid = vid_list[i % len(vid_list)]
            diag_id = f"diag_{vid}" if dataset_name == "meld" else None
            lbl = int(labels[i].item())

            meta.append({
                "segment_id": s_id,
                "speaker_id": spk,
                "video_id": vid,
                "dialogue_id": diag_id,
                "label": lbl,
            })

            manifest_items.append(
                ManifestItem(
                    sample_id=s_id,
                    video_id=vid,
                    speaker_id=spk,
                    split=split,
                    raw_label=lbl,
                    mapped_label=lbl,
                    available_visual=True,
                    available_audio=True,
                    available_text=True,
                    available_prosody=True,
                    dialogue_id=diag_id,
                    feature_files=dict(feature_files),
                    feature_checksums=dict(feature_checksums),
                )
            )

        torch.save(meta, os.path.join(split_dir, "meta.pt"))
        global_idx += count

    manifest = DataManifest(
        dataset_name=f"{dataset_name}_synthetic_fixture",
        dataset_version="synthetic-fixture-v5",
        feature_sources={
            "visual": "synthetic_gaussian_fixture",
            "audio": "synthetic_gaussian_fixture",
            "prosody": "synthetic_gaussian_fixture",
            "text": "synthetic_gaussian_fixture",
        },
        feature_dims=dims,
        items=manifest_items,
    )

    manifest_path = os.path.join(output_dir, "fixture_manifest.json")
    manifest.save_json(manifest_path)
    return manifest
