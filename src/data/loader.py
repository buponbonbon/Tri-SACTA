"""
src/data/loader.py: Clean data loader for cached CMU-MOSEI features.
"""

from __future__ import annotations

import os
from typing import Dict, Any, List
import numpy as np
import torch

SUBSET_METADATA = {
    "T": {"mods": ["text"], "in_dim": 768, "cost": 1.0, "latency_ms": 1.0},
    "A": {"mods": ["audio"], "in_dim": 74, "cost": 1.2, "latency_ms": 5.0},
    "V": {"mods": ["visual"], "in_dim": 35, "cost": 1.3, "latency_ms": 8.0},
    "TA": {"mods": ["text", "audio"], "in_dim": 842, "cost": 2.2, "latency_ms": 6.0},
    "TV": {"mods": ["text", "visual"], "in_dim": 803, "cost": 2.3, "latency_ms": 9.0},
    "AV": {"mods": ["audio", "visual"], "in_dim": 109, "cost": 2.5, "latency_ms": 13.0},
    "TAV": {"mods": ["text", "audio", "visual"], "in_dim": 877, "cost": 3.5, "latency_ms": 14.0},
}


def load_split(split_name: str, repo_root: str = ".") -> Dict[str, Any]:
    cache_dir = os.path.join(repo_root, "data", "cmu_mosei", "features_cache", split_name)
    meta = torch.load(os.path.join(cache_dir, "meta.pt"), weights_only=False)
    text = torch.load(os.path.join(cache_dir, "text.pt"), weights_only=False).float()
    audio = torch.load(os.path.join(cache_dir, "audio.pt"), weights_only=False).float()
    visual = torch.load(os.path.join(cache_dir, "visual.pt"), weights_only=False).float()

    raw_sentiment = np.array([m["raw_sentiment"] for m in meta], dtype=np.float32)
    labels = np.zeros(len(raw_sentiment), dtype=np.int64)
    labels[raw_sentiment < 0.0] = 0
    labels[raw_sentiment == 0.0] = 1
    labels[raw_sentiment > 0.0] = 2

    video_ids = [m["video_id"] for m in meta]
    segment_ids = [m["segment_id"] for m in meta]

    return {
        "text": text,
        "audio": audio,
        "visual": visual,
        "labels": torch.tensor(labels, dtype=torch.long),
        "video_ids": video_ids,
        "segment_ids": segment_ids,
        "raw_sentiment": raw_sentiment,
    }
