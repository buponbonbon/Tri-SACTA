"""
scripts/v13_generate_workloads.py: Deterministic Workload Generator for Tri-SACTA V13 Benchmark.

Generates 7 realistic stress scenarios on CMU-MOSEI development ('val') and confirmatory ('test') splits:
1. CLEAN_MULTIMODAL: Clean authentic multimodal feature vectors.
2. TEXT_EMBEDDING_NOISE: Emulates ASR transcription drift via Gaussian noise on text embeddings (sigma=0.40).
3. AUDIO_FEATURE_MASKING: Simulates acoustic channel dropouts / microphone mute (audio masked to zero).
4. VISUAL_ZERO_MASK: Simulates camera occlusion / frame packet drop (visual masked to zero).
5. COMPOUND_FAULT: Simultaneous text noise (sigma=0.50) + acoustic channel masking.
6. INTERMITTENT_MISSINGNESS: Bernoulli modality dropping per sample across audio (0.30), visual (0.40), text (0.10).
7. HETEROGENEOUS_DEADLINE_STRESS: Cycles through deadlines (15 ms unimodal, 30 ms partial, 50 ms relaxed).
"""

from __future__ import annotations

import os
import sys
import json
import logging
from typing import Dict, Any, List
import numpy as np
import torch

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_workloads_for_split(split_name: str = "val"):
    logger.info(f"Generating deterministic V13 benchmark workloads for split: '{split_name}'...")
    data_dir = os.path.join(repo_root, "data", "cmu_mosei", "features_cache", split_name)
    out_dir = os.path.join(repo_root, "results", "v13_utility", "workloads", split_name)
    os.makedirs(out_dir, exist_ok=True)

    text_clean = torch.load(os.path.join(data_dir, "text.pt"), map_location="cpu", weights_only=False).float()
    audio_clean = torch.load(os.path.join(data_dir, "audio.pt"), map_location="cpu", weights_only=False).float()
    visual_clean = torch.load(os.path.join(data_dir, "visual.pt"), map_location="cpu", weights_only=False).float()
    meta = torch.load(os.path.join(data_dir, "meta.pt"), map_location="cpu", weights_only=False)

    raw_sentiment = np.array([m["raw_sentiment"] for m in meta], dtype=np.float32)
    labels_3c = np.zeros(len(raw_sentiment), dtype=np.int64)
    labels_3c[raw_sentiment < 0.0] = 0
    labels_3c[raw_sentiment == 0.0] = 1
    labels_3c[raw_sentiment > 0.0] = 2
    N = len(labels_3c)
    video_ids = [m["video_id"] for m in meta]

    workloads: Dict[str, Dict[str, Any]] = {}

    # 1. CLEAN_MULTIMODAL
    workloads["CLEAN_MULTIMODAL"] = {
        "description": "Standard unaltered multimodal features from CMU-MOSEI",
        "text": text_clean.clone(),
        "audio": audio_clean.clone(),
        "visual": visual_clean.clone(),
        "modality_mask": {m: [True] * N for m in ["text", "audio", "visual"]},
        "deadline_ms": [50.0] * N,
        "cost_budget": [3.5] * N,
    }

    # 2. TEXT_EMBEDDING_NOISE
    torch.manual_seed(42)
    noise_t = torch.randn_like(text_clean) * 0.40
    workloads["TEXT_EMBEDDING_NOISE"] = {
        "description": "Emulates ASR transcription errors and acoustic drift via Gaussian noise on text embeddings (sigma=0.40)",
        "text": text_clean + noise_t,
        "audio": audio_clean.clone(),
        "visual": visual_clean.clone(),
        "modality_mask": {m: [True] * N for m in ["text", "audio", "visual"]},
        "deadline_ms": [50.0] * N,
        "cost_budget": [3.5] * N,
    }

    # 3. AUDIO_FEATURE_MASKING
    workloads["AUDIO_FEATURE_MASKING"] = {
        "description": "Acoustic sensor failure / microphone mute: audio features masked to zero",
        "text": text_clean.clone(),
        "audio": torch.zeros_like(audio_clean),
        "visual": visual_clean.clone(),
        "modality_mask": {"text": [True] * N, "audio": [False] * N, "visual": [True] * N},
        "deadline_ms": [50.0] * N,
        "cost_budget": [3.5] * N,
    }

    # 4. VISUAL_ZERO_MASK
    workloads["VISUAL_ZERO_MASK"] = {
        "description": "Visual camera occlusion / frame corruption: visual features masked to zero",
        "text": text_clean.clone(),
        "audio": audio_clean.clone(),
        "visual": torch.zeros_like(visual_clean),
        "modality_mask": {"text": [True] * N, "audio": [True] * N, "visual": [False] * N},
        "deadline_ms": [50.0] * N,
        "cost_budget": [3.5] * N,
    }

    # 5. COMPOUND_FAULT
    torch.manual_seed(100)
    workloads["COMPOUND_FAULT"] = {
        "description": "Simultaneous severe text noise (sigma=0.50) + acoustic channel masking",
        "text": text_clean + (torch.randn_like(text_clean) * 0.50),
        "audio": torch.zeros_like(audio_clean),
        "visual": visual_clean.clone(),
        "modality_mask": {"text": [True] * N, "audio": [False] * N, "visual": [True] * N},
        "deadline_ms": [50.0] * N,
        "cost_budget": [3.5] * N,
    }

    # 6. INTERMITTENT_MISSINGNESS
    rng = np.random.default_rng(2026)
    audio_present = (rng.uniform(size=N) > 0.30).tolist()
    visual_present = (rng.uniform(size=N) > 0.40).tolist()
    text_present = (rng.uniform(size=N) > 0.10).tolist()

    workloads["INTERMITTENT_MISSINGNESS"] = {
        "description": "Dynamic Bernoulli sensor packet drop: Audio p_drop=0.30, Visual p_drop=0.40, Text p_drop=0.10",
        "text": text_clean.clone(),
        "audio": audio_clean.clone(),
        "visual": visual_clean.clone(),
        "modality_mask": {
            "text": text_present,
            "audio": audio_present,
            "visual": visual_present,
        },
        "deadline_ms": [50.0] * N,
        "cost_budget": [3.5] * N,
    }

    # 7. HETEROGENEOUS_DEADLINE_STRESS
    deadlines = [15.0 if i % 3 == 0 else (30.0 if i % 3 == 1 else 50.0) for i in range(N)]
    costs = [1.0 if i % 3 == 0 else (2.2 if i % 3 == 1 else 3.5) for i in range(N)]

    workloads["HETEROGENEOUS_DEADLINE_STRESS"] = {
        "description": "Heterogeneous dynamic QoS stress cycling through 15 ms, 30 ms, and 50 ms deadlines",
        "text": text_clean.clone(),
        "audio": audio_clean.clone(),
        "visual": visual_clean.clone(),
        "modality_mask": {m: [True] * N for m in ["text", "audio", "visual"]},
        "deadline_ms": deadlines,
        "cost_budget": costs,
    }

    manifest = {}
    for name, w in workloads.items():
        fpath = os.path.join(out_dir, f"{name}.pt")
        torch.save({
            "text": w["text"],
            "audio": w["audio"],
            "visual": w["visual"],
            "modality_mask": w["modality_mask"],
            "deadline_ms": w["deadline_ms"],
            "cost_budget": w["cost_budget"],
            "labels": labels_3c,
            "video_ids": video_ids,
            "meta": meta,
        }, fpath)

        manifest[name] = {
            "description": w["description"],
            "num_samples": N,
            "num_video_clusters": len(np.unique(video_ids)),
            "artifact_path": fpath,
        }
        logger.info(f"Saved workload {name} ({N} samples, {len(np.unique(video_ids))} videos)")

    manifest_path = os.path.join(out_dir, "workloads_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"All 7 benchmark workloads for '{split_name}' saved to {out_dir}")


if __name__ == "__main__":
    splits = sys.argv[1:] if len(sys.argv) > 1 else ["val", "test"]
    for s in splits:
        generate_workloads_for_split(s)
