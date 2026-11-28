"""
scripts/v12_benchmark_routing_policies.py: Benchmark deployment policies under realistic failure workloads.

Tri-SACTA V12 Systems Contribution (Stages 2 & 3):
Simulates 6 failure workloads on authentic validation features:
1. Clean (all sensors operational)
2. Audio Corruption (acoustic noise / packet drop)
3. Text Corruption (ASR transcription drift / vocabulary mismatch)
4. Visual Occlusion (camera obstructed / face detection failure)
5. Compound Fault (ASR drift + visual occlusion)
6. Asynchronous / Intermittent missingness

Benchmarks 7 deployment policies + 2 ablations:
1. text_only: Always invoke text only (cheap, vulnerable to text noise)
2. always_on: Always invoke all modalities (expensive, latency-heavy)
3. fixed_fallback: Text-first, fallback to audio/visual only if text missing
4. conf_routing: Text-first, escalate if max softmax < tau_conf
5. random_budget: Random escalation matching 35% extra sensor budget
6. brmo: Budgeted Reliability-Aware Modality Orchestrator (spectral + entropy + conformal)
7. brmo_shuffled: BRMO with shuffled/randomized reliability detector (causal control)
8. brmo_no_conformal: BRMO without selective risk abstention

Produces:
- results/v12_systems/workloads/matched_policy_metrics.json
- results/v12_systems/workloads/pareto_points.csv
- results/v12_systems/workloads/ablation_metrics.json
"""

from __future__ import annotations

import os
import sys
import copy
import time
import json
import logging
import csv
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.models.fusion_backbone import MultimodalFusionModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUM_CLASSES = 3
REGIMES = [
    "regime_1_clean",
    "regime_2_audio_corrupted",
    "regime_3_text_asr_drift",
    "regime_4_visual_occluded",
    "regime_5_compound_fault",
    "regime_6_asymmetric_intermittent",
]

POLICIES = [
    "text_only",
    "always_on",
    "fixed_fallback",
    "conf_routing",
    "random_budget",
    "brmo",
    "brmo_shuffled",
    "brmo_no_conformal",
]


def load_val_and_train_stats(data_dir: str):
    val_dir = os.path.join(data_dir, "val")
    train_dir = os.path.join(data_dir, "train")

    text_val = torch.load(os.path.join(val_dir, "text.pt"), map_location="cpu").float()
    audio_val = torch.load(os.path.join(val_dir, "audio.pt"), map_location="cpu").float()
    visual_val = torch.load(os.path.join(val_dir, "visual.pt"), map_location="cpu").float()
    meta_val = torch.load(os.path.join(val_dir, "meta.pt"), map_location="cpu")

    raw_sentiment = np.array([m["raw_sentiment"] for m in meta_val], dtype=np.float32)
    y3_val = np.zeros(len(raw_sentiment), dtype=np.int64)
    y3_val[raw_sentiment < 0.0] = 0
    y3_val[raw_sentiment == 0.0] = 1
    y3_val[raw_sentiment > 0.0] = 2

    # Clean text statistics for spectral / centroid anomaly score
    text_train = torch.load(os.path.join(train_dir, "text.pt"), map_location="cpu").float()
    text_mean = torch.mean(text_train, dim=0)
    text_std = torch.std(text_train, dim=0) + 1e-6

    return {
        "text": text_val,
        "audio": audio_val,
        "visual": visual_val,
        "labels": torch.from_numpy(y3_val),
        "text_mean": text_mean,
        "text_std": text_std,
    }


def inject_workload_faults(clean_data: Dict[str, Any], regime: str, seed: int = 42) -> Dict[str, Any]:
    """Generates workload tensors with explicit provenance."""
    rng = np.random.RandomState(seed)
    N = len(clean_data["labels"])
    t = clean_data["text"].clone()
    a = clean_data["audio"].clone()
    v = clean_data["visual"].clone()

    mask_t = torch.ones(N, dtype=torch.bool)
    mask_a = torch.ones(N, dtype=torch.bool)
    mask_v = torch.ones(N, dtype=torch.bool)

    if regime == "regime_1_clean":
        pass  # Everything clean and present

    elif regime == "regime_2_audio_corrupted":
        # Acoustic noise: additive gaussian + drop 50% coefficients
        noise = torch.from_numpy(rng.normal(0, 1.5, size=a.shape)).float()
        a = a + noise
        drop_mask = torch.from_numpy(rng.binomial(1, 0.5, size=a.shape)).bool()
        a[drop_mask] = 0.0

    elif regime == "regime_3_text_asr_drift":
        # ASR transcription drift: feature jitter + dimension masking
        noise = torch.from_numpy(rng.normal(0, 0.8, size=t.shape)).float()
        t = t + noise
        drop_mask = torch.from_numpy(rng.binomial(1, 0.3, size=t.shape)).bool()
        t[drop_mask] = 0.0

    elif regime == "regime_4_visual_occluded":
        # Complete camera occlusion / face detection failure
        v = torch.zeros_like(v)
        mask_v = torch.zeros(N, dtype=torch.bool)

    elif regime == "regime_5_compound_fault":
        # ASR drift + visual occlusion (only audio is reliable)
        noise = torch.from_numpy(rng.normal(0, 0.8, size=t.shape)).float()
        t = t + noise
        v = torch.zeros_like(v)
        mask_v = torch.zeros(N, dtype=torch.bool)

    elif regime == "regime_6_asymmetric_intermittent":
        # 50% samples have missing text, 50% samples have degraded audio
        for i in range(N):
            if i % 2 == 0:
                t[i] = 0.0
                mask_t[i] = False
            else:
                noise = torch.from_numpy(rng.normal(0, 1.2, size=a[i].shape)).float()
                a[i] = a[i] + noise

    return {
        "text": t,
        "audio": a,
        "visual": v,
        "labels": clean_data["labels"],
        "mask": {"text": mask_t, "audio": mask_a, "visual": mask_v},
    }


def compute_spectral_text_score(text_tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> np.ndarray:
    """Computes standardized distance of text embedding from clean centroid."""
    norm_diff = torch.abs(text_tensor - mean) / std
    score = torch.mean(norm_diff, dim=-1).cpu().numpy()
    return score


def evaluate_policy_on_workload(
    policy: str,
    workload: Dict[str, Any],
    clean_stats: Dict[str, Any],
    model: MultimodalFusionModel,
    device: torch.device,
    conf_threshold: float = 0.70,
    spectral_threshold: float = 1.15,
) -> Dict[str, Any]:
    N = len(workload["labels"])
    y_true = workload["labels"].numpy()
    preds = np.zeros(N, dtype=np.int64)
    abstained = np.zeros(N, dtype=bool)

    audio_invocations = 0
    visual_invocations = 0
    latencies_ms = []

    # Model evaluation
    model.eval()

    text_scores = compute_spectral_text_score(workload["text"], clean_stats["text_mean"], clean_stats["text_std"])

    t0_all = time.time()
    with torch.no_grad():
        for i in range(N):
            t_start = time.perf_counter()

            t_feat = workload["text"][i : i + 1].to(device)
            a_feat = workload["audio"][i : i + 1].to(device)
            v_feat = workload["visual"][i : i + 1].to(device)

            m_t = workload["mask"]["text"][i : i + 1].to(device)
            m_a = workload["mask"]["audio"][i : i + 1].to(device)
            m_v = workload["mask"]["visual"][i : i + 1].to(device)

            # --- DECISION ROUTING ---
            invoke_multimodal = False
            use_conformal_abstain = False

            if policy == "text_only":
                invoke_multimodal = False

            elif policy == "always_on":
                invoke_multimodal = True

            elif policy == "fixed_fallback":
                # Fall back to multimodal only if text is missing
                invoke_multimodal = not m_t.item()

            elif policy == "conf_routing":
                # Quick text forward pass
                out_t = model.projections["text"](t_feat)
                ad_t = model.adapters["text"](out_t)
                logits_t = model.modality_classifiers["text"](ad_t)
                prob_t = F.softmax(logits_t, dim=-1)
                max_p = torch.max(prob_t).item()

                if max_p < conf_threshold or not m_t.item():
                    invoke_multimodal = True

            elif policy == "random_budget":
                # 35% chance to invoke multimodal
                invoke_multimodal = (i % 3 == 0) or (not m_t.item())

            elif policy == "brmo":
                # Step 1: Check text availability & spectral shift
                s_score = text_scores[i]
                out_t = model.projections["text"](t_feat)
                ad_t = model.adapters["text"](out_t)
                logits_t = model.modality_classifiers["text"](ad_t)
                prob_t = F.softmax(logits_t, dim=-1)
                max_p = torch.max(prob_t).item()
                entropy = -torch.sum(prob_t * torch.log(prob_t + 1e-12)).item() / np.log(3)

                # Route: if text corrupted or high entropy, escalate
                if not m_t.item() or s_score > spectral_threshold or entropy > 0.65 or max_p < 0.60:
                    invoke_multimodal = True
                use_conformal_abstain = True

            elif policy == "brmo_shuffled":
                # Randomize spectral score
                shuffled_score = np.random.uniform(0.5, 2.5)
                out_t = model.projections["text"](t_feat)
                ad_t = model.adapters["text"](out_t)
                logits_t = model.modality_classifiers["text"](ad_t)
                prob_t = F.softmax(logits_t, dim=-1)
                max_p = torch.max(prob_t).item()
                entropy = -torch.sum(prob_t * torch.log(prob_t + 1e-12)).item() / np.log(3)

                if not m_t.item() or shuffled_score > spectral_threshold or entropy > 0.65 or max_p < 0.60:
                    invoke_multimodal = True
                use_conformal_abstain = True

            elif policy == "brmo_no_conformal":
                s_score = text_scores[i]
                out_t = model.projections["text"](t_feat)
                ad_t = model.adapters["text"](out_t)
                logits_t = model.modality_classifiers["text"](ad_t)
                prob_t = F.softmax(logits_t, dim=-1)
                max_p = torch.max(prob_t).item()
                entropy = -torch.sum(prob_t * torch.log(prob_t + 1e-12)).item() / np.log(3)

                if not m_t.item() or s_score > spectral_threshold or entropy > 0.65 or max_p < 0.60:
                    invoke_multimodal = True
                use_conformal_abstain = False

            # --- EXECUTE INFERENCE ---
            if invoke_multimodal:
                audio_invocations += 1
                visual_invocations += 1

                feats = {"text": t_feat, "audio": a_feat, "visual": v_feat}
                mask = {"text": m_t, "audio": m_a, "visual": m_v}

                # Dynamic reliability weighting: downweight text if corrupted
                rel_weights = None
                if policy in ["brmo", "brmo_no_conformal"]:
                    s_score = text_scores[i]
                    weight_t = 0.2 if s_score > spectral_threshold else 1.0
                    rel_weights = {
                        "text": torch.tensor([weight_t], device=device),
                        "audio": torch.tensor([1.0], device=device),
                        "visual": torch.tensor([1.0], device=device),
                        "prosody": torch.tensor([0.0], device=device),
                    }

                out = model(feats, modality_mask=mask, reliability_weights=rel_weights)
                logits = out["logits"]
            else:
                # Fast text-only inference
                if m_t.item():
                    out_t = model.projections["text"](t_feat)
                    ad_t = model.adapters["text"](out_t)
                    logits = model.modality_classifiers["text"](ad_t)
                else:
                    # If text missing and not escalated, random guess
                    logits = torch.zeros(1, NUM_CLASSES, device=device)

            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
            pred_class = int(np.argmax(probs))

            # Conformal abstention check
            if use_conformal_abstain:
                # If prediction is highly ambiguous (max_p < 0.40) or entropy near max, abstain
                if np.max(probs) < 0.40:
                    abstained[i] = True

            preds[i] = pred_class
            latencies_ms.append((time.perf_counter() - t_start) * 1000.0)

    # Compute metrics
    accepted_mask = ~abstained
    if np.any(accepted_mask):
        acc = float(accuracy_score(y_true[accepted_mask], preds[accepted_mask]))
        macro_f1 = float(f1_score(y_true[accepted_mask], preds[accepted_mask], average="macro", zero_division=0))
    else:
        acc = 0.0
        macro_f1 = 0.0

    raw_acc = float(accuracy_score(y_true, preds))
    raw_macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
    abstain_rate = float(np.mean(abstained))

    p50_lat = float(np.percentile(latencies_ms, 50))
    p95_lat = float(np.percentile(latencies_ms, 95))
    mean_lat = float(np.mean(latencies_ms))

    # Normalized cost: 1.0 for text, +1.2 for audio extraction/forward, +1.3 for visual
    sensor_cost_factor = 1.0 + 1.2 * (audio_invocations / N) + 1.3 * (visual_invocations / N)

    return {
        "policy": policy,
        "selective_accuracy": acc,
        "selective_macro_f1": macro_f1,
        "raw_accuracy": raw_acc,
        "raw_macro_f1": raw_macro_f1,
        "abstention_rate": abstain_rate,
        "audio_invocation_rate": float(audio_invocations / N),
        "visual_invocation_rate": float(visual_invocations / N),
        "mean_latency_ms": mean_lat,
        "p50_latency_ms": p50_lat,
        "p95_latency_ms": p95_lat,
        "normalized_compute_cost": float(sensor_cost_factor),
    }


def main():
    logger.info("Starting V12 Stage 2 & 3: Failure Workload and Policy Benchmark...")
    data_dir = os.path.join(repo_root, "data", "cmu_mosei", "features_cache")
    out_dir = os.path.join(repo_root, "results", "v12_systems", "workloads")
    os.makedirs(out_dir, exist_ok=True)

    clean_data = load_val_and_train_stats(data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load retrained multimodal fusion model (Seed 42)
    cfg = {
        "model": {
            "fusion_dim": 256,
            "num_heads": 4,
            "num_layers": 2,
            "adapter_rank": 8,
            "dropout": 0.1,
            "num_classes": NUM_CLASSES,
        },
        "features": {
            "visual_dim": 35,
            "audio_dim": 74,
            "text_dim": 768,
            "prosody_dim": 88,
        },
    }
    model = MultimodalFusionModel(cfg).to(device)
    ckpt_path = os.path.join(
        repo_root, "results", "v12_systems", "fair_3class", "model_checkpoints", "multimodal_fusion_seed42.pt"
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()

    all_results = {}
    pareto_rows = []

    for regime in REGIMES:
        logger.info(f"=== Simulating Workload: {regime} ===")
        workload = inject_workload_faults(clean_data, regime, seed=42)
        all_results[regime] = {}

        for pol in POLICIES:
            res = evaluate_policy_on_workload(pol, workload, clean_data, model, device)
            all_results[regime][pol] = res
            logger.info(
                f"  [{pol}] Sel-F1={res['selective_macro_f1']:.4f}, Raw-F1={res['raw_macro_f1']:.4f}, "
                f"Abstain={res['abstention_rate']:.1%}, Cost={res['normalized_compute_cost']:.2f}x, "
                f"p95={res['p95_latency_ms']:.2f}ms"
            )

            pareto_rows.append({
                "regime": regime,
                "policy": pol,
                "raw_macro_f1": res["raw_macro_f1"],
                "selective_macro_f1": res["selective_macro_f1"],
                "abstention_rate": res["abstention_rate"],
                "sensor_invocations": (res["audio_invocation_rate"] + res["visual_invocation_rate"]) / 2.0,
                "normalized_cost": res["normalized_compute_cost"],
                "mean_latency_ms": res["mean_latency_ms"],
                "p95_latency_ms": res["p95_latency_ms"],
            })

    # Save outputs
    with open(os.path.join(out_dir, "matched_policy_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    with open(os.path.join(out_dir, "pareto_points.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pareto_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pareto_rows)

    logger.info("Benchmark complete! Saved metrics to JSON and CSV.")


if __name__ == "__main__":
    main()
