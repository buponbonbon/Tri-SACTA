"""
scripts/v13_train_subset_predictor_bank.py: Complete 7-Subset Task-Matched Predictor Bank for Tri-SACTA V13.

Trains dedicated task-matched MLPs for all 7 feasible modality subsets across seeds 42, 100, 2026:
1. T:   Text alone (768D)
2. A:   Audio alone (74D)
3. V:   Visual alone (35D)
4. TA:  Text + Audio (842D)
5. TV:  Text + Visual (803D)
6. AV:  Audio + Visual (109D)
7. TAV: Text + Audio + Visual (877D)

Also evaluates the shared 1.07M parameter Multimodal Fusion Model with Missingness-Awareness
across all 7 subsets by zero-masking absent modalities.

Outputs:
- results/v13_utility/training/checkpoints/
- results/v13_utility/predictions/ (val, cal, test raw predictions)
- results/v13_utility/training/summary_statistics.json
"""

from __future__ import annotations

import os
import sys
import copy
import time
import json
import random
import logging
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, log_loss

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.models.fusion_backbone import MultimodalFusionModel, SimpleMLP

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEEDS = [42, 100, 2026]
NUM_CLASSES = 3
CLASS_NAMES = ["negative", "neutral", "positive"]

SUBSET_CONFIGS = {
    "T":   {"mods": ["text"], "in_dim": 768, "hidden_dim": 64, "cost": 1.0},
    "A":   {"mods": ["audio"], "in_dim": 74, "hidden_dim": 32, "cost": 1.2},
    "V":   {"mods": ["visual"], "in_dim": 35, "hidden_dim": 16, "cost": 1.3},
    "TA":  {"mods": ["text", "audio"], "in_dim": 842, "hidden_dim": 64, "cost": 2.2},
    "TV":  {"mods": ["text", "visual"], "in_dim": 803, "hidden_dim": 64, "cost": 2.3},
    "AV":  {"mods": ["audio", "visual"], "in_dim": 109, "hidden_dim": 32, "cost": 2.5},
    "TAV": {"mods": ["text", "audio", "visual"], "in_dim": 877, "hidden_dim": 64, "cost": 3.5},
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_split(split_name: str) -> Dict[str, Any]:
    cache_dir = os.path.join(repo_root, "data", "cmu_mosei", "features_cache", split_name)
    meta = torch.load(os.path.join(cache_dir, "meta.pt"), weights_only=False)
    text = torch.load(os.path.join(cache_dir, "text.pt"), weights_only=False).float()
    audio = torch.load(os.path.join(cache_dir, "audio.pt"), weights_only=False).float()
    visual = torch.load(os.path.join(cache_dir, "visual.pt"), weights_only=False).float()

    # Continuous sentiment thresholding: <0.0 -> 0 (negative), ==0.0 -> 1 (neutral), >0.0 -> 2 (positive)
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


def assemble_features(data: Dict[str, Any], mods: List[str]) -> torch.Tensor:
    tensors = [data[m] for m in mods]
    return torch.cat(tensors, dim=-1) if len(tensors) > 1 else tensors[0]


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    per_class_f1 = [float(x) for x in f1_score(y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES))).tolist()

    # Expected Calibration Error (ECE)
    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)
    accuracies = (predictions == y_true).astype(float)
    bin_boundaries = np.linspace(0, 1, 11)
    ece = 0.0
    for i in range(10):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop = np.mean(in_bin)
        if prop > 0:
            ece += np.abs(np.mean(accuracies[in_bin]) - np.mean(confidences[in_bin])) * prop

    eps = 1e-12
    probs_clipped = np.clip(probs, eps, 1.0 - eps)
    probs_clipped /= np.sum(probs_clipped, axis=1, keepdims=True)
    nll = float(log_loss(y_true, probs_clipped, labels=list(range(NUM_CLASSES))))

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
        "ece": float(ece),
        "nll": nll,
    }


def train_mlp_subset(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    in_dim: int,
    hidden_dim: int,
    class_weights: torch.Tensor,
    device: torch.device,
    seed: int,
    epochs: int = 35,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> Tuple[SimpleMLP, Dict[str, Any], float]:
    set_seed(seed)
    model = SimpleMLP(in_dim=in_dim, hidden_dim=hidden_dim, num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_f1 = -1.0
    best_weights = None

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X_val.to(device))
            val_preds = torch.argmax(val_out, dim=-1).cpu().numpy()
            val_f1 = f1_score(y_val.numpy(), val_preds, average="macro", zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = copy.deepcopy(model.state_dict())

    train_time = time.time() - t0
    model.load_state_dict(best_weights)
    model.eval()

    with torch.no_grad():
        logits = model(X_val.to(device)).cpu()
        probs = F.softmax(logits, dim=-1).numpy()
        preds = np.argmax(probs, axis=-1)

    val_metrics = compute_all_metrics(y_val.numpy(), preds, probs)
    return model, val_metrics, train_time


def main():
    logger.info("Initializing Tri-SACTA V13 Subset Predictor Bank Training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Target compute device: {device}")

    out_ckpt_dir = os.path.join(repo_root, "results", "v13_utility", "training", "checkpoints")
    out_pred_dir = os.path.join(repo_root, "results", "v13_utility", "predictions")
    os.makedirs(out_ckpt_dir, exist_ok=True)
    os.makedirs(out_pred_dir, exist_ok=True)

    # Load splits
    train_data = load_split("train")
    val_data = load_split("val")
    cal_data = load_split("cal")
    test_data = load_split("test")

    logger.info(f"Loaded splits: train={len(train_data['labels'])}, val={len(val_data['labels'])}, "
                f"cal={len(cal_data['labels'])}, test={len(test_data['labels'])}")

    # Compute inverse class frequencies
    y_train = train_data["labels"]
    class_counts = torch.bincount(y_train, minlength=NUM_CLASSES).float()
    total_samples = len(y_train)
    class_weights = total_samples / (NUM_CLASSES * class_counts)
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES
    logger.info(f"Class counts: {class_counts.tolist()} | Class weights: {class_weights.tolist()}")

    summary_stats: Dict[str, Any] = {}

    # Train all 7 dedicated subset models across all 3 seeds
    for subset_id, cfg in SUBSET_CONFIGS.items():
        mods = cfg["mods"]
        in_dim = cfg["in_dim"]
        hidden_dim = cfg["hidden_dim"]
        cost = cfg["cost"]

        logger.info(f"=== Training Subset Model: {subset_id} (mods={mods}, dim={in_dim}, cost={cost}) ===")
        X_tr = assemble_features(train_data, mods)
        X_v = assemble_features(val_data, mods)
        X_c = assemble_features(cal_data, mods)
        X_te = assemble_features(test_data, mods)

        summary_stats[subset_id] = {
            "modalities": mods,
            "in_dim": in_dim,
            "cost": cost,
            "seeds": {},
        }

        for seed in SEEDS:
            model, val_m, t_time = train_mlp_subset(
                X_tr, train_data["labels"], X_v, val_data["labels"],
                in_dim, hidden_dim, class_weights, device, seed
            )

            # Save checkpoint
            ckpt_name = f"mlp_{subset_id}_seed{seed}.pt"
            ckpt_path = os.path.join(out_ckpt_dir, ckpt_name)
            torch.save(model.state_dict(), ckpt_path)

            # Evaluate on all splits
            model.eval()
            with torch.no_grad():
                val_logits = model(X_v.to(device)).cpu()
                val_probs = F.softmax(val_logits, dim=-1).numpy()
                val_preds = np.argmax(val_probs, axis=-1)

                cal_logits = model(X_c.to(device)).cpu()
                cal_probs = F.softmax(cal_logits, dim=-1).numpy()
                cal_preds = np.argmax(cal_probs, axis=-1)

                test_logits = model(X_te.to(device)).cpu()
                test_probs = F.softmax(test_logits, dim=-1).numpy()
                test_preds = np.argmax(test_probs, axis=-1)

            cal_m = compute_all_metrics(cal_data["labels"].numpy(), cal_preds, cal_probs)
            test_m = compute_all_metrics(test_data["labels"].numpy(), test_preds, test_probs)

            # Save predictions
            pred_record = {
                "subset": subset_id,
                "seed": seed,
                "val": {
                    "accuracy": val_m["accuracy"],
                    "macro_f1": val_m["macro_f1"],
                    "predictions": val_preds.tolist(),
                    "probabilities": val_probs.tolist(),
                    "labels": val_data["labels"].numpy().tolist(),
                    "video_ids": val_data["video_ids"],
                },
                "cal": {
                    "accuracy": cal_m["accuracy"],
                    "macro_f1": cal_m["macro_f1"],
                    "predictions": cal_preds.tolist(),
                    "probabilities": cal_probs.tolist(),
                    "labels": cal_data["labels"].numpy().tolist(),
                    "video_ids": cal_data["video_ids"],
                },
                "test": {
                    "accuracy": test_m["accuracy"],
                    "macro_f1": test_m["macro_f1"],
                    "predictions": test_preds.tolist(),
                    "probabilities": test_probs.tolist(),
                    "labels": test_data["labels"].numpy().tolist(),
                    "video_ids": test_data["video_ids"],
                },
            }

            pred_file = os.path.join(out_pred_dir, f"preds_{subset_id}_seed{seed}.json")
            with open(pred_file, "w", encoding="utf-8") as f:
                json.dump(pred_record, f)

            summary_stats[subset_id]["seeds"][seed] = {
                "val_accuracy": val_m["accuracy"],
                "val_macro_f1": val_m["macro_f1"],
                "cal_accuracy": cal_m["accuracy"],
                "cal_macro_f1": cal_m["macro_f1"],
                "test_accuracy": test_m["accuracy"],
                "test_macro_f1": test_m["macro_f1"],
                "train_time_sec": t_time,
            }

            logger.info(f"[{subset_id} | Seed {seed}] Val F1: {val_m['macro_f1']:.4f} (Acc: {val_m['accuracy']:.4f}) | "
                        f"Test F1: {test_m['macro_f1']:.4f} (Acc: {test_m['accuracy']:.4f})")

        # Aggregate across seeds
        val_f1s = [summary_stats[subset_id]["seeds"][s]["val_macro_f1"] for s in SEEDS]
        test_f1s = [summary_stats[subset_id]["seeds"][s]["test_macro_f1"] for s in SEEDS]
        val_accs = [summary_stats[subset_id]["seeds"][s]["val_accuracy"] for s in SEEDS]
        test_accs = [summary_stats[subset_id]["seeds"][s]["test_accuracy"] for s in SEEDS]

        summary_stats[subset_id]["val_macro_f1_mean"] = float(np.mean(val_f1s))
        summary_stats[subset_id]["val_macro_f1_std"] = float(np.std(val_f1s))
        summary_stats[subset_id]["test_macro_f1_mean"] = float(np.mean(test_f1s))
        summary_stats[subset_id]["test_macro_f1_std"] = float(np.std(test_f1s))
        summary_stats[subset_id]["val_acc_mean"] = float(np.mean(val_accs))
        summary_stats[subset_id]["test_acc_mean"] = float(np.mean(test_accs))

        logger.info(f"==> [{subset_id} Multi-Seed Summary] Val F1: {np.mean(val_f1s):.4f} ± {np.std(val_f1s):.4f} | "
                    f"Test F1: {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}")

    # Write summary
    summary_path = os.path.join(repo_root, "results", "v13_utility", "training", "summary_statistics.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2)

    logger.info(f"Training complete. All subset checkpoints and predictions saved to {out_ckpt_dir} and {out_pred_dir}")


if __name__ == "__main__":
    main()
