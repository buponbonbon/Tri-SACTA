"""
scripts/v12_1_train_predictor_bank.py: Train and save the comprehensive, fair predictor bank.

Part of Tri-SACTA V12.1 Systems Research Rectification (Stage A).
Trains and evaluates task-matched 3-class models across seeds [42, 100, 2026]:
1. Majority class baseline
2. Text-only Logistic Regression
3. Text-only Neural MLP (768 -> 64 -> 3)
4. Audio-only Neural MLP (74 -> 32 -> 3)
5. Visual-only Neural MLP (35 -> 16 -> 3)
6. Multimodal Concat Logistic Regression (877D)
7. Multimodal Concat Neural MLP (877 -> 64 -> 3)
8. Multimodal Fusion Model (1.07M cross-modal transformer)
9. Multimodal Fusion Missingness-Aware (with 20% training modality dropout)

Features:
- Genuine checkpoints saved to results/v12_1_systems/training/checkpoints/
- Raw per-sample predictions saved to results/v12_1_systems/predictions/
- True video-grouped block bootstrap with paired deltas against Text-only MLP.
- Strict isolation of development-exposed test set.
"""

from __future__ import annotations

import os
import sys
import copy
import time
import json
import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, log_loss

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.models.fusion_backbone import MultimodalFusionModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEEDS = [42, 100, 2026]
NUM_CLASSES = 3
CLASS_NAMES = ["negative", "neutral", "positive"]


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_3class_data(data_dir: str):
    """Loads authentic features and maps sentiment to 3 classes."""
    data = {}
    for split in ["train", "val", "cal"]:
        split_dir = os.path.join(data_dir, split)
        text = torch.load(os.path.join(split_dir, "text.pt"), map_location="cpu", weights_only=False).float()
        audio = torch.load(os.path.join(split_dir, "audio.pt"), map_location="cpu", weights_only=False).float()
        visual = torch.load(os.path.join(split_dir, "visual.pt"), map_location="cpu", weights_only=False).float()
        meta = torch.load(os.path.join(split_dir, "meta.pt"), map_location="cpu", weights_only=False)

        raw_sentiment = np.array([m["raw_sentiment"] for m in meta], dtype=np.float32)
        y3 = np.zeros(len(raw_sentiment), dtype=np.int64)
        y3[raw_sentiment < 0.0] = 0
        y3[raw_sentiment == 0.0] = 1
        y3[raw_sentiment > 0.0] = 2

        video_ids = [m["video_id"] for m in meta]
        sample_ids = [m.get("segment_id", f"{split}_{i}") for i, m in enumerate(meta)]
        concat = torch.cat([text, audio, visual], dim=-1)

        data[split] = {
            "text": text,
            "audio": audio,
            "visual": visual,
            "concat": concat,
            "labels": torch.from_numpy(y3),
            "video_ids": video_ids,
            "sample_ids": sample_ids,
            "meta": meta,
        }
    return data


def compute_calibration_metrics(probs: np.ndarray, y_true: np.ndarray, num_bins: int = 10) -> Dict[str, float]:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    one_hot = np.zeros_like(probs)
    for idx, label in enumerate(y_true):
        one_hot[idx, label] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    eps = 1e-12
    probs_clipped = np.clip(probs, eps, 1.0 - eps)
    probs_clipped /= np.sum(probs_clipped, axis=1, keepdims=True)
    nll = float(log_loss(y_true, probs_clipped, labels=list(range(NUM_CLASSES))))

    return {
        "ece": float(ece),
        "brier": brier,
        "nll": nll,
        "mean_confidence": float(np.mean(confidences)),
        "std_confidence": float(np.std(confidences)),
    }


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    per_class_f1 = [float(x) for x in f1_score(y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES))).tolist()
    cal = compute_calibration_metrics(probs, y_true)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
        "calibration": cal,
    }


class SimpleMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, num_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    class_weights: torch.Tensor,
    hidden_dim: int,
    device: torch.device,
    seed: int,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> Tuple[SimpleMLP, Dict[str, Any], np.ndarray, np.ndarray, float]:
    set_seed(seed)
    model = SimpleMLP(X_train.shape[1], hidden_dim=hidden_dim, num_classes=NUM_CLASSES).to(device)
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

    metrics = compute_all_metrics(y_val.numpy(), preds, probs)
    return model, metrics, preds, probs, train_time


def train_multimodal_fusion(
    train_data: Dict[str, Any],
    val_data: Dict[str, Any],
    class_weights: torch.Tensor,
    device: torch.device,
    seed: int,
    modality_dropout: float = 0.0,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    aux_weight: float = 0.05,
) -> Tuple[MultimodalFusionModel, Dict[str, Any], np.ndarray, np.ndarray, float, float]:
    set_seed(seed)
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

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model = MultimodalFusionModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    N_train = len(train_data["labels"])
    best_val_f1 = -1.0
    best_weights = None
    patience = 7
    patience_counter = 0

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(N_train)
        for start_idx in range(0, N_train, batch_size):
            b_idx = indices[start_idx : start_idx + batch_size]
            B = len(b_idx)
            feats = {
                "text": train_data["text"][b_idx].to(device),
                "audio": train_data["audio"][b_idx].to(device),
                "visual": train_data["visual"][b_idx].to(device),
            }

            # Optional modality dropout during training for missingness-aware model
            mask = {
                "text": torch.ones(B, dtype=torch.bool, device=device),
                "audio": torch.ones(B, dtype=torch.bool, device=device),
                "visual": torch.ones(B, dtype=torch.bool, device=device),
            }
            if modality_dropout > 0.0:
                for m in ["audio", "visual"]:
                    drop_prob = torch.rand(B, device=device) < modality_dropout
                    mask[m] = ~drop_prob

            labels = train_data["labels"][b_idx].to(device)

            optimizer.zero_grad()
            outputs = model(feats, modality_mask=mask)
            loss_fused = criterion(outputs["logits"], labels)

            aux_losses = []
            for m in ["text", "audio", "visual"]:
                if m in outputs.get("per_modality_logits", {}):
                    m_mask = mask[m]
                    if m_mask.any():
                        aux_losses.append(criterion(outputs["per_modality_logits"][m][m_mask], labels[m_mask]))
            loss_aux = torch.stack(aux_losses).mean() if aux_losses else torch.tensor(0.0, device=device)
            loss = loss_fused + aux_weight * loss_aux

            loss.backward()
            optimizer.step()

        # Val eval
        model.eval()
        with torch.no_grad():
            val_feats = {
                "text": val_data["text"].to(device),
                "audio": val_data["audio"].to(device),
                "visual": val_data["visual"].to(device),
            }
            val_mask = {
                "text": torch.ones(len(val_data["labels"]), dtype=torch.bool, device=device),
                "audio": torch.ones(len(val_data["labels"]), dtype=torch.bool, device=device),
                "visual": torch.ones(len(val_data["labels"]), dtype=torch.bool, device=device),
            }
            val_out = model(val_feats, modality_mask=val_mask)
            val_preds = torch.argmax(val_out["logits"], dim=-1).cpu().numpy()
            val_f1 = f1_score(val_data["labels"].numpy(), val_preds, average="macro", zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Seed {seed}: Early stopping at epoch {epoch + 1}")
                break

    train_time = time.time() - t0
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0

    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        val_feats = {
            "text": val_data["text"].to(device),
            "audio": val_data["audio"].to(device),
            "visual": val_data["visual"].to(device),
        }
        val_mask = {
            "text": torch.ones(len(val_data["labels"]), dtype=torch.bool, device=device),
            "audio": torch.ones(len(val_data["labels"]), dtype=torch.bool, device=device),
            "visual": torch.ones(len(val_data["labels"]), dtype=torch.bool, device=device),
        }
        final_out = model(val_feats, modality_mask=val_mask)
        logits = final_out["logits"].cpu()
        probs = F.softmax(logits, dim=-1).numpy()
        preds = np.argmax(probs, axis=-1)

    metrics = compute_all_metrics(val_data["labels"].numpy(), preds, probs)
    return model, metrics, preds, probs, train_time, peak_vram_mb


def video_grouped_block_bootstrap_paired(
    y_true: np.ndarray,
    preds_candidate: np.ndarray,
    preds_ref: np.ndarray,
    video_ids: List[str],
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Paired block bootstrap by video group."""
    rng = np.random.RandomState(seed)
    unique_vids = np.array(sorted(list(set(video_ids))))
    n_vids = len(unique_vids)
    vid_to_indices = {v: np.where(np.array(video_ids) == v)[0] for v in unique_vids}

    acc_c_list, f1_c_list = [], []
    delta_acc_list, delta_f1_list = [], []

    for _ in range(n_bootstraps):
        sampled_vids = rng.choice(unique_vids, size=n_vids, replace=True)
        sampled_idx = np.concatenate([vid_to_indices[v] for v in sampled_vids])

        sub_true = y_true[sampled_idx]
        sub_c = preds_candidate[sampled_idx]
        sub_r = preds_ref[sampled_idx]

        acc_c = accuracy_score(sub_true, sub_c)
        f1_c = f1_score(sub_true, sub_c, average="macro", zero_division=0)
        acc_r = accuracy_score(sub_true, sub_r)
        f1_r = f1_score(sub_true, sub_r, average="macro", zero_division=0)

        acc_c_list.append(acc_c)
        f1_c_list.append(f1_c)
        delta_acc_list.append(acc_c - acc_r)
        delta_f1_list.append(f1_c - f1_r)

    return {
        "accuracy": {
            "mean": float(np.mean(acc_c_list)),
            "std": float(np.std(acc_c_list)),
            "ci_95": [float(np.percentile(acc_c_list, 2.5)), float(np.percentile(acc_c_list, 97.5))],
        },
        "macro_f1": {
            "mean": float(np.mean(f1_c_list)),
            "std": float(np.std(f1_c_list)),
            "ci_95": [float(np.percentile(f1_c_list, 2.5)), float(np.percentile(f1_c_list, 97.5))],
        },
        "paired_delta_macro_f1": {
            "mean": float(np.mean(delta_f1_list)),
            "std": float(np.std(delta_f1_list)),
            "ci_95": [float(np.percentile(delta_f1_list, 2.5)), float(np.percentile(delta_f1_list, 97.5))],
            "one_sided_lower_95": float(np.percentile(delta_f1_list, 5.0)),
        },
    }


def main():
    logger.info("Starting V12.1 Stage A: Fair Predictor Bank Training & Prediction Extraction...")
    data_dir = os.path.join(repo_root, "data", "cmu_mosei", "features_cache")
    out_dir = os.path.join(repo_root, "results", "v12_1_systems")
    ckpt_dir = os.path.join(out_dir, "training", "checkpoints")
    pred_dir = os.path.join(out_dir, "predictions")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    data = load_3class_data(data_dir)
    y_train = data["train"]["labels"]
    y_val = data["val"]["labels"]
    val_vids = data["val"]["video_ids"]
    val_sids = data["val"]["sample_ids"]

    counts = np.bincount(y_train.numpy(), minlength=NUM_CLASSES)
    weights = len(y_train) / (NUM_CLASSES * counts.astype(np.float32))
    class_weights = torch.from_numpy(weights).float()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Target compute device: {device}")

    # Baseline 1: Majority Class
    maj_class = int(np.argmax(counts))
    preds_maj = np.full(len(y_val), maj_class)
    probs_maj = np.zeros((len(y_val), NUM_CLASSES))
    probs_maj[:, maj_class] = 1.0
    maj_metrics = compute_all_metrics(y_val.numpy(), preds_maj, probs_maj)

    # Baseline 2: Text-only Logistic Regression
    clf_text_lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    clf_text_lr.fit(data["train"]["text"].numpy(), y_train.numpy())
    probs_text_lr = clf_text_lr.predict_proba(data["val"]["text"].numpy())
    preds_text_lr = clf_text_lr.predict(data["val"]["text"].numpy())
    text_lr_metrics = compute_all_metrics(y_val.numpy(), preds_text_lr, probs_text_lr)

    # Baseline 5: Concat Multimodal Logistic Regression
    clf_concat_lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    clf_concat_lr.fit(data["train"]["concat"].numpy(), y_train.numpy())
    probs_concat_lr = clf_concat_lr.predict_proba(data["val"]["concat"].numpy())
    preds_concat_lr = clf_concat_lr.predict(data["val"]["concat"].numpy())
    concat_lr_metrics = compute_all_metrics(y_val.numpy(), preds_concat_lr, probs_concat_lr)

    # Save deterministic baselines predictions
    for name, pr, pb in [("majority_class", preds_maj, probs_maj),
                         ("text_only_lr", preds_text_lr, probs_text_lr),
                         ("multimodal_concat_lr", preds_concat_lr, probs_concat_lr)]:
        with open(os.path.join(pred_dir, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump({
                "sample_ids": val_sids,
                "video_ids": val_vids,
                "y_true": y_val.numpy().tolist(),
                "y_pred": pr.tolist(),
                "probabilities": pb.tolist(),
            }, f)

    all_seed_results = {
        "text_only_mlp": {},
        "audio_only_mlp": {},
        "visual_only_mlp": {},
        "multimodal_concat_mlp": {},
        "multimodal_fusion": {},
        "multimodal_fusion_missingness_aware": {},
    }

    # Reference predictions for paired bootstrap (Text-only MLP Seed 42)
    ref_preds_text_mlp = None

    for seed in SEEDS:
        logger.info(f"--- Training Predictor Bank for Seed {seed} ---")

        # 3. Text-only MLP
        t_mlp, t_met, t_pred, t_prob, t_time = train_mlp(
            data["train"]["text"], y_train, data["val"]["text"], y_val, class_weights, 64, device, seed
        )
        torch.save(t_mlp.state_dict(), os.path.join(ckpt_dir, f"text_only_mlp_seed{seed}.pt"))
        all_seed_results["text_only_mlp"][f"seed_{seed}"] = t_met
        with open(os.path.join(pred_dir, f"text_only_mlp_seed{seed}.json"), "w", encoding="utf-8") as f:
            json.dump({"sample_ids": val_sids, "video_ids": val_vids, "y_true": y_val.numpy().tolist(),
                       "y_pred": t_pred.tolist(), "probabilities": t_prob.tolist()}, f)
        if seed == 42:
            ref_preds_text_mlp = t_pred

        # 4. Audio-only MLP
        a_mlp, a_met, a_pred, a_prob, a_time = train_mlp(
            data["train"]["audio"], y_train, data["val"]["audio"], y_val, class_weights, 32, device, seed
        )
        torch.save(a_mlp.state_dict(), os.path.join(ckpt_dir, f"audio_only_mlp_seed{seed}.pt"))
        all_seed_results["audio_only_mlp"][f"seed_{seed}"] = a_met
        with open(os.path.join(pred_dir, f"audio_only_mlp_seed{seed}.json"), "w", encoding="utf-8") as f:
            json.dump({"sample_ids": val_sids, "video_ids": val_vids, "y_true": y_val.numpy().tolist(),
                       "y_pred": a_pred.tolist(), "probabilities": a_prob.tolist()}, f)

        # 5. Visual-only MLP
        v_mlp, v_met, v_pred, v_prob, v_time = train_mlp(
            data["train"]["visual"], y_train, data["val"]["visual"], y_val, class_weights, 16, device, seed
        )
        torch.save(v_mlp.state_dict(), os.path.join(ckpt_dir, f"visual_only_mlp_seed{seed}.pt"))
        all_seed_results["visual_only_mlp"][f"seed_{seed}"] = v_met
        with open(os.path.join(pred_dir, f"visual_only_mlp_seed{seed}.json"), "w", encoding="utf-8") as f:
            json.dump({"sample_ids": val_sids, "video_ids": val_vids, "y_true": y_val.numpy().tolist(),
                       "y_pred": v_pred.tolist(), "probabilities": v_prob.tolist()}, f)

        # 6. Concat MLP
        c_mlp, c_met, c_pred, c_prob, c_time = train_mlp(
            data["train"]["concat"], y_train, data["val"]["concat"], y_val, class_weights, 64, device, seed
        )
        torch.save(c_mlp.state_dict(), os.path.join(ckpt_dir, f"multimodal_concat_mlp_seed{seed}.pt"))
        all_seed_results["multimodal_concat_mlp"][f"seed_{seed}"] = c_met
        with open(os.path.join(pred_dir, f"multimodal_concat_mlp_seed{seed}.json"), "w", encoding="utf-8") as f:
            json.dump({"sample_ids": val_sids, "video_ids": val_vids, "y_true": y_val.numpy().tolist(),
                       "y_pred": c_pred.tolist(), "probabilities": c_prob.tolist()}, f)

        # 7. Multimodal Fusion Model
        f_model, f_met, f_pred, f_prob, f_time, f_vram = train_multimodal_fusion(
            data["train"], data["val"], class_weights, device, seed, modality_dropout=0.0
        )
        torch.save(f_model.state_dict(), os.path.join(ckpt_dir, f"multimodal_fusion_seed{seed}.pt"))
        all_seed_results["multimodal_fusion"][f"seed_{seed}"] = f_met
        with open(os.path.join(pred_dir, f"multimodal_fusion_seed{seed}.json"), "w", encoding="utf-8") as f:
            json.dump({"sample_ids": val_sids, "video_ids": val_vids, "y_true": y_val.numpy().tolist(),
                       "y_pred": f_pred.tolist(), "probabilities": f_prob.tolist()}, f)

        # 8. Multimodal Fusion Missingness-Aware
        fm_model, fm_met, fm_pred, fm_prob, fm_time, fm_vram = train_multimodal_fusion(
            data["train"], data["val"], class_weights, device, seed, modality_dropout=0.20
        )
        torch.save(fm_model.state_dict(), os.path.join(ckpt_dir, f"multimodal_fusion_missingness_aware_seed{seed}.pt"))
        all_seed_results["multimodal_fusion_missingness_aware"][f"seed_{seed}"] = fm_met
        with open(os.path.join(pred_dir, f"multimodal_fusion_missingness_aware_seed{seed}.json"), "w", encoding="utf-8") as f:
            json.dump({"sample_ids": val_sids, "video_ids": val_vids, "y_true": y_val.numpy().tolist(),
                       "y_pred": fm_pred.tolist(), "probabilities": fm_prob.tolist()}, f)

    # Compute paired bootstrap statistics for each model family against reference Text-only MLP (Seed 42)
    summary_stats = {
        "majority_class": {
            "metrics": maj_metrics,
            "bootstrap": video_grouped_block_bootstrap_paired(y_val.numpy(), preds_maj, ref_preds_text_mlp, val_vids),
        },
        "text_only_lr": {
            "metrics": text_lr_metrics,
            "bootstrap": video_grouped_block_bootstrap_paired(y_val.numpy(), preds_text_lr, ref_preds_text_mlp, val_vids),
        },
        "multimodal_concat_lr": {
            "metrics": concat_lr_metrics,
            "bootstrap": video_grouped_block_bootstrap_paired(y_val.numpy(), preds_concat_lr, ref_preds_text_mlp, val_vids),
        },
    }

    for model_name, seed_dict in all_seed_results.items():
        accs = [seed_dict[f"seed_{s}"]["accuracy"] for s in SEEDS]
        f1s = [seed_dict[f"seed_{s}"]["macro_f1"] for s in SEEDS]
        eces = [seed_dict[f"seed_{s}"]["calibration"]["ece"] for s in SEEDS]

        # Use Seed 42 prediction file for grouped paired bootstrap
        with open(os.path.join(pred_dir, f"{model_name}_seed42.json"), "r") as f:
            p_data = json.load(f)
            preds_seed42 = np.array(p_data["y_pred"])

        boot = video_grouped_block_bootstrap_paired(y_val.numpy(), preds_seed42, ref_preds_text_mlp, val_vids)

        summary_stats[model_name] = {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_std": float(np.std(f1s)),
            "ece_mean": float(np.mean(eces)),
            "per_seed_accuracy": accs,
            "per_seed_macro_f1": f1s,
            "bootstrap": boot,
        }

    summary_file = os.path.join(out_dir, "training", "summary_statistics.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2)

    logger.info(f"Summary statistics saved to {summary_file}")
    logger.info("=== SUMMARY OF PREDICTOR BANK (VAL SET, 3-SEED MEAN) ===")
    for k, v in summary_stats.items():
        acc = v.get("accuracy_mean", v.get("metrics", {}).get("accuracy", 0.0))
        f1 = v.get("macro_f1_mean", v.get("metrics", {}).get("macro_f1", 0.0))
        logger.info(f"{k}: Acc={acc:.4f}, Macro-F1={f1:.4f}")


if __name__ == "__main__":
    main()
