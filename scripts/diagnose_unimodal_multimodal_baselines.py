"""
diagnose_unimodal_multimodal_baselines.py: Comprehensive unimodal and multimodal baseline evaluation (V7-01, V7-02).

Evaluates on identical authentic CMU-MOSEI splits:
1. Unimodal Text (768d): Logistic Regression (unweighted & balanced), 2-layer MLP
2. Unimodal Audio (74d): Logistic Regression (unweighted & balanced), 2-layer MLP
3. Unimodal Visual (35d): Logistic Regression (unweighted & balanced), 2-layer MLP
4. Early Fusion / Concatenation (877d): Logistic Regression (unweighted & balanced), 2-layer MLP
5. Late Gated Fusion (learnable gating between unimodal heads)
6. Multimodal Transformer Fusion Backbone (Tri-SACTA backbone)

Saves full comparative metrics, per-class F1, and confusion matrices.
"""

from __future__ import annotations

import os
import sys
import json
import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.fusion_backbone import MultimodalFusionModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 7) -> Dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    per_class_f1 = [float(x) for x in f1_score(y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0)]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist()
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
    }


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, num_classes: int = 7, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedFusionModel(nn.Module):
    """Gated multimodal fusion combining unimodal representations via soft gating."""
    def __init__(self, dims: Dict[str, int], fusion_dim: int = 128, num_classes: int = 7, dropout: float = 0.2):
        super().__init__()
        self.projs = nn.ModuleDict({
            m: nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for m, d in dims.items()
        })
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * len(dims), len(dims)),
            nn.Softmax(dim=-1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        h = [self.projs[m](feats[m]) for m in sorted(feats.keys())]
        h_cat = torch.cat(h, dim=-1)  # (B, M * D)
        gates = self.gate(h_cat)  # (B, M)
        h_stack = torch.stack(h, dim=1)  # (B, M, D)
        fused = (h_stack * gates.unsqueeze(-1)).sum(dim=1)  # (B, D)
        return self.classifier(fused)


def train_torch_model(
    model: nn.Module,
    X_train: torch.Tensor | Dict[str, torch.Tensor],
    y_train: torch.Tensor,
    X_val: torch.Tensor | Dict[str, torch.Tensor],
    y_val: torch.Tensor,
    num_classes: int = 7,
    epochs: int = 40,
    lr: float = 1e-3,
    class_weights: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> nn.Module:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    if class_weights:
        counts = np.bincount(y_train.cpu().numpy(), minlength=num_classes)
        counts = np.maximum(counts, 1)
        w = len(y_train) / (num_classes * counts)
        w = w / w.sum() * num_classes
        crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    else:
        crit = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_state = None

    is_dict = isinstance(X_train, dict)
    N = len(y_train)
    batch_size = 32

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(N)
        for b_start in range(0, N, batch_size):
            b_idx = perm[b_start : b_start + batch_size]
            if is_dict:
                b_x = {m: X_train[m][b_idx].to(device) for m in X_train}
            else:
                b_x = X_train[b_idx].to(device)
            b_y = y_train[b_idx].to(device)

            optimizer.zero_grad()
            logits = model(b_x)
            loss = crit(logits, b_y)
            loss.backward()
            optimizer.step()

        # Val eval
        model.eval()
        with torch.no_grad():
            if is_dict:
                v_x = {m: X_val[m].to(device) for m in X_val}
            else:
                v_x = X_val.to(device)
            v_logits = model(v_x).cpu()
            v_pred = v_logits.argmax(dim=-1).numpy()
            v_f1 = float(f1_score(y_val.numpy(), v_pred, average="macro", zero_division=0))

        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            import copy
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main():
    data_dir = "data/cmu_mosei/features_cache"
    output_dir = "results/diagnostics"
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading authentic CMU-MOSEI features from {data_dir}...")
    splits = {}
    for s in ["train", "val", "cal", "test"]:
        s_dir = os.path.join(data_dir, s)
        splits[s] = {
            "text": torch.load(os.path.join(s_dir, "text.pt"), map_location="cpu").float(),
            "visual": torch.load(os.path.join(s_dir, "visual.pt"), map_location="cpu").float(),
            "audio": torch.load(os.path.join(s_dir, "audio.pt"), map_location="cpu").float(),
            "labels": torch.load(os.path.join(s_dir, "labels.pt"), map_location="cpu").long(),
        }

    y_train = splits["train"]["labels"]
    y_val = splits["val"]["labels"]
    y_test = splits["test"]["labels"]
    y_test_np = y_test.numpy()

    results = {}

    # 1. Majority-Class Baseline
    maj_class = int(np.bincount(y_train.numpy()).argmax())
    y_pred_maj = np.full_like(y_test_np, maj_class)
    results["majority_class"] = compute_metrics(y_test_np, y_pred_maj)
    logger.info(f"Majority-Class Predictor ({maj_class}): Acc={results['majority_class']['accuracy']*100:.2f}%, Macro-F1={results['majority_class']['macro_f1']:.4f}")

    # 2. Unimodal Baselines
    for m in ["text", "audio", "visual"]:
        X_tr = splits["train"][m]
        X_v = splits["val"][m]
        X_te = splits["test"][m]
        dim = X_tr.shape[-1]

        # 2a. Logistic Regression Unweighted
        lr_unw = LogisticRegression(max_iter=1000, random_state=42)
        lr_unw.fit(X_tr.numpy(), y_train.numpy())
        y_pred_lr_unw = lr_unw.predict(X_te.numpy())
        results[f"lr_unweighted_{m}"] = compute_metrics(y_test_np, y_pred_lr_unw)

        # 2b. Logistic Regression Balanced
        lr_bal = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        lr_bal.fit(X_tr.numpy(), y_train.numpy())
        y_pred_lr_bal = lr_bal.predict(X_te.numpy())
        results[f"lr_balanced_{m}"] = compute_metrics(y_test_np, y_pred_lr_bal)

        # 2c. 2-layer MLP (Balanced)
        mlp = MLPClassifier(in_dim=dim, hidden_dim=128)
        mlp = train_torch_model(mlp, X_tr, y_train, X_v, y_val, class_weights=True, device=device)
        mlp.eval()
        with torch.no_grad():
            y_pred_mlp = mlp(X_te.to(device)).argmax(dim=-1).cpu().numpy()
        results[f"mlp_balanced_{m}"] = compute_metrics(y_test_np, y_pred_mlp)

        logger.info(
            f"Unimodal [{m.upper()}]: LR_unw={results[f'lr_unweighted_{m}']['macro_f1']:.4f}, "
            f"LR_bal={results[f'lr_balanced_{m}']['macro_f1']:.4f}, "
            f"MLP_bal={results[f'mlp_balanced_{m}']['macro_f1']:.4f}"
        )

    # 3. Concatenation Early Fusion (Text + Audio + Visual = 877d)
    X_cat_tr = torch.cat([splits["train"]["text"], splits["train"]["audio"], splits["train"]["visual"]], dim=-1)
    X_cat_v = torch.cat([splits["val"]["text"], splits["val"]["audio"], splits["val"]["visual"]], dim=-1)
    X_cat_te = torch.cat([splits["test"]["text"], splits["test"]["audio"], splits["test"]["visual"]], dim=-1)
    cat_dim = X_cat_tr.shape[-1]

    lr_cat_unw = LogisticRegression(max_iter=1000, random_state=42)
    lr_cat_unw.fit(X_cat_tr.numpy(), y_train.numpy())
    results["lr_unweighted_concat"] = compute_metrics(y_test_np, lr_cat_unw.predict(X_cat_te.numpy()))

    lr_cat_bal = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr_cat_bal.fit(X_cat_tr.numpy(), y_train.numpy())
    results["lr_balanced_concat"] = compute_metrics(y_test_np, lr_cat_bal.predict(X_cat_te.numpy()))

    mlp_cat = MLPClassifier(in_dim=cat_dim, hidden_dim=256)
    mlp_cat = train_torch_model(mlp_cat, X_cat_tr, y_train, X_cat_v, y_val, class_weights=True, device=device)
    mlp_cat.eval()
    with torch.no_grad():
        results["mlp_balanced_concat"] = compute_metrics(y_test_np, mlp_cat(X_cat_te.to(device)).argmax(dim=-1).cpu().numpy())

    logger.info(
        f"Early Fusion (Concat): LR_unw={results['lr_unweighted_concat']['macro_f1']:.4f}, "
        f"LR_bal={results['lr_balanced_concat']['macro_f1']:.4f}, "
        f"MLP_bal={results['mlp_balanced_concat']['macro_f1']:.4f}"
    )

    # 4. Gated Late Fusion
    gated_model = GatedFusionModel(dims={"text": 768, "audio": 74, "visual": 35}, fusion_dim=128)
    dict_tr = {m: splits["train"][m] for m in ["text", "audio", "visual"]}
    dict_v = {m: splits["val"][m] for m in ["text", "audio", "visual"]}
    dict_te = {m: splits["test"][m] for m in ["text", "audio", "visual"]}
    gated_model = train_torch_model(gated_model, dict_tr, y_train, dict_v, y_val, class_weights=True, device=device)
    gated_model.eval()
    with torch.no_grad():
        te_x_dev = {m: dict_te[m].to(device) for m in dict_te}
        results["gated_fusion_balanced"] = compute_metrics(y_test_np, gated_model(te_x_dev).argmax(dim=-1).cpu().numpy())
    logger.info(f"Gated Fusion: Macro-F1={results['gated_fusion_balanced']['macro_f1']:.4f}, Acc={results['gated_fusion_balanced']['accuracy']*100:.2f}%")

    # 5. Multimodal Transformer Backbone (Existing Checkpoint)
    ckpt_path = "checkpoints/cmu_mosei/real/seed42/source_model.pt"
    if os.path.exists(ckpt_path):
        raw_obj = torch.load(ckpt_path, map_location=device)
        model = MultimodalFusionModel(raw_obj["config"]).to(device)
        model.load_state_dict(raw_obj["model_state_dict"])
        model.eval()
        with torch.no_grad():
            feats = {
                "text": splits["test"]["text"].to(device),
                "audio": splits["test"]["audio"].to(device),
                "visual": splits["test"]["visual"].to(device),
                "prosody": torch.zeros(len(y_test), 88, device=device),
            }
            mask = {
                "text": torch.ones(len(y_test), dtype=torch.bool, device=device),
                "audio": torch.ones(len(y_test), dtype=torch.bool, device=device),
                "visual": torch.ones(len(y_test), dtype=torch.bool, device=device),
                "prosody": torch.zeros(len(y_test), dtype=torch.bool, device=device),
            }
            out = model(feats, modality_mask=mask)
            results["multimodal_transformer_backbone"] = compute_metrics(y_test_np, out["logits"].argmax(dim=-1).cpu().numpy())
        logger.info(
            f"Existing Multimodal Transformer: Macro-F1={results['multimodal_transformer_backbone']['macro_f1']:.4f}, "
            f"Acc={results['multimodal_transformer_backbone']['accuracy']*100:.2f}%"
        )

    # Save detailed JSON report
    out_json = os.path.join(output_dir, "unimodal_multimodal_baselines_comparison.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    # Save markdown summary
    out_md = os.path.join(output_dir, "unimodal_multimodal_baselines_comparison.md")
    with open(out_md, "w") as f:
        f.write("# Unimodal and Multimodal Baselines Comparative Evaluation\n\n")
        f.write("**Dataset:** CMU-MOSEI Authentic (N_train=1000, N_val=250, N_test=500)\n\n")
        f.write("| Architecture / Baseline | Modality Scope | Accuracy | Macro-F1 | Weighted-F1 |\n")
        f.write("|---|---|---|---|---|\n")
        f.write(f"| Majority-Class Predictor | None | {results['majority_class']['accuracy']*100:.2f}% | {results['majority_class']['macro_f1']:.4f} | {results['majority_class']['weighted_f1']:.4f} |\n")
        f.write(f"| Logistic Regression (Unweighted) | Text-only (768d) | {results['lr_unweighted_text']['accuracy']*100:.2f}% | {results['lr_unweighted_text']['macro_f1']:.4f} | {results['lr_unweighted_text']['weighted_f1']:.4f} |\n")
        f.write(f"| **Logistic Regression (Balanced)** | **Text-only (768d)** | **{results['lr_balanced_text']['accuracy']*100:.2f}%** | **{results['lr_balanced_text']['macro_f1']:.4f}** | **{results['lr_balanced_text']['weighted_f1']:.4f}** |\n")
        f.write(f"| 2-Layer MLP (Balanced) | Text-only (768d) | {results['mlp_balanced_text']['accuracy']*100:.2f}% | {results['mlp_balanced_text']['macro_f1']:.4f} | {results['mlp_balanced_text']['weighted_f1']:.4f} |\n")
        f.write(f"| Logistic Regression (Balanced) | Audio-only (74d) | {results['lr_balanced_audio']['accuracy']*100:.2f}% | {results['lr_balanced_audio']['macro_f1']:.4f} | {results['lr_balanced_audio']['weighted_f1']:.4f} |\n")
        f.write(f"| 2-Layer MLP (Balanced) | Audio-only (74d) | {results['mlp_balanced_audio']['accuracy']*100:.2f}% | {results['mlp_balanced_audio']['macro_f1']:.4f} | {results['mlp_balanced_audio']['weighted_f1']:.4f} |\n")
        f.write(f"| Logistic Regression (Balanced) | Visual-only (35d) | {results['lr_balanced_visual']['accuracy']*100:.2f}% | {results['lr_balanced_visual']['macro_f1']:.4f} | {results['lr_balanced_visual']['weighted_f1']:.4f} |\n")
        f.write(f"| 2-Layer MLP (Balanced) | Visual-only (35d) | {results['mlp_balanced_visual']['accuracy']*100:.2f}% | {results['mlp_balanced_visual']['macro_f1']:.4f} | {results['mlp_balanced_visual']['weighted_f1']:.4f} |\n")
        f.write(f"| **Logistic Regression (Balanced)** | **Concat Early Fusion (877d)** | **{results['lr_balanced_concat']['accuracy']*100:.2f}%** | **{results['lr_balanced_concat']['macro_f1']:.4f}** | **{results['lr_balanced_concat']['weighted_f1']:.4f}** |\n")
        f.write(f"| 2-Layer MLP (Balanced) | Concat Early Fusion (877d) | {results['mlp_balanced_concat']['accuracy']*100:.2f}% | {results['mlp_balanced_concat']['macro_f1']:.4f} | {results['mlp_balanced_concat']['weighted_f1']:.4f} |\n")
        f.write(f"| **Gated Late Fusion (Balanced)** | **Multimodal (T+A+V)** | **{results['gated_fusion_balanced']['accuracy']*100:.2f}%** | **{results['gated_fusion_balanced']['macro_f1']:.4f}** | **{results['gated_fusion_balanced']['weighted_f1']:.4f}** |\n")
        if "multimodal_transformer_backbone" in results:
            f.write(f"| Multimodal Transformer (V6 Checkpoint) | Multimodal (T+A+V) | {results['multimodal_transformer_backbone']['accuracy']*100:.2f}% | {results['multimodal_transformer_backbone']['macro_f1']:.4f} | {results['multimodal_transformer_backbone']['weighted_f1']:.4f} |\n")

    logger.info(f"Baseline diagnosis complete. Report saved to {out_md}")


if __name__ == "__main__":
    main()
