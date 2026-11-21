"""
01_train_source.py: Source Model Training & Spectral Anchor Statistics Extraction.

Steps:
1. Load configuration and initialize deterministic seed.
2. Build lightweight MultimodalFusionModel (<2M parameters).
3. Train with Cross-Entropy loss and PyTorch AMP (FP16) on speaker-disjoint source data.
4. Early stopping on validation split Macro-F1.
5. Compute and freeze source anchor statistics (mean & covariance eigenvalues) via SpectralShiftDetector.
6. Calibrate static conformal baseline on validation/calibration split.
7. Save source checkpoints & spectral anchor stats for streaming test-time adaptation.
"""

from __future__ import annotations

import os
import sys
import argparse
import logging
import yaml
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import get_dataloaders
from src.models.fusion_backbone import MultimodalFusionModel
from src.models.spectral_detector import SpectralShiftDetector
from src.conformal.static_cp import SplitConformalPredictor
from src.utils.metrics import MetricsTracker, VRAMProfiler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, optimizer, scaler, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        features = {k: v.to(device) for k, v in batch["features"].items()}
        mask = {k: v.to(device) for k, v in batch["modality_mask"].items()}
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        with autocast(enabled=torch.cuda.is_available()):
            outputs = model(features, modality_mask=mask)
            loss = criterion(outputs["logits"], labels)

        if torch.cuda.is_available():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, num_classes, device):
    model.eval()
    tracker = MetricsTracker(num_classes=num_classes)
    for batch in loader:
        features = {k: v.to(device) for k, v in batch["features"].items()}
        mask = {k: v.to(device) for k, v in batch["modality_mask"].items()}
        labels = batch["labels"].to(device)

        with autocast(enabled=torch.cuda.is_available()):
            outputs = model(features, modality_mask=mask)

        tracker.update(outputs["logits"], labels)

    return tracker.compute()


def main():
    parser = argparse.ArgumentParser(description="Train Source Model for Tri-SACTA")
    parser.add_argument("--config", type=str, default="configs/mosei_mvp.yaml", help="Path to config YAML")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    vram = VRAMProfiler(device)
    logger.info(f"Using device: {device} | Random seed: {args.seed}")

    # 1. Dataloaders
    batch_size = config.get("training", {}).get("batch_size", 32)
    dataloaders = get_dataloaders(config, batch_size=batch_size, seed=args.seed)

    # 2. Build model
    model = MultimodalFusionModel(config).to(device)
    total_params = model.get_total_params()
    logger.info(f"Model parameters: {total_params}")

    # 3. Optimizer & Scaler
    lr = float(config.get("training", {}).get("lr", 1e-3))
    epochs = args.epochs or config.get("training", {}).get("epochs", 30)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    criterion = nn.CrossEntropyLoss()

    num_classes = config.get("model", {}).get("num_classes", 7)
    best_val_f1 = -1.0
    best_weights = None
    patience = config.get("training", {}).get("patience", 10)
    patience_cnt = 0

    logger.info("Starting source model training...")
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, dataloaders["train"], optimizer, scaler, criterion, device)
        val_metrics = evaluate(model, dataloaders["val"], num_classes, device)
        scheduler.step()

        val_f1 = val_metrics["macro_f1"]
        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | Val Macro-F1: {val_f1:.4f} | Peak VRAM: {vram.snapshot()['peak_mb']:.1f} MB"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = model.state_dict()
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

    # Load best weights
    model.load_state_dict(best_weights)
    ckpt_path = os.path.join(args.output_dir, f"source_model_seed{args.seed}.pt")
    torch.save(best_weights, ckpt_path)
    logger.info(f"Saved best source model to {ckpt_path}")

    # 4. Extract and register source anchor statistics via SpectralShiftDetector
    logger.info("Computing source anchor statistics on training set...")
    model.eval()
    all_train_features = {m: [] for m in ["visual", "audio", "text", "prosody"]}
    with torch.no_grad():
        for batch in dataloaders["train"]:
            for m in all_train_features:
                if m in batch["features"]:
                    all_train_features[m].append(batch["features"][m].cpu())

    stacked_features = {m: torch.cat(tensors, dim=0) for m, tensors in all_train_features.items() if tensors}
    feature_dims = {m: stacked_features[m].shape[1] for m in stacked_features}
    
    detector = SpectralShiftDetector(
        feature_dims=feature_dims,
        gamma=config.get("tta", {}).get("gamma_eig", 1.0),
        top_k=config.get("tta", {}).get("top_k_eig", 32),
    )
    detector.register_source_stats(stacked_features)

    stats_path = os.path.join(args.output_dir, f"source_spectral_stats_seed{args.seed}.pt")
    detector.save_source_stats(stats_path)
    logger.info(f"Saved spectral anchor statistics to {stats_path}")

    # 5. Check Conformal Calibration baseline
    cal_metrics = evaluate(model, dataloaders["cal"], num_classes, device)
    logger.info(f"Calibration split performance: Acc={cal_metrics['accuracy']:.4f}, Macro-F1={cal_metrics['macro_f1']:.4f}")

    logger.info("Source training and anchor registration complete.")


if __name__ == "__main__":
    main()
