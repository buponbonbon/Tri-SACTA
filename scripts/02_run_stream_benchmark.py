"""
02_run_stream_benchmark.py: Continual Compound Shift Stream Evaluation.

Evaluates test-time adaptation and conformal prediction across 5 sequential compound shift regimes:
- Regime 1: Clean baseline (0% missing).
- Regime 2: Audio degradation (Gaussian noise / SNR 5dB -> 0dB) + Visual clean.
- Regime 3: ASR Text drift (simulated WER 15% -> 35% perturbation).
- Regime 4: Visual corruption (blur & occlusion) + 30% random missing modality.
- Regime 5: Severe compound shift (multi-channel noise + 50% missing modality) + return to clean.

Supported TTA methods:
- tri_sacta: Tri-Modal Spectral-Aligned Continual TTA (Ours)
- tent: Test Entropy Minimization (LayerNorm adaptation)
- sar: Sharpness-Aware Reliable TTA
- read: Multimodal TTA baseline (READ / AV-CTTA style)
- source_only: No adaptation

Supported Conformal Methods:
- online_aci: Mask-and-Disagreement-Driven Online ACI (Ours)
- static_cp: Split Conformal Prediction
- mccf: Mask-Conditional Static Conformal Prediction
- none: No conformal prediction
"""

from __future__ import annotations

import os
import sys
import argparse
import logging
import json
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import get_dataloaders
from src.data.shift_generator import ShiftStreamGenerator
from src.models.fusion_backbone import MultimodalFusionModel
from src.models.spectral_detector import SpectralShiftDetector
from src.adaptation.tri_sacta import TriSACTA
from src.adaptation.baselines_tta import TENT, SAR, MultimodalTTABaseline
from src.conformal.online_aci import OnlineACI
from src.conformal.static_cp import SplitConformalPredictor, MaskConditionalCP
from src.utils.metrics import MetricsTracker, VRAMProfiler, LatencyProfiler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODALITY_NAMES = ["visual", "audio", "text", "prosody"]


def get_mask_key(modality_mask: Dict[str, torch.Tensor], sample_idx: int) -> str:
    """Generate binary mask string '1110' for the sample."""
    return "".join(["1" if modality_mask[m][sample_idx].item() else "0" for m in MODALITY_NAMES])


def main():
    parser = argparse.ArgumentParser(description="Run Continual Compound Shift Benchmark")
    parser.add_argument("--config", type=str, default="configs/mosei_mvp.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--spectral_stats", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", type=str, default="tri_sacta", choices=["tri_sacta", "tent", "sar", "read", "source_only"])
    parser.add_argument("--conformal", type=str, default="online_aci", choices=["online_aci", "static_cp", "mccf", "none"])
    parser.add_argument("--delay", type=int, default=0, help="Delayed feedback steps (0, 10, 50)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    vram = VRAMProfiler(device)
    latency = LatencyProfiler()

    # 1. Dataloaders
    batch_size = config.get("tta", {}).get("batch_size", 32)
    dataloaders = get_dataloaders(config, batch_size=batch_size, seed=args.seed)
    test_loader = dataloaders["test"]
    cal_loader = dataloaders["cal"]

    # 2. Model setup
    model = MultimodalFusionModel(config).to(device)
    num_classes = config.get("model", {}).get("num_classes", 7)
    target_alpha = config.get("conformal", {}).get("alpha", 0.10)

    # Load checkpoint if provided
    ckpt_path = args.checkpoint or f"checkpoints/source_model_seed{args.seed}.pt"
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded source checkpoint from {ckpt_path}")
    else:
        logger.warning(f"Checkpoint not found at {ckpt_path}. Using current model weights.")

    # 3. Spectral detector setup
    detector = None
    if args.method == "tri_sacta" or args.conformal == "online_aci":
        feats_cfg = config.get("features", {})
        feature_dims = {
            "visual": feats_cfg.get("visual_dim", 512),
            "audio": feats_cfg.get("audio_dim", 768),
            "text": feats_cfg.get("text_dim", 768),
            "prosody": feats_cfg.get("prosody_dim", 88),
        }
        detector = SpectralShiftDetector(
            feature_dims=feature_dims,
            gamma=config.get("tta", {}).get("gamma_eig", 1.0),
            top_k=config.get("tta", {}).get("top_k_eig", 32),
        )
        stats_path = args.spectral_stats or f"checkpoints/source_spectral_stats_seed{args.seed}.pt"
        if os.path.exists(stats_path):
            detector.load_source_stats(stats_path)
            logger.info(f"Loaded source anchor statistics from {stats_path}")
        else:
            logger.info("Computing on-the-fly source statistics from training split...")
            train_feats = {m: [] for m in MODALITY_NAMES}
            for batch in dataloaders["train"]:
                for m in MODALITY_NAMES:
                    if m in batch["features"]:
                        train_feats[m].append(batch["features"][m].cpu())
            stacked = {m: torch.cat(tensors, dim=0) for m, tensors in train_feats.items() if tensors}
            detector.register_source_stats(stacked)

    # 4. Initialize TTA Method
    tta_method = None
    if args.method == "tri_sacta":
        tta_method = TriSACTA(model, detector, config)
    elif args.method == "tent":
        tta_method = TENT(model, config)
    elif args.method == "sar":
        tta_method = SAR(model, config)
    elif args.method == "read":
        tta_method = MultimodalTTABaseline(model, config)
    elif args.method == "source_only":
        model.eval()

    # 5. Initialize Conformal Predictor
    conformal_predictor = None
    if args.conformal == "online_aci":
        conf_cfg = copy_cfg = dict(config.get("conformal", {}))
        conf_cfg["delay"] = args.delay
        conformal_predictor = OnlineACI(num_classes=num_classes, alpha=target_alpha, config=conf_cfg)
    elif args.conformal == "static_cp":
        conformal_predictor = SplitConformalPredictor(num_classes=num_classes, alpha=target_alpha)
        # Calibrate on cal set
        cal_logits, cal_labels = [], []
        model.eval()
        with torch.no_grad():
            for batch in cal_loader:
                feats = {k: v.to(device) for k, v in batch["features"].items()}
                mask = {k: v.to(device) for k, v in batch["modality_mask"].items()}
                out = model(feats, modality_mask=mask)
                cal_logits.append(out["logits"].cpu())
                cal_labels.append(batch["labels"])
        conformal_predictor.calibrate(torch.cat(cal_logits, dim=0), torch.cat(cal_labels, dim=0))
    elif args.conformal == "mccf":
        conformal_predictor = MaskConditionalCP(num_classes=num_classes, alpha=target_alpha)
        cal_logits, cal_labels, cal_masks = [], [], []
        model.eval()
        with torch.no_grad():
            for batch in cal_loader:
                feats = {k: v.to(device) for k, v in batch["features"].items()}
                mask = {k: v.to(device) for k, v in batch["modality_mask"].items()}
                out = model(feats, modality_mask=mask)
                cal_logits.append(out["logits"].cpu())
                cal_labels.append(batch["labels"])
                for i in range(len(batch["labels"])):
                    cal_masks.append(get_mask_key(batch["modality_mask"], i))
        conformal_predictor.calibrate(torch.cat(cal_logits, dim=0), torch.cat(cal_labels, dim=0), cal_masks)

    # 6. Stream benchmark through 5 sequential regimes
    shift_gen = ShiftStreamGenerator(config, seed=args.seed)
    regimes = shift_gen.get_regime_sequence()

    overall_tracker = MetricsTracker(num_classes=num_classes, target_coverage=(1.0 - target_alpha))
    regime_results = {}
    time_series_data = {
        "shift_velocity": [],
        "cross_modal_disagreement": [],
        "alpha_t": [],
        "rolling_coverage": [],
        "prediction_set_sizes": [],
    }

    logger.info(f"Starting Continual Compound Shift Stream benchmark ({args.method} + {args.conformal})...")
    vram.reset_peak()

    for regime_info in regimes:
        reg_idx = regime_info["regime_idx"]
        reg_name = regime_info["name"]
        reg_tracker = MetricsTracker(num_classes=num_classes, target_coverage=(1.0 - target_alpha))
        logger.info(f"Entering {reg_name}...")

        total_batches = len(test_loader)
        for step, raw_batch in enumerate(test_loader):
            # Apply deterministic continual shift for current regime
            batch = shift_gen.apply_regime(raw_batch, regime_idx=reg_idx, step_in_regime=step, total_steps_in_regime=total_batches)

            features = {k: v.to(device) for k, v in batch["features"].items()}
            mod_mask = {k: v.to(device) for k, v in batch["modality_mask"].items()}
            labels = batch["labels"].to(device)
            bsz = labels.size(0)

            latency.start()

            # Adaptation / Inference step
            if tta_method is not None:
                step_res = tta_method.adapt_step(features, mod_mask)
                logits = step_res["logits"]
                shift_vel = tta_method.get_shift_velocity()
                avg_shift_vel = float(np.mean(list(shift_vel.values()))) if shift_vel else 0.0
                disagree = step_res.get("cross_modal_disagreement", 0.0)
                if isinstance(disagree, torch.Tensor):
                    disagree = float(disagree.item())
            else:
                # Source-only forward pass
                with torch.no_grad():
                    out = model(features, modality_mask=mod_mask)
                    logits = out["logits"]
                avg_shift_vel = 0.0
                disagree = 0.0

            # Conformal Prediction Step
            pred_sets = None
            if conformal_predictor is not None:
                pred_sets = []
                for i in range(bsz):
                    m_key = get_mask_key(batch["modality_mask"], i)
                    sample_logits = logits[i : i + 1]

                    if args.conformal == "online_aci":
                        p_set = conformal_predictor.predict_set(sample_logits, m_key)[0]
                        pred_sets.append(p_set)
                        # Online update with delayed buffer support
                        conformal_predictor.update(
                            sample_logits,
                            labels[i : i + 1],
                            m_key,
                            shift_velocity=avg_shift_vel,
                            disagreement=disagree,
                        )
                        time_series_data["alpha_t"].append(conformal_predictor.alphas[m_key])
                    elif args.conformal == "static_cp":
                        p_set = conformal_predictor.predict_set(sample_logits)[0]
                        pred_sets.append(p_set)
                    elif args.conformal == "mccf":
                        p_set = conformal_predictor.predict_set(sample_logits, m_key)[0]
                        pred_sets.append(p_set)

            latency_ms = latency.stop(num_samples=bsz)

            # Update trackers
            reg_tracker.update(logits, labels, pred_sets)
            overall_tracker.update(logits, labels, pred_sets)

            time_series_data["shift_velocity"].append(avg_shift_vel)
            time_series_data["cross_modal_disagreement"].append(disagree)
            if pred_sets:
                time_series_data["prediction_set_sizes"].extend([len(s) for s in pred_sets])

        reg_metrics = reg_tracker.compute()
        regime_results[f"regime_{reg_idx}"] = {
            "name": reg_name,
            "metrics": reg_metrics,
        }
        logger.info(
            f"Completed {reg_name}: Acc={reg_metrics['accuracy']:.4f}, Macro-F1={reg_metrics['macro_f1']:.4f}, "
            f"Coverage={reg_metrics.get('empirical_coverage', 0.0):.4f}, Avg Set Size={reg_metrics.get('avg_set_size', 0.0):.2f}"
        )

    # 7. Aggregate overall metrics
    overall_metrics = overall_tracker.compute()
    vram_stats = vram.snapshot()
    avg_latency = latency.average_latency_per_sample()

    time_series_data["rolling_coverage"] = overall_tracker.coverage_binary_stream

    final_results = {
        "dataset": config.get("dataset", {}).get("name", "mosei"),
        "method": args.method,
        "conformal": args.conformal,
        "seed": args.seed,
        "delay": args.delay,
        "overall_metrics": overall_metrics,
        "regime_metrics": regime_results,
        "efficiency": {
            "peak_vram_mb": vram_stats["peak_mb"],
            "avg_latency_ms_per_sample": avg_latency,
        },
        "time_series": time_series_data,
    }

    out_file = os.path.join(args.output_dir, f"{args.method}_{args.conformal}_{config['dataset']['name']}_seed{args.seed}.json")
    with open(out_file, "w") as f:
        json.dump(final_results, f, indent=2)

    logger.info(f"Results successfully saved to {out_file}")
    logger.info(
        f"Overall: Macro-F1={overall_metrics['macro_f1']:.4f}, Coverage={overall_metrics.get('empirical_coverage', 0.0):.4f}, "
        f"Coverage Gap={overall_metrics.get('coverage_gap', 0.0):.4f}, Set Size={overall_metrics.get('avg_set_size', 0.0):.2f}, "
        f"Peak VRAM={vram_stats['peak_mb']:.1f} MB, Latency={avg_latency:.2f} ms/sample"
    )


if __name__ == "__main__":
    main()
