"""
02_run_stream_benchmark.py: Continual Compound Shift Stream Evaluation.

Evaluates test-time adaptation and conformal prediction across 5 sequential compound shift regimes:
- Regime 1: Clean baseline (0% missing).
- Regime 2: Audio degradation (Gaussian noise / SNR 5dB -> 0dB) + Visual clean.
- Regime 3: ASR Text drift (simulated WER 15% -> 35% perturbation).
- Regime 4: Visual corruption (blur & occlusion) + 30% random missing modality.
- Regime 5: Severe compound shift (multi-channel noise + 50% missing modality) + return to clean.

Key scientific improvements:
1. Strict fail-closed checkpoint/data verification (AUD-10).
2. Safe baseline shift-velocity check (AUD-15).
3. Cross-modal disagreement passed to Online ACI (AUD-07).
4. Delayed feedback with immutable prediction sets (AUD-08).
5. Explicit data_mode and delay in output filenames (AUD-19, AUD-24).
"""

from __future__ import annotations

import os
import sys
import copy
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
from src.utils.naming import get_checkpoint_path, get_spectral_stats_path, get_run_output_path
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
    parser.add_argument("--protocol", type=str, default="prequential", choices=["prequential", "transductive"], help="Stream protocol (R3-06)")
    parser.add_argument("--data-mode", type=str, default="synthetic", choices=["synthetic", "real"], help="Data mode (AUD-10)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    dataset_name = config.get("dataset", {}).get("name", "mosei")

    # Deterministic seeding
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    vram = VRAMProfiler(device)
    latency = LatencyProfiler()

    # 1. Dataloaders
    batch_size = config.get("tta", {}).get("batch_size", 32)
    dataloaders = get_dataloaders(config, batch_size=batch_size, seed=args.seed, data_mode=args.data_mode)
    test_loader = dataloaders["test"]
    cal_loader = dataloaders["cal"]

    # 2. Model setup
    model = MultimodalFusionModel(config).to(device)
    num_classes = config.get("model", {}).get("num_classes", 7)
    target_alpha = config.get("conformal", {}).get("alpha", 0.10)

    # Resolve namespaced checkpoint (R3-05)
    ckpt_path = args.checkpoint
    if not ckpt_path:
        candidate_paths = [
            get_checkpoint_path("checkpoints", dataset_name, args.data_mode, args.seed),
            f"checkpoints/source_model_{args.data_mode}_seed{args.seed}.pt",
            f"checkpoints/source_model_seed{args.seed}.pt",
        ]
        for cp in candidate_paths:
            if os.path.exists(cp):
                ckpt_path = cp
                break

    if ckpt_path and os.path.exists(ckpt_path):
        raw_obj = torch.load(ckpt_path, map_location=device)
        if isinstance(raw_obj, dict) and "model_state_dict" in raw_obj:
            # Validate checkpoint metadata against run configuration (R3-05)
            ckpt_dataset = raw_obj.get("dataset")
            ckpt_mode = raw_obj.get("data_mode")
            if args.data_mode == "real":
                if ckpt_mode != "real":
                    raise ValueError(f"[FAIL-CLOSED] Attempted to load {ckpt_mode} checkpoint into real mode run! (R3-05)")
                if ckpt_dataset and ckpt_dataset != dataset_name:
                    raise ValueError(f"[FAIL-CLOSED] Cross-dataset checkpoint rejected: {ckpt_dataset} vs {dataset_name} (R3-05)")
            state_dict = raw_obj["model_state_dict"]
        else:
            state_dict = raw_obj

        model.load_state_dict(state_dict)
        logger.info(f"Loaded verified source checkpoint from {ckpt_path}")
    else:
        if args.data_mode == "real":
            raise FileNotFoundError(f"[FAIL-CLOSED] Checkpoint not found for real data mode at {ckpt_path}! Please train source first.")
        logger.warning(f"No trained checkpoint found at {ckpt_path}. Using current model weights for debug.")

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

        stats_path = args.spectral_stats
        if not stats_path:
            cand_stats = [
                get_spectral_stats_path("checkpoints", dataset_name, args.data_mode, args.seed),
                f"checkpoints/source_spectral_stats_{args.data_mode}_seed{args.seed}.pt",
                f"checkpoints/source_spectral_stats_seed{args.seed}.pt",
            ]
            for cs in cand_stats:
                if os.path.exists(cs):
                    stats_path = cs
                    break

        if stats_path and os.path.exists(stats_path):
            detector.load_source_stats(stats_path)
            logger.info(f"Loaded source anchor statistics from {stats_path}")
        else:
            if args.data_mode == "real":
                raise FileNotFoundError(f"[FAIL-CLOSED] Spectral stats missing for real mode: {stats_path}")
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
        conf_cfg = dict(config.get("conformal", {}))
        conf_cfg["delay"] = args.delay
        conformal_predictor = OnlineACI(num_classes=num_classes, alpha=target_alpha, config=conf_cfg)
    elif args.conformal == "static_cp":
        conformal_predictor = SplitConformalPredictor(num_classes=num_classes, alpha=target_alpha)
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

    # 6. Stream benchmark through 6 sequential regimes (Protocol V2 with explicit R6 Recovery, R3-06)
    shift_gen = ShiftStreamGenerator(config, seed=args.seed, use_protocol_v2=True)
    regimes = shift_gen.get_regime_sequence()

    overall_tracker = MetricsTracker(num_classes=num_classes, target_coverage=(1.0 - target_alpha))
    regime_results = {}
    time_series_data = {
        "shift_velocity": [],
        "cross_modal_disagreement": [],
        "alpha_t": [],
        "coverage_hits": [],
        "prediction_set_sizes": [],
    }

    logger.info(
        f"Starting Continual Compound Shift Stream benchmark ({args.method} + {args.conformal} | "
        f"protocol: {args.protocol} | mode: {args.data_mode} | delay: {args.delay})..."
    )
    vram.reset_peak()

    for regime_info in regimes:
        reg_idx = regime_info["regime_idx"]
        reg_name = regime_info["name"]
        reg_tracker = MetricsTracker(num_classes=num_classes, target_coverage=(1.0 - target_alpha))
        logger.info(f"Entering {reg_name}...")

        total_batches = len(test_loader)
        for step, raw_batch in enumerate(test_loader):
            batch = shift_gen.apply_regime(raw_batch, regime_idx=reg_idx, step_in_regime=step, total_steps_in_regime=total_batches)

            features = {k: v.to(device) for k, v in batch["features"].items()}
            mod_mask = {k: v.to(device) for k, v in batch["modality_mask"].items()}
            labels = batch["labels"].to(device)
            bsz = labels.size(0)

            latency.start()

            # Prequential protocol (R3-06): evaluate prediction with theta_t BEFORE adapting
            if args.protocol == "prequential":
                if tta_method is not None and hasattr(tta_method, "detector"):
                    shift_scores = tta_method.detector.compute_shift_scores(features, mod_mask)
                    rel_weights = tta_method.detector.compute_reliability_weights(shift_scores, mod_mask, tau=tta_method.tau)
                    anchor_s, adaptable_s, corrupted_s = tta_method._classify_modalities(rel_weights, mod_mask)
                    eff_mask = {m: (mod_mask[m] > 0.5) & (m not in corrupted_s) for m in MODALITY_NAMES}
                    with torch.no_grad():
                        out_pre = model(features, modality_mask=mod_mask, reliability_weights=rel_weights, effective_mask=eff_mask)
                        logits = out_pre["logits"]
                    avg_shift_vel = tta_method.detector.compute_shift_velocity()
                    disagree = tta_method.get_cross_modal_disagreement(
                        out_pre["per_modality_logits"],
                        [m for m in MODALITY_NAMES if m not in corrupted_s],
                        modality_mask=mod_mask,
                    )
                else:
                    with torch.no_grad():
                        out_pre = model(features, modality_mask=mod_mask)
                        logits = out_pre["logits"]
                    avg_shift_vel = 0.0
                    disagree = 0.0
            else:
                # Transductive protocol: adapt first, then predict
                if tta_method is not None:
                    step_res = tta_method.adapt_step(features, mod_mask)
                    logits = step_res["logits"]
                    shift_vel = tta_method.get_shift_velocity() if hasattr(tta_method, "get_shift_velocity") else {}
                    avg_shift_vel = float(np.mean(list(shift_vel.values()))) if shift_vel else 0.0
                    disagree = step_res.get("cross_modal_disagreement", 0.0)
                    if isinstance(disagree, torch.Tensor):
                        disagree = float(disagree.item())
                else:
                    with torch.no_grad():
                        out_trans = model(features, modality_mask=mod_mask)
                        logits = out_trans["logits"]
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
                        p_set = conformal_predictor.predict_set(
                            sample_logits,
                            m_key,
                            shift_velocity=avg_shift_vel,
                            disagreement=disagree,
                        )[0]
                        pred_sets.append(p_set)
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

            # In prequential protocol: adapt model AFTER prediction sets are emitted (R3-06)
            if args.protocol == "prequential" and tta_method is not None:
                _ = tta_method.adapt_step(features, mod_mask)

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

    time_series_data["coverage_hits"] = overall_tracker.coverage_binary_stream

    final_results = {
        "dataset": dataset_name,
        "method": args.method,
        "conformal": args.conformal,
        "protocol": args.protocol,
        "data_mode": args.data_mode,
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

    # Save canonical namespaced artifact (R3-10)
    canonical_file = get_run_output_path(
        args.output_dir, dataset_name, args.data_mode, args.protocol,
        args.method, args.conformal, args.delay, args.seed
    )
    with open(canonical_file, "w") as f:
        json.dump(final_results, f, indent=2)

    # Save flat backward-compatible output
    delay_tag = f"_delay{args.delay}" if args.delay > 0 else ""
    flat_file = os.path.join(
        args.output_dir,
        f"{args.method}_{args.conformal}_{dataset_name}_{args.data_mode}{delay_tag}_seed{args.seed}.json"
    )
    with open(flat_file, "w") as f:
        json.dump(final_results, f, indent=2)

    logger.info(f"Results successfully saved to {canonical_file} (and {flat_file})")
    logger.info(
        f"Overall: Macro-F1={overall_metrics['macro_f1']:.4f}, Coverage={overall_metrics.get('empirical_coverage', 0.0):.4f}, "
        f"Coverage Gap={overall_metrics.get('coverage_gap', 0.0):.4f}, Set Size={overall_metrics.get('avg_set_size', 0.0):.2f}, "
        f"Peak VRAM={vram_stats['peak_mb']:.1f} MB, Latency={avg_latency:.2f} ms/sample"
    )


if __name__ == "__main__":
    main()
