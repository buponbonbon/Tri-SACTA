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
import math
import hashlib
import argparse
import logging
import json
import yaml
from collections import defaultdict
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
    parser.add_argument("--spectral_stats", "--spectral-stats", dest="spectral_stats", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", type=str, default="tri_sacta", choices=["tri_sacta", "tent", "sar", "read", "source_only", "reliability_only", "full_adaptation"])
    parser.add_argument("--conformal", type=str, default="online_aci", choices=["online_aci", "fixed_mask_aci", "dynamic_mask_aci", "global_aci", "static_cp", "mccf", "none"])
    parser.add_argument("--delay", type=int, default=0, help="Delayed feedback steps (0, 10, 50)")
    parser.add_argument("--protocol", type=str, default="prequential", choices=["prequential", "transductive"], help="Stream protocol (R3-06)")
    parser.add_argument("--data-mode", "--data_mode", dest="data_mode", type=str, default="synthetic", choices=["synthetic", "real"], help="Data mode (AUD-10)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", type=str, default="results")
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

    current_manifest_hash = None
    if args.data_mode == "real" and config.get("dataset", {}).get("manifest_path"):
        from src.data.manifest import DataManifest
        m_path = config["dataset"]["manifest_path"]
        if os.path.exists(m_path):
            current_manifest = DataManifest.load_json(m_path)
            current_manifest_hash = current_manifest.compute_manifest_hash()

    # 2. Model setup
    model = MultimodalFusionModel(config).to(device)
    num_classes = config.get("model", {}).get("num_classes", 7)
    target_alpha = config.get("conformal", {}).get("alpha", 0.10)

    # Resolve namespaced checkpoint (R3-05, V5-05)
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
            # Validate checkpoint metadata against run configuration (R3-05, V5-05)
            ckpt_dataset = raw_obj.get("dataset")
            ckpt_mode = raw_obj.get("data_mode")
            ckpt_seed = raw_obj.get("seed")
            ckpt_manifest_hash = raw_obj.get("manifest_hash")

            if args.data_mode == "real":
                if ckpt_mode != "real":
                    raise ValueError(f"[FAIL-CLOSED] Attempted to load {ckpt_mode} checkpoint into real mode run! (R3-05)")
                if ckpt_dataset and ckpt_dataset != dataset_name:
                    raise ValueError(f"[FAIL-CLOSED] Cross-dataset checkpoint rejected: {ckpt_dataset} vs {dataset_name} (R3-05)")
                if ckpt_seed is not None and ckpt_seed != args.seed:
                    raise ValueError(f"[FAIL-CLOSED] Seed mismatch in checkpoint: {ckpt_seed} vs {args.seed} (V5-05)")
                if current_manifest_hash and ckpt_manifest_hash != current_manifest_hash:
                    raise ValueError(
                        f"[FAIL-CLOSED] Checkpoint manifest_hash mismatch: "
                        f"checkpoint has '{ckpt_manifest_hash}', current manifest is '{current_manifest_hash}' (V5-05)"
                    )
            state_dict = raw_obj["model_state_dict"]
        else:
            if args.data_mode == "real":
                raise ValueError(
                    f"[FAIL-CLOSED] Legacy bare state dictionary rejected in real mode (V5-05).\n"
                    f"A verified schema_v5 checkpoint with cryptographic manifest hash is required."
                )
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
            expected_meta = None
            if args.data_mode == "real":
                expected_meta = {
                    "dataset_name": dataset_name,
                    "data_mode": args.data_mode,
                    "manifest_hash": current_manifest_hash,
                }
            detector.load_source_stats(stats_path, expected_metadata=expected_meta)
            logger.info(f"Loaded verified source anchor statistics from {stats_path}")
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

    # 4. Initialize TTA Method (Task 9: source_only, reliability_only, full_adaptation, tri_sacta)
    tta_method = None
    if args.method == "tri_sacta":
        tta_method = TriSACTA(model, detector, config)
    elif args.method == "reliability_only":
        rel_config = copy.deepcopy(config)
        rel_config["tta"]["lr"] = 0.0
        tta_method = TriSACTA(model, detector, rel_config)
    elif args.method == "full_adaptation":
        full_config = copy.deepcopy(config)
        full_config["tta"]["theta_high"] = 0.0
        full_config["tta"]["theta_low"] = 0.0
        tta_method = TriSACTA(model, detector, full_config)
    elif args.method == "tent":
        tta_method = TENT(model, config)
    elif args.method == "sar":
        tta_method = SAR(model, config)
    elif args.method == "read":
        tta_method = MultimodalTTABaseline(model, config)
    elif args.method == "source_only":
        model.eval()

    # 5. Initialize Conformal Predictor (Task 10: fixed_mask_aci, dynamic_mask_aci, global_aci, mccf, static_cp)
    conformal_predictor = None
    if args.conformal in ["online_aci", "fixed_mask_aci", "dynamic_mask_aci", "global_aci"]:
        conf_cfg = dict(config.get("conformal", {}))
        conf_cfg["delay"] = args.delay
        if args.conformal == "fixed_mask_aci":
            conf_cfg["controller"] = "fixed_mask_aci"
        elif args.conformal in ["online_aci", "dynamic_mask_aci"]:
            conf_cfg["controller"] = "dynamic_mask_aci"
        elif args.conformal == "global_aci":
            conf_cfg["controller"] = "fixed_mask_aci"
            conf_cfg["min_mask_support"] = 999999  # Force pooled global window
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

    # Provenance hashes (V5-05, V6-09)
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()
    checkpoint_hash = "none"
    if ckpt_path and os.path.exists(ckpt_path):
        checkpoint_hash = hashlib.sha256(open(ckpt_path, "rb").read()).hexdigest()
    stream_hasher = hashlib.sha256()

    all_pred_labels: List[int] = []
    all_true_labels: List[int] = []
    all_modality_masks: List[str] = []
    per_event_telemetry: List[Dict[str, Any]] = []
    total_adapted_batches: int = 0
    total_evaluated_batches: int = 0
    per_regime_adapted: Dict[str, int] = defaultdict(int)
    total_param_delta: float = 0.0

    # Genuine abstention counters (Task 6)
    total_abstained_count: int = 0
    per_regime_abstained: Dict[str, int] = defaultdict(int)
    non_abstained_preds: List[int] = []
    non_abstained_trues: List[int] = []
    if hasattr(model, "reset_counters"):
        model.reset_counters()

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

            # Update stream hash deterministically (Task 8: segment IDs, corrupted features, physical masks)
            stream_hasher.update(str(reg_idx).encode("utf-8"))
            seg_ids_str = ",".join(batch.get("segment_ids", [str(i) for i in range(bsz)]))
            stream_hasher.update(seg_ids_str.encode("utf-8"))
            stream_hasher.update(labels.detach().cpu().numpy().tobytes())
            for m in sorted(batch["features"].keys()):
                stream_hasher.update(m.encode("utf-8"))
                stream_hasher.update(batch["features"][m].detach().cpu().numpy().tobytes())
            for m in sorted(batch["modality_mask"].keys()):
                stream_hasher.update(m.encode("utf-8"))
                stream_hasher.update(batch["modality_mask"][m].detach().cpu().numpy().tobytes())

            latency.start()

            # Prequential protocol (R3-06): evaluate prediction with theta_t BEFORE adapting
            precomputed_shift_scores = None
            step_res = None
            if args.protocol == "prequential":
                if tta_method is not None and hasattr(tta_method, "detector"):
                    # Update shift detector history EXACTLY ONCE per batch (R4-07)
                    shift_scores = tta_method.detector.compute_shift_scores(features, mod_mask, update_history=True)
                    precomputed_shift_scores = shift_scores
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
                    shift_vel = tta_method.get_shift_velocity() if hasattr(tta_method, "get_shift_velocity") else 0.0
                    if isinstance(shift_vel, dict):
                        avg_shift_vel = float(np.mean(list(shift_vel.values()))) if shift_vel else 0.0
                    elif isinstance(shift_vel, (int, float)):
                        avg_shift_vel = max(0.0, float(shift_vel))
                    else:
                        avg_shift_vel = 0.0
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

                    if args.conformal in ["online_aci", "fixed_mask_aci", "dynamic_mask_aci", "global_aci"]:
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

            # In prequential protocol: adapt model AFTER prediction sets are emitted (R3-06, R4-07)
            if args.protocol == "prequential" and tta_method is not None:
                if precomputed_shift_scores is not None and hasattr(tta_method, "adapt_step"):
                    step_res = tta_method.adapt_step(features, mod_mask, precomputed_shift_scores=precomputed_shift_scores)
                else:
                    step_res = tta_method.adapt_step(features, mod_mask)

            # Track adaptation activity
            total_evaluated_batches += 1
            if step_res is not None and step_res.get("adapted", False):
                total_adapted_batches += 1
                per_regime_adapted[f"regime_{reg_idx}"] += 1
                total_param_delta += float(step_res.get("param_delta", 0.0))

            latency_ms = latency.stop(num_samples=bsz)

            # Per-event conformal and label telemetry (V6-09, Task 6, Task 7)
            for i in range(bsz):
                s_pred = int(logits[i].argmax().item())
                s_true = int(labels[i].item())
                m_str = get_mask_key(batch["modality_mask"], i)
                s_set = pred_sets[i] if pred_sets is not None else [s_pred]
                is_hit = bool(s_true in s_set)

                all_pred_labels.append(s_pred)
                all_true_labels.append(s_true)
                all_modality_masks.append(m_str)

                # Abstention evaluation: check if sample had all modalities rejected (Task 6)
                is_abstained = False
                if args.protocol == "prequential":
                    if "all_modalities_rejected" in out_pre:
                        is_abstained = bool(out_pre["all_modalities_rejected"][i].item())
                else:
                    if step_res is not None and "all_modalities_rejected" in step_res:
                        is_abstained = bool(step_res["all_modalities_rejected"][i].item())
                    elif "all_modalities_rejected" in out_trans:
                        is_abstained = bool(out_trans["all_modalities_rejected"][i].item())

                if is_abstained:
                    total_abstained_count += 1
                    per_regime_abstained[f"regime_{reg_idx}"] += 1
                else:
                    non_abstained_preds.append(s_pred)
                    non_abstained_trues.append(s_true)

                # Pull precise telemetry directly from OnlineACI state (Task 7)
                if conformal_predictor is not None and hasattr(conformal_predictor, "get_last_event_telemetry"):
                    last_ev = conformal_predictor.get_last_event_telemetry()
                    if last_ev is not None:
                        alpha_before = last_ev["alpha_before"]
                        alpha_val = last_ev["alpha_after"]
                        eta_val = last_ev["eta_used"]
                        quantile_val = last_ev["quantile"]
                        residual_val = last_ev["clipping_residual"]
                        event_id_val = last_ev["event_id"]
                    else:
                        alpha_before = float(target_alpha)
                        alpha_val = float(conformal_predictor.alphas[m_str])
                        eta_val = float(getattr(conformal_predictor, "eta_0", 0.0))
                        quantile_val = 0.0
                        residual_val = 0.0
                        event_id_val = len(all_pred_labels)
                else:
                    alpha_before = float(target_alpha)
                    alpha_val = float(target_alpha)
                    eta_val = 0.0
                    quantile_val = 0.0
                    residual_val = 0.0
                    event_id_val = len(all_pred_labels)

                per_event_telemetry.append({
                    "t": len(all_pred_labels),
                    "event_id": event_id_val,
                    "regime": reg_idx,
                    "regime_name": reg_name,
                    "mask": m_str,
                    "true_label": s_true,
                    "predicted_label": s_pred,
                    "prediction_set": s_set,
                    "set_size": len(s_set),
                    "covered": is_hit,
                    "is_abstained": is_abstained,
                    "alpha_before": alpha_before,
                    "alpha_after": alpha_val,
                    "alpha_t": alpha_val,
                    "eta_t": eta_val,
                    "quantile": quantile_val,
                    "clipping_residual": residual_val,
                    "shift_velocity": float(avg_shift_vel),
                    "disagreement": float(disagree),
                })

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

    stream_hash_str = stream_hasher.hexdigest()

    # Calculate selective prediction metrics on non-abstained predictions (Task 6)
    from sklearn.metrics import accuracy_score, f1_score
    if non_abstained_trues:
        selective_acc = float(accuracy_score(non_abstained_trues, non_abstained_preds))
        selective_f1 = float(f1_score(non_abstained_trues, non_abstained_preds, average="macro", zero_division=0))
    else:
        selective_acc = 0.0
        selective_f1 = 0.0
    rejection_rate = float(total_abstained_count / max(1, len(all_pred_labels)))

    final_results = {
        "dataset": dataset_name,
        "method": args.method,
        "conformal": args.conformal,
        "protocol": args.protocol,
        "data_mode": args.data_mode,
        "seed": args.seed,
        "delay": args.delay,
        "manifest_hash": current_manifest_hash or "none",
        "checkpoint_hash": checkpoint_hash,
        "config_hash": config_hash,
        "stream_hash": stream_hash_str,
        "predicted_labels": all_pred_labels,
        "true_labels": all_true_labels,
        "modality_masks": all_modality_masks,
        "adaptation_counts": {
            "total_adapted_batches": total_adapted_batches,
            "total_evaluated_batches": total_evaluated_batches,
            "adaptation_rate": float(total_adapted_batches / max(total_evaluated_batches, 1)),
            "per_regime_adapted": dict(per_regime_adapted),
            "total_param_delta": float(total_param_delta),
        },
        "abstention_counts": {
            "total_emitted_predictions": len(all_pred_labels),
            "total_abstain": total_abstained_count,
            "rejection_rate": rejection_rate,
            "selective_accuracy": selective_acc,
            "selective_macro_f1": selective_f1,
            "per_regime_abstained": dict(per_regime_abstained),
            "total_fallback": getattr(model, "fallback_count", 0),
            "empty_prediction_sets": getattr(conformal_predictor, "_empty_set_count", 0) if conformal_predictor else 0,
            "full_prediction_sets": getattr(conformal_predictor, "_full_set_count", 0) if conformal_predictor else 0,
        },
        "overall_metrics": overall_metrics,
        "regime_metrics": regime_results,
        "efficiency": {
            "peak_vram_mb": vram_stats["peak_mb"],
            "avg_latency_ms_per_sample": avg_latency,
        },
        "time_series": time_series_data,
        "per_event_conformal_telemetry": per_event_telemetry,
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
