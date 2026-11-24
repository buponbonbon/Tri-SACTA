"""
src/service/app.py: Fail-Closed FastAPI Service for Tri-SACTA V12.1.
Exposes the Budgeted Reliability-Aware Modality Orchestration (BRMO) service.
Guarantees fail-closed initialization: raises explicit RuntimeError if model checkpoints
or calibration statistics are missing.
"""

from __future__ import annotations

import os
import sys
import glob
from typing import Dict, Any, List, Optional
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, status
from pydantic import ValidationError

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.models.fusion_backbone import MultimodalFusionModel, SimpleMLP
from src.service.schemas import (
    CLASS_NAMES,
    ModalityInput,
    PredictRequest,
    PredictResponse,
)
from src.service.router import BRMORouter, OrchestratorConfig
from src.service.reliability import SplitConformalPredictor, StandardizedCentroidDistance


app = FastAPI(
    title="Tri-SACTA V12.1 BRMO Inference Service",
    description="Scientific Rectification: Budget-Constrained Multimodal Orchestration with Calibrated Conformal Risk",
    version="12.1.0",
)

# Global router singleton
_ROUTER_INSTANCE: Optional[BRMORouter] = None


def _load_fail_closed_router() -> BRMORouter:
    """Loads all models and calibration assets, failing closed if any artifact is missing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.join(repo_root, "results", "v12_1_systems", "training", "checkpoints")

    if not os.path.exists(ckpt_dir):
        raise RuntimeError(
            f"FAIL-CLOSED: Checkpoint directory '{ckpt_dir}' does not exist. "
            f"Run 'python scripts/v12_1_train_predictor_bank.py' before starting service."
        )

    # Verify clean text reference statistics
    train_text_path = os.path.join(repo_root, "data", "cmu_mosei", "features_cache", "train", "text.pt")
    if not os.path.exists(train_text_path):
        raise RuntimeError(f"FAIL-CLOSED: Clean text training cache not found at '{train_text_path}'.")

    text_train = torch.load(train_text_path, map_location="cpu", weights_only=False).float()
    text_mean = torch.mean(text_train, dim=0)
    text_std = torch.std(text_train, dim=0) + 1e-6

    # Load cal split for conformal calibration
    cal_text_path = os.path.join(repo_root, "data", "cmu_mosei", "features_cache", "cal", "text.pt")
    cal_audio_path = os.path.join(repo_root, "data", "cmu_mosei", "features_cache", "cal", "audio.pt")
    cal_visual_path = os.path.join(repo_root, "data", "cmu_mosei", "features_cache", "cal", "visual.pt")
    cal_meta_path = os.path.join(repo_root, "data", "cmu_mosei", "features_cache", "cal", "meta.pt")

    if not all(os.path.exists(p) for p in [cal_text_path, cal_audio_path, cal_visual_path, cal_meta_path]):
        raise RuntimeError("FAIL-CLOSED: Held-out 'cal' split features not found in data/cmu_mosei/features_cache/cal.")

    cal_text = torch.load(cal_text_path, map_location="cpu", weights_only=False).float()
    cal_audio = torch.load(cal_audio_path, map_location="cpu", weights_only=False).float()
    cal_visual = torch.load(cal_visual_path, map_location="cpu", weights_only=False).float()
    cal_meta = torch.load(cal_meta_path, map_location="cpu", weights_only=False)
    raw_cal_sentiment = np.array([m["raw_sentiment"] for m in cal_meta], dtype=np.float32)
    cal_labels_np = np.zeros(len(raw_cal_sentiment), dtype=np.int64)
    cal_labels_np[raw_cal_sentiment < 0.0] = 0
    cal_labels_np[raw_cal_sentiment == 0.0] = 1
    cal_labels_np[raw_cal_sentiment > 0.0] = 2

    # Deterministic LR baselines
    import joblib
    lr_t_path = os.path.join(ckpt_dir, "text_only_lr.joblib")
    lr_c_path = os.path.join(ckpt_dir, "multimodal_concat_lr.joblib")
    clf_text_lr = joblib.load(lr_t_path) if os.path.exists(lr_t_path) else None
    clf_concat_lr = joblib.load(lr_c_path) if os.path.exists(lr_c_path) else None

    models_by_seed: Dict[int, Dict[str, Any]] = {}
    conformal_by_seed: Dict[int, SplitConformalPredictor] = {}

    SEEDS = [42, 100, 2026]
    for seed in SEEDS:
        seed_models: Dict[str, Any] = {}

        # 1. Text MLP
        t_path = os.path.join(ckpt_dir, f"text_only_mlp_seed{seed}.pt")
        if not os.path.exists(t_path):
            raise RuntimeError(f"FAIL-CLOSED: Missing checkpoint '{t_path}'.")
        t_mlp = SimpleMLP(in_dim=768, hidden_dim=64, num_classes=3)
        t_mlp.load_state_dict(torch.load(t_path, map_location=device, weights_only=False))
        seed_models["text_only_mlp"] = t_mlp

        # 2. Audio MLP
        a_path = os.path.join(ckpt_dir, f"audio_only_mlp_seed{seed}.pt")
        if not os.path.exists(a_path):
            raise RuntimeError(f"FAIL-CLOSED: Missing checkpoint '{a_path}'.")
        a_mlp = SimpleMLP(in_dim=74, hidden_dim=32, num_classes=3)
        a_mlp.load_state_dict(torch.load(a_path, map_location=device, weights_only=False))
        seed_models["audio_only_mlp"] = a_mlp

        # 3. Visual MLP
        v_path = os.path.join(ckpt_dir, f"visual_only_mlp_seed{seed}.pt")
        if not os.path.exists(v_path):
            raise RuntimeError(f"FAIL-CLOSED: Missing checkpoint '{v_path}'.")
        v_mlp = SimpleMLP(in_dim=35, hidden_dim=16, num_classes=3)
        v_mlp.load_state_dict(torch.load(v_path, map_location=device, weights_only=False))
        seed_models["visual_only_mlp"] = v_mlp

        # 4. Concat MLP
        c_path = os.path.join(ckpt_dir, f"multimodal_concat_mlp_seed{seed}.pt")
        if not os.path.exists(c_path):
            raise RuntimeError(f"FAIL-CLOSED: Missing checkpoint '{c_path}'.")
        c_mlp = SimpleMLP(in_dim=768 + 74 + 35, hidden_dim=64, num_classes=3)
        c_mlp.load_state_dict(torch.load(c_path, map_location=device, weights_only=False))
        seed_models["multimodal_concat_mlp"] = c_mlp

        # 5. Multimodal Fusion Model
        f_path = os.path.join(ckpt_dir, f"multimodal_fusion_seed{seed}.pt")
        if not os.path.exists(f_path):
            raise RuntimeError(f"FAIL-CLOSED: Missing checkpoint '{f_path}'.")
        cfg_f = {
            "model": {"fusion_dim": 256, "num_heads": 4, "num_layers": 2, "adapter_rank": 8, "dropout": 0.1, "num_classes": 3},
            "features": {"visual_dim": 35, "audio_dim": 74, "text_dim": 768, "prosody_dim": 88},
        }
        f_model = MultimodalFusionModel(cfg_f)
        f_model.load_state_dict(torch.load(f_path, map_location=device, weights_only=False))
        seed_models["multimodal_fusion"] = f_model

        # 6. Multimodal Fusion Missingness-Aware
        fm_path = os.path.join(ckpt_dir, f"multimodal_fusion_missingness_aware_seed{seed}.pt")
        if not os.path.exists(fm_path):
            raise RuntimeError(f"FAIL-CLOSED: Missing checkpoint '{fm_path}'.")
        fm_model = MultimodalFusionModel(cfg_f)
        fm_model.load_state_dict(torch.load(fm_path, map_location=device, weights_only=False))
        seed_models["multimodal_fusion_missingness_aware"] = fm_model

        if clf_text_lr is not None:
            seed_models["text_only_lr"] = clf_text_lr
        if clf_concat_lr is not None:
            seed_models["multimodal_concat_lr"] = clf_concat_lr

        models_by_seed[seed] = seed_models

        # Calibrate true Split Conformal Predictor on held-out 'cal' split for this seed
        cp = SplitConformalPredictor(alpha=0.10)
        fm_model.eval().to(device)
        with torch.no_grad():
            cal_feats = {
                "text": cal_text.to(device),
                "audio": cal_audio.to(device),
                "visual": cal_visual.to(device),
            }
            cal_mask = {
                "text": torch.ones(len(cal_labels_np), dtype=torch.bool, device=device),
                "audio": torch.ones(len(cal_labels_np), dtype=torch.bool, device=device),
                "visual": torch.ones(len(cal_labels_np), dtype=torch.bool, device=device),
            }
            out_cal = fm_model(cal_feats, modality_mask=cal_mask)
            cal_probs = torch.softmax(out_cal["logits"], dim=-1).cpu().numpy()

        cp.fit_from_probabilities(cal_probs, cal_labels_np)
        conformal_by_seed[seed] = cp

    router = BRMORouter(
        models_by_seed=models_by_seed,
        clean_text_mean=text_mean,
        clean_text_std=text_std,
        conformal_predictors_by_seed=conformal_by_seed,
        device=device,
    )
    return router


def get_router() -> BRMORouter:
    global _ROUTER_INSTANCE
    if _ROUTER_INSTANCE is None:
        _ROUTER_INSTANCE = _load_fail_closed_router()
    return _ROUTER_INSTANCE


@app.get("/health")
def health_check():
    """Health check endpoint providing runtime device, parameter counts, and calibration telemetry."""
    try:
        router = get_router()
        ref_model = router.models_by_seed[42]["multimodal_fusion"]
        ref_cp = router.conformal_predictors_by_seed[42]
        return {
            "status": "healthy",
            "service": "Tri-SACTA V12.1 BRMO",
            "device": str(router.device),
            "available_seeds": list(router.models_by_seed.keys()),
            "total_fusion_parameters": sum(p.numel() for p in ref_model.parameters()),
            "conformal_calibration": {
                "status": "calibrated",
                "calibration_samples": ref_cp.num_calibration_samples,
                "target_coverage": float(1.0 - ref_cp.alpha),
                "calibrated_quantile": float(ref_cp.calibrated_quantile),
            },
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service initialization failed (fail-closed): {str(e)}",
        )


@app.get("/policies")
def list_policies():
    """Lists all valid routing policy identifiers."""
    return {
        "available_policies": [
            "majority_class",
            "text_only_lr",
            "text_only_mlp",
            "audio_only_mlp",
            "visual_only_mlp",
            "always_on_fusion",
            "multimodal_concat_mlp",
            "fixed_fallback",
            "confidence_routing",
            "historical_v12_brmo",
            "orchestrated_brmo",
        ],
        "default": "orchestrated_brmo",
    }


@app.get("/calibration")
def get_calibration_info(seed: int = 42):
    """Returns conformal calibration details for specified seed."""
    router = get_router()
    if seed not in router.conformal_predictors_by_seed:
        raise HTTPException(status_code=400, detail=f"Seed {seed} not available.")
    cp = router.conformal_predictors_by_seed[seed]
    return {
        "seed": seed,
        "split": "cal",
        "num_calibration_samples": cp.num_calibration_samples,
        "alpha": cp.alpha,
        "target_coverage": float(1.0 - cp.alpha),
        "calibrated_quantile": float(cp.calibrated_quantile),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Performs budgeted, reliability-aware multimodal sentiment classification."""
    router = get_router()
    feats: Dict[str, Optional[torch.Tensor]] = {}

    if request.features.text is not None:
        feats["text"] = torch.tensor(request.features.text, dtype=torch.float32)
    if request.features.audio is not None:
        feats["audio"] = torch.tensor(request.features.audio, dtype=torch.float32)
    if request.features.visual is not None:
        feats["visual"] = torch.tensor(request.features.visual, dtype=torch.float32)

    try:
        res = router.route_and_predict(
            features=feats,
            modality_mask=request.modality_mask,
            deadline_ms=request.deadline_ms,
            cost_budget=request.cost_budget,
            policy=request.policy,
            seed=request.seed,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
