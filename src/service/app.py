"""
src/service/app.py: FastAPI Inference Application for Tri-SACTA V12 Systems Instrument.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Any, List, Optional
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.models.fusion_backbone import MultimodalFusionModel
from src.service.router import BRMORouter, OrchestratorConfig

app = FastAPI(
    title="Tri-SACTA V12 BRMO Inference Service",
    description="Budgeted Reliability-Aware Modality Orchestration for Multimodal Sentiment Systems",
    version="12.0.0",
)

# Global router instance
router_instance: Optional[BRMORouter] = None


class ModalityInput(BaseModel):
    text: Optional[List[float]] = Field(None, description="768D text feature vector")
    audio: Optional[List[float]] = Field(None, description="74D audio feature vector")
    visual: Optional[List[float]] = Field(None, description="35D visual feature vector")


class PredictRequest(BaseModel):
    features: ModalityInput
    modality_mask: Optional[Dict[str, bool]] = Field(None, description="Presence mask per modality")
    policy: str = Field("orchestrated_brmo", description="text_only, always_on, fixed_fallback, orchestrated_brmo")


class PredictResponse(BaseModel):
    prediction: str
    predicted_class_id: int
    confidence: float
    probabilities: Dict[str, float]
    abstained: bool
    routing_decision: str
    invoked_modalities: List[str]
    spectral_anomaly_score: float
    normalized_entropy: float
    latency_ms: float
    compute_cost: float
    policy: str


def get_router() -> BRMORouter:
    global router_instance
    if router_instance is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg = {
            "model": {
                "fusion_dim": 256,
                "num_heads": 4,
                "num_layers": 2,
                "adapter_rank": 8,
                "dropout": 0.1,
                "num_classes": 3,
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
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))

        # Training text stats
        train_text_path = os.path.join(repo_root, "data", "cmu_mosei", "features_cache", "train", "text.pt")
        if os.path.exists(train_text_path):
            text_train = torch.load(train_text_path, map_location="cpu", weights_only=False).float()
            text_mean = torch.mean(text_train, dim=0)
            text_std = torch.std(text_train, dim=0) + 1e-6
        else:
            text_mean = torch.zeros(768)
            text_std = torch.ones(768)

        router_instance = BRMORouter(model, text_mean, text_std, device=device)
    return router_instance


@app.get("/health")
def health_check():
    router = get_router()
    return {
        "status": "healthy",
        "service": "Tri-SACTA V12 BRMO",
        "device": str(router.device),
        "total_parameters": sum(p.numel() for p in router.model.parameters()),
    }


@app.get("/policies")
def list_policies():
    return {
        "available_policies": [
            "text_only",
            "always_on",
            "fixed_fallback",
            "orchestrated_brmo",
        ],
        "default": "orchestrated_brmo",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    router = get_router()
    feats = {}
    if request.features.text is not None:
        feats["text"] = torch.tensor(request.features.text, dtype=torch.float32)
    if request.features.audio is not None:
        feats["audio"] = torch.tensor(request.features.audio, dtype=torch.float32)
    if request.features.visual is not None:
        feats["visual"] = torch.tensor(request.features.visual, dtype=torch.float32)

    try:
        res = router.route_and_predict(feats, modality_mask=request.modality_mask, policy=request.policy)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
