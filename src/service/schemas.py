"""
src/service/schemas.py: Strongly typed schemas and validators for Tri-SACTA V12.1 service.
Defines input payloads, validation rules (dimension check, finite values, positive budgets),
and structured decision telemetry responses.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


CLASS_NAMES = ["negative", "neutral", "positive"]

EXPECTED_DIMS = {
    "text": 768,
    "audio": 74,
    "visual": 35,
}


def _check_finite_list(vals: Optional[List[float]], name: str, expected_dim: int) -> None:
    if vals is None:
        return
    if len(vals) != expected_dim:
        raise ValueError(
            f"Modality '{name}' feature dimension mismatch: expected {expected_dim}, got {len(vals)}"
        )
    for i, v in enumerate(vals):
        if not math.isfinite(v):
            raise ValueError(f"Modality '{name}' contains non-finite float at index {i}: {v}")


class ModalityInput(BaseModel):
    """Raw feature embeddings for multimodal sentiment classification."""
    text: Optional[List[float]] = Field(None, description="768-dimensional text embedding vector")
    audio: Optional[List[float]] = Field(None, description="74-dimensional acoustic feature vector")
    visual: Optional[List[float]] = Field(None, description="35-dimensional visual feature vector")

    @model_validator(mode="after")
    def validate_features(self) -> ModalityInput:
        _check_finite_list(self.text, "text", EXPECTED_DIMS["text"])
        _check_finite_list(self.audio, "audio", EXPECTED_DIMS["audio"])
        _check_finite_list(self.visual, "visual", EXPECTED_DIMS["visual"])
        return self


class PredictRequest(BaseModel):
    """Inference request with optional quality-of-service and budget constraints."""
    features: ModalityInput
    modality_mask: Optional[Dict[str, bool]] = Field(
        None,
        description="Explicit modality availability mask. If omitted, inferred from non-null features."
    )
    deadline_ms: Optional[float] = Field(
        None,
        gt=0.0,
        description="Inference deadline in milliseconds (e.g. 15.0, 30.0, 50.0). None implies no deadline."
    )
    cost_budget: Optional[float] = Field(
        None,
        ge=0.0,
        description="Total computational/sensor cost ceiling (e.g. 1.0 for text-only, 3.5 for full A+V). None implies unbounded."
    )
    policy: str = Field(
        "orchestrated_brmo",
        description="Routing policy: majority_class, text_only_lr, text_only_mlp, audio_only_mlp, "
                    "visual_only_mlp, always_on_fusion, multimodal_concat_mlp, fixed_fallback, "
                    "confidence_routing, seeded_random_budget, historical_v12_brmo, orchestrated_brmo"
    )
    seed: int = Field(
        42,
        description="Model checkpoint seed to execute (42, 100, 2026)"
    )

    @model_validator(mode="after")
    def validate_request(self) -> PredictRequest:
        f = self.features
        if f.text is None and f.audio is None and f.visual is None:
            raise ValueError("At least one modality feature vector (text, audio, or visual) must be provided.")
        return self


class PredictResponse(BaseModel):
    """Comprehensive decision telemetry and prediction response."""
    prediction: str = Field(..., description="Predicted sentiment class (negative, neutral, positive)")
    predicted_class_id: int = Field(..., description="Predicted class integer ID [0, 1, 2]")
    confidence: float = Field(..., description="Softmax confidence score of winning class")
    probabilities: Dict[str, float] = Field(..., description="Full 3-class probability distribution")
    abstained: bool = Field(..., description="Whether prediction was withheld due to uncertainty/conformal set size")
    nonconformity_score: float = Field(..., description="Empirical nonconformity score 1 - p(y_pred)")
    conformal_set: List[str] = Field(..., description="Conformal prediction set at calibrated coverage 1 - alpha")
    conformal_set_size: int = Field(..., description="Cardinality of conformal prediction set")
    routing_decision: str = Field(..., description="System routing execution path identifier")
    invoked_modalities: List[str] = Field(..., description="List of modalities actually fetched and computed")
    standardized_centroid_distance: float = Field(..., description="Standardized L1 centroid deviation from training mean")
    normalized_entropy: float = Field(..., description="Entropy of text-stage probabilities normalized by ln(3)")
    latency_ms: float = Field(..., description="Total elapsed latency in milliseconds")
    compute_cost: float = Field(..., description="Total sensor acquisition and inference cost incurred")
    policy: str = Field(..., description="Routing policy executed")
    deadline_met: bool = Field(..., description="Whether execution finished within requested deadline_ms")
    budget_met: bool = Field(..., description="Whether execution remained within requested cost_budget")
