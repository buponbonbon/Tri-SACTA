"""
src/service/router.py: Budgeted Reliability-Aware Modality Orchestrator (BRMO) Service Engine.

Core Systems Component of Tri-SACTA V12.
Provides dynamic multi-tier routing across unimodal, multimodal, and abstention pathways
subject to latency and cost budgets.
"""

from __future__ import annotations

import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from src.models.fusion_backbone import MultimodalFusionModel

CLASS_NAMES = ["negative", "neutral", "positive"]


class OrchestratorConfig:
    def __init__(
        self,
        spectral_threshold: float = 1.15,
        entropy_threshold: float = 0.65,
        min_confidence: float = 0.60,
        abstain_confidence: float = 0.40,
        max_latency_budget_ms: float = 50.0,
        cost_weights: Optional[Dict[str, float]] = None,
    ):
        self.spectral_threshold = spectral_threshold
        self.entropy_threshold = entropy_threshold
        self.min_confidence = min_confidence
        self.abstain_confidence = abstain_confidence
        self.max_latency_budget_ms = max_latency_budget_ms
        self.cost_weights = cost_weights or {"text": 1.0, "audio": 1.2, "visual": 1.3}


class BRMORouter:
    """Budgeted Reliability-Aware Modality Orchestrator Router."""

    def __init__(
        self,
        model: MultimodalFusionModel,
        clean_text_mean: torch.Tensor,
        clean_text_std: torch.Tensor,
        config: Optional[OrchestratorConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.config = config or OrchestratorConfig()
        self.device = device or next(model.parameters()).device
        self.clean_text_mean = clean_text_mean.to(self.device)
        self.clean_text_std = clean_text_std.to(self.device)
        self.model.eval()

    def compute_text_spectral_score(self, text_feat: torch.Tensor) -> float:
        """Computes normalized distance of text vector from clean training distribution."""
        norm_diff = torch.abs(text_feat - self.clean_text_mean) / self.clean_text_std
        return float(torch.mean(norm_diff).item())

    def route_and_predict(
        self,
        features: Dict[str, torch.Tensor],
        modality_mask: Optional[Dict[str, bool]] = None,
        policy: str = "orchestrated_brmo",
    ) -> Dict[str, Any]:
        """Routes multimodal request according to specified policy and returns full decision telemetry."""
        t0 = time.perf_counter()
        modality_mask = modality_mask or {m: True for m in ["text", "audio", "visual"]}

        has_text = modality_mask.get("text", False) and ("text" in features)
        has_audio = modality_mask.get("audio", False) and ("audio" in features)
        has_visual = modality_mask.get("visual", False) and ("visual" in features)

        t_feat = features["text"].to(self.device).view(1, -1) if has_text else None
        a_feat = features["audio"].to(self.device).view(1, -1) if has_audio else None
        v_feat = features["visual"].to(self.device).view(1, -1) if has_visual else None

        routing_decision = "text_fast_path"
        invoked_modalities = ["text"] if has_text else []
        spectral_score = 0.0
        entropy = 0.0
        max_p = 0.0
        abstained = False

        with torch.no_grad():
            if policy == "text_only":
                if has_text:
                    out_t = self.model.projections["text"](t_feat)
                    ad_t = self.model.adapters["text"](out_t)
                    logits = self.model.modality_classifiers["text"](ad_t)
                else:
                    logits = torch.zeros(1, 3, device=self.device)

            elif policy == "always_on":
                routing_decision = "multimodal_full_escalation"
                invoked_modalities = ["text", "audio", "visual"]
                feats = {
                    "text": t_feat if has_text else torch.zeros(1, 768, device=self.device),
                    "audio": a_feat if has_audio else torch.zeros(1, 74, device=self.device),
                    "visual": v_feat if has_visual else torch.zeros(1, 35, device=self.device),
                }
                mask = {
                    "text": torch.tensor([has_text], device=self.device),
                    "audio": torch.tensor([has_audio], device=self.device),
                    "visual": torch.tensor([has_visual], device=self.device),
                }
                out = self.model(feats, modality_mask=mask)
                logits = out["logits"]

            elif policy == "fixed_fallback":
                if has_text:
                    out_t = self.model.projections["text"](t_feat)
                    ad_t = self.model.adapters["text"](out_t)
                    logits = self.model.modality_classifiers["text"](ad_t)
                else:
                    routing_decision = "fallback_audio_visual"
                    invoked_modalities = ["audio", "visual"]
                    feats = {
                        "text": torch.zeros(1, 768, device=self.device),
                        "audio": a_feat if has_audio else torch.zeros(1, 74, device=self.device),
                        "visual": v_feat if has_visual else torch.zeros(1, 35, device=self.device),
                    }
                    mask = {
                        "text": torch.tensor([False], device=self.device),
                        "audio": torch.tensor([has_audio], device=self.device),
                        "visual": torch.tensor([has_visual], device=self.device),
                    }
                    out = self.model(feats, modality_mask=mask)
                    logits = out["logits"]

            elif policy == "orchestrated_brmo":
                # Tier 1: Fast unimodal triage
                if has_text:
                    spectral_score = self.compute_text_spectral_score(t_feat)
                    out_t = self.model.projections["text"](t_feat)
                    ad_t = self.model.adapters["text"](out_t)
                    logits_t = self.model.modality_classifiers["text"](ad_t)
                    probs_t = F.softmax(logits_t, dim=-1)
                    max_p = torch.max(probs_t).item()
                    entropy = -torch.sum(probs_t * torch.log(probs_t + 1e-12)).item() / np.log(3)

                    # Triage condition: clean text with low entropy
                    if (
                        spectral_score <= self.config.spectral_threshold
                        and entropy <= self.config.entropy_threshold
                        and max_p >= self.config.min_confidence
                    ):
                        routing_decision = "text_fast_path"
                        logits = logits_t
                    else:
                        # Escalate to multimodal
                        routing_decision = "escalated_multimodal_brmo"
                        invoked_modalities = ["text", "audio", "visual"]
                        weight_t = 0.2 if spectral_score > self.config.spectral_threshold else 1.0
                        rel_weights = {
                            "text": torch.tensor([weight_t], device=self.device),
                            "audio": torch.tensor([1.0], device=self.device),
                            "visual": torch.tensor([1.0], device=self.device),
                            "prosody": torch.tensor([0.0], device=self.device),
                        }
                        feats = {
                            "text": t_feat,
                            "audio": a_feat if has_audio else torch.zeros(1, 74, device=self.device),
                            "visual": v_feat if has_visual else torch.zeros(1, 35, device=self.device),
                        }
                        mask = {
                            "text": torch.tensor([True], device=self.device),
                            "audio": torch.tensor([has_audio], device=self.device),
                            "visual": torch.tensor([has_visual], device=self.device),
                        }
                        out = self.model(feats, modality_mask=mask, reliability_weights=rel_weights)
                        logits = out["logits"]
                else:
                    routing_decision = "fallback_missing_text"
                    invoked_modalities = ["audio", "visual"]
                    feats = {
                        "text": torch.zeros(1, 768, device=self.device),
                        "audio": a_feat if has_audio else torch.zeros(1, 74, device=self.device),
                        "visual": v_feat if has_visual else torch.zeros(1, 35, device=self.device),
                    }
                    mask = {
                        "text": torch.tensor([False], device=self.device),
                        "audio": torch.tensor([has_audio], device=self.device),
                        "visual": torch.tensor([has_visual], device=self.device),
                    }
                    out = self.model(feats, modality_mask=mask)
                    logits = out["logits"]

            else:
                raise ValueError(f"Unknown routing policy: {policy}")

            # Compute output probabilities
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
            pred_class = int(np.argmax(probs))
            final_conf = float(np.max(probs))

            # Tier 3: Conformal risk check
            if policy == "orchestrated_brmo" and final_conf < self.config.abstain_confidence:
                abstained = True

        lat_ms = (time.perf_counter() - t0) * 1000.0
        cost = sum(self.config.cost_weights.get(m, 1.0) for m in invoked_modalities)

        return {
            "prediction": CLASS_NAMES[pred_class],
            "predicted_class_id": pred_class,
            "confidence": final_conf,
            "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(3)},
            "abstained": abstained,
            "routing_decision": routing_decision,
            "invoked_modalities": invoked_modalities,
            "spectral_anomaly_score": spectral_score,
            "normalized_entropy": entropy,
            "latency_ms": lat_ms,
            "compute_cost": cost,
            "policy": policy,
        }
