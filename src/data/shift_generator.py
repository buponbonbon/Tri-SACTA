"""
Continual Compound Shift Stream Generator for Tri-SACTA.

Implements the deterministic 5 sequential shift regimes on the test stream:
- Regime 1: Clean baseline (0% missing).
- Regime 2: Audio degradation (Gaussian noise / SNR 5dB -> 0dB) + Visual clean.
- Regime 3: ASR Text drift (simulated WER 15% -> 35% perturbation) + Audio/Visual clean.
- Regime 4: Visual corruption (Gaussian blur, occlusion/masking) + 30% random missing modality.
- Regime 5: Severe compound shift (multi-channel noise + 50% missing modality) + return to Regime 1 clean.
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional, Tuple, Any

import torch
import numpy as np

logger = logging.getLogger(__name__)

MODALITY_NAMES = ["visual", "audio", "text", "prosody"]


class ShiftStreamGenerator:
    """Applies streaming compound shifts across sequential test regimes."""

    def __init__(self, config: Dict[str, Any], seed: int = 42):
        self.config = config
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.regimes = self._init_regime_definitions()

    def _init_regime_definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "regime_idx": 1,
                "name": "Regime 1: Clean Baseline",
                "audio_noise_std": 0.0,
                "text_wer_drift": 0.0,
                "visual_blur_sigma": 0.0,
                "missing_rate": 0.0,
            },
            {
                "regime_idx": 2,
                "name": "Regime 2: Audio Degradation",
                "audio_noise_std": 0.8,  # Equivalent to ~5dB-0dB SNR degradation
                "text_wer_drift": 0.0,
                "visual_blur_sigma": 0.0,
                "missing_rate": 0.0,
            },
            {
                "regime_idx": 3,
                "name": "Regime 3: ASR Text Drift",
                "audio_noise_std": 0.0,
                "text_wer_drift": 0.35,  # 35% simulated ASR token perturbation
                "visual_blur_sigma": 0.0,
                "missing_rate": 0.0,
            },
            {
                "regime_idx": 4,
                "name": "Regime 4: Visual Corruption + 30% Missing",
                "audio_noise_std": 0.0,
                "text_wer_drift": 0.0,
                "visual_blur_sigma": 1.2,
                "missing_rate": 0.30,
            },
            {
                "regime_idx": 5,
                "name": "Regime 5: Severe Compound Shift + Return to Clean",
                "audio_noise_std": 1.0,
                "text_wer_drift": 0.40,
                "visual_blur_sigma": 1.5,
                "missing_rate": 0.50,
            },
        ]

    def get_regime_sequence(self) -> List[Dict[str, Any]]:
        """Return the sequence of test regimes."""
        return self.regimes

    def apply_regime(
        self,
        batch: Dict[str, Any],
        regime_idx: int,
        step_in_regime: int = 0,
        total_steps_in_regime: int = 100,
    ) -> Dict[str, Any]:
        """Apply corruptions and missingness corresponding to the given regime.
        
        Args:
            batch: Collated batch dictionary with 'features' and 'modality_mask'.
            regime_idx: 1-indexed regime indicator (1 to 5).
            step_in_regime: Current step index inside regime.
            total_steps_in_regime: Total steps in regime (for return-to-clean in regime 5).
        """
        # Shallow copy features dict
        corrupted_batch = {
            "features": {k: v.clone() for k, v in batch["features"].items()},
            "modality_mask": {k: v.clone() for k, v in batch["modality_mask"].items()},
            "labels": batch["labels"],
            "segment_ids": batch.get("segment_ids", []),
            "speaker_ids": batch.get("speaker_ids", []),
        }

        # Handle return-to-clean in latter half of Regime 5
        if regime_idx == 5 and step_in_regime >= (total_steps_in_regime // 2):
            # Return to clean (evaluates catastrophic forgetting)
            return corrupted_batch

        regime_cfg = self.regimes[min(regime_idx - 1, len(self.regimes) - 1)]

        # 1. Audio noise
        audio_noise = regime_cfg["audio_noise_std"]
        if audio_noise > 0.0 and "audio" in corrupted_batch["features"]:
            feat = corrupted_batch["features"]["audio"]
            noise = torch.randn_like(feat) * audio_noise
            corrupted_batch["features"]["audio"] = feat + noise
            if "prosody" in corrupted_batch["features"]:
                p_noise = torch.randn_like(corrupted_batch["features"]["prosody"]) * (audio_noise * 0.5)
                corrupted_batch["features"]["prosody"] += p_noise

        # 2. Text / ASR WER drift
        text_drift = regime_cfg["text_wer_drift"]
        if text_drift > 0.0 and "text" in corrupted_batch["features"]:
            feat = corrupted_batch["features"]["text"]
            # Random dimension masking + phonetic/synonym noise substitution in latent space
            mask = torch.rand_like(feat) < text_drift
            noise = torch.randn_like(feat) * 0.8
            corrupted_batch["features"]["text"] = torch.where(mask, feat * 0.2 + noise, feat)

        # 3. Visual corruption (blur & occlusion)
        v_blur = regime_cfg["visual_blur_sigma"]
        if v_blur > 0.0 and "visual" in corrupted_batch["features"]:
            feat = corrupted_batch["features"]["visual"]
            # Gaussian feature smoothing simulation + random spatial/channel occlusion
            occlusion_mask = torch.rand_like(feat) < 0.25
            noise = torch.randn_like(feat) * v_blur
            corrupted_batch["features"]["visual"] = torch.where(occlusion_mask, torch.zeros_like(feat), feat + noise)

        # 4. Dynamic missing modality dropping
        missing_rate = regime_cfg["missing_rate"]
        if missing_rate > 0.0:
            corrupted_batch = self._apply_missing_modality(corrupted_batch, missing_rate)

        return corrupted_batch

    def _apply_missing_modality(self, batch: Dict[str, Any], missing_rate: float) -> Dict[str, Any]:
        """Stochastically drop modalities per sample while ensuring at least 1 remains available."""
        bsz = batch["labels"].shape[0]
        mods = list(batch["features"].keys())

        for i in range(bsz):
            dropped = []
            for m in mods:
                if self.rng.rand() < missing_rate:
                    dropped.append(m)

            # Ensure non-empty set of modalities (at least 1 must remain active)
            if len(dropped) == len(mods):
                keep = self.rng.choice(mods)
                dropped.remove(keep)

            for m in dropped:
                batch["features"][m][i].zero_()
                batch["modality_mask"][m][i] = False

        return batch
