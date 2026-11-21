"""
Continual Compound Shift Stream Generator for Tri-SACTA (Protocol V2, AUD-16).

Implements:
- Physically-grounded SNR-calibrated audio noise.
- Structured ASR word error rate (WER) latent drift.
- Spatial/feature visual occlusion and Gaussian blur simulation.
- Correlated sensor dropout (e.g. structured audio/prosody joint missingness).
- Support for both 5-regime legacy stream and 6-regime Protocol V2 stream with explicit R6 Recovery.
- 100% deterministic reproducibility via dedicated torch.Generator and numpy RandomState.
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
    """Applies streaming compound distribution shifts across sequential test regimes."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, seed: int = 42, use_protocol_v2: bool = False):
        self.config = config or {}
        self.seed = seed
        self.use_protocol_v2 = use_protocol_v2
        self.rng = np.random.RandomState(seed)
        self.torch_gen = torch.Generator()
        self.torch_gen.manual_seed(seed)
        self.regimes = self._init_regime_definitions()

    def _init_regime_definitions(self) -> List[Dict[str, Any]]:
        base_regimes = [
            {
                "regime_idx": 1,
                "name": "Regime 1: Clean Baseline",
                "audio_snr_db": None,  # Clean (infinite SNR)
                "audio_noise_std": 0.0,
                "text_wer_drift": 0.0,
                "visual_blur_sigma": 0.0,
                "visual_occlusion_rate": 0.0,
                "missing_rate": 0.0,
            },
            {
                "regime_idx": 2,
                "name": "Regime 2: Audio Degradation (SNR ~5dB)",
                "audio_snr_db": 5.0,  # 5 dB SNR degradation
                "audio_noise_std": 0.8,
                "text_wer_drift": 0.0,
                "visual_blur_sigma": 0.0,
                "visual_occlusion_rate": 0.0,
                "missing_rate": 0.0,
            },
            {
                "regime_idx": 3,
                "name": "Regime 3: ASR Text Drift (WER ~35%)",
                "audio_snr_db": None,
                "audio_noise_std": 0.0,
                "text_wer_drift": 0.35,  # 35% simulated ASR token perturbation
                "visual_blur_sigma": 0.0,
                "visual_occlusion_rate": 0.0,
                "missing_rate": 0.0,
            },
            {
                "regime_idx": 4,
                "name": "Regime 4: Visual Corruption + 30% Missing",
                "audio_snr_db": None,
                "audio_noise_std": 0.0,
                "text_wer_drift": 0.0,
                "visual_blur_sigma": 1.2,
                "visual_occlusion_rate": 0.25,
                "missing_rate": 0.30,
            },
        ]

        if self.use_protocol_v2:
            # Protocol V2: R5 is Pure Severe Compound Shift; R6 is explicit Clean Recovery
            base_regimes.extend([
                {
                    "regime_idx": 5,
                    "name": "Regime 5: Severe Compound Shift",
                    "audio_snr_db": 0.0,  # 0 dB SNR severe noise
                    "audio_noise_std": 1.0,
                    "text_wer_drift": 0.40,
                    "visual_blur_sigma": 1.5,
                    "visual_occlusion_rate": 0.35,
                    "missing_rate": 0.50,
                },
                {
                    "regime_idx": 6,
                    "name": "Regime 6: Return-to-Clean Recovery",
                    "audio_snr_db": None,
                    "audio_noise_std": 0.0,
                    "text_wer_drift": 0.0,
                    "visual_blur_sigma": 0.0,
                    "visual_occlusion_rate": 0.0,
                    "missing_rate": 0.0,
                },
            ])
        else:
            # Legacy 5-Regime: R5 is Compound Shift + internal return to clean in second half
            base_regimes.append({
                "regime_idx": 5,
                "name": "Regime 5: Severe Compound Shift + Return to Clean",
                "audio_snr_db": 0.0,
                "audio_noise_std": 1.0,
                "text_wer_drift": 0.40,
                "visual_blur_sigma": 1.5,
                "visual_occlusion_rate": 0.35,
                "missing_rate": 0.50,
            })

        return base_regimes

    def get_regime_sequence(self) -> List[Dict[str, Any]]:
        """Return the ordered list of test regimes."""
        return self.regimes

    def apply_regime(
        self,
        batch: Dict[str, Any],
        regime_idx: int,
        step_in_regime: int = 0,
        total_steps_in_regime: int = 100,
    ) -> Dict[str, Any]:
        """Apply corruptions and missingness corresponding to the given regime index."""
        corrupted_batch = {
            "features": {k: v.clone() for k, v in batch["features"].items()},
            "modality_mask": {k: v.clone() for k, v in batch["modality_mask"].items()},
            "labels": batch["labels"],
            "segment_ids": batch.get("segment_ids", []),
            "speaker_ids": batch.get("speaker_ids", []),
        }

        # For legacy 5-regime mode: handle return-to-clean in second half of Regime 5
        if not self.use_protocol_v2 and regime_idx == 5 and step_in_regime >= (total_steps_in_regime // 2):
            return corrupted_batch

        reg_idx = min(max(1, regime_idx) - 1, len(self.regimes) - 1)
        regime_cfg = self.regimes[reg_idx]

        # 1. Audio noise (calibrated SNR or standard deviation)
        if "audio" in corrupted_batch["features"]:
            feat = corrupted_batch["features"]["audio"]
            snr_db = regime_cfg.get("audio_snr_db")
            if snr_db is not None:
                # Calibrated SNR: P_noise = P_signal * 10^(-SNR / 10)
                signal_power = torch.mean(feat ** 2, dim=-1, keepdim=True).clamp(min=1e-6)
                noise_power = signal_power * (10.0 ** (-snr_db / 10.0))
                noise_sigma = torch.sqrt(noise_power)
                noise = torch.randn(feat.shape, generator=self.torch_gen, device=feat.device, dtype=feat.dtype) * noise_sigma
                corrupted_batch["features"]["audio"] = feat + noise
                if "prosody" in corrupted_batch["features"]:
                    p_feat = corrupted_batch["features"]["prosody"]
                    p_noise = torch.randn(p_feat.shape, generator=self.torch_gen, device=p_feat.device, dtype=p_feat.dtype) * (noise_sigma * 0.5)
                    corrupted_batch["features"]["prosody"] = p_feat + p_noise
            elif regime_cfg.get("audio_noise_std", 0.0) > 0.0:
                noise_std = regime_cfg["audio_noise_std"]
                noise = torch.randn(feat.shape, generator=self.torch_gen, device=feat.device, dtype=feat.dtype) * noise_std
                corrupted_batch["features"]["audio"] = feat + noise
                if "prosody" in corrupted_batch["features"]:
                    p_feat = corrupted_batch["features"]["prosody"]
                    p_noise = torch.randn(p_feat.shape, generator=self.torch_gen, device=p_feat.device, dtype=p_feat.dtype) * (noise_std * 0.5)
                    corrupted_batch["features"]["prosody"] = p_feat + p_noise

        # 2. Text / ASR WER drift
        text_drift = regime_cfg.get("text_wer_drift", 0.0)
        if text_drift > 0.0 and "text" in corrupted_batch["features"]:
            feat = corrupted_batch["features"]["text"]
            # Generate deterministic mask for perturbed dimensions
            rand_u = torch.rand(feat.shape, generator=self.torch_gen, device=feat.device)
            mask = rand_u < text_drift
            noise = torch.randn(feat.shape, generator=self.torch_gen, device=feat.device, dtype=feat.dtype) * 0.8
            corrupted_batch["features"]["text"] = torch.where(mask, feat * 0.2 + noise, feat)

        # 3. Visual corruption (blur & occlusion)
        v_blur = regime_cfg.get("visual_blur_sigma", 0.0)
        v_occ = regime_cfg.get("visual_occlusion_rate", 0.0)
        if (v_blur > 0.0 or v_occ > 0.0) and "visual" in corrupted_batch["features"]:
            feat = corrupted_batch["features"]["visual"]
            if v_occ > 0.0:
                rand_occ = torch.rand(feat.shape, generator=self.torch_gen, device=feat.device)
                occlusion_mask = rand_occ < v_occ
            else:
                occlusion_mask = torch.zeros(feat.shape, dtype=torch.bool, device=feat.device)

            noise = torch.randn(feat.shape, generator=self.torch_gen, device=feat.device, dtype=feat.dtype) * v_blur
            corrupted_batch["features"]["visual"] = torch.where(occlusion_mask, torch.zeros_like(feat), feat + noise)

        # 4. Missing modality dropping with structured audio-prosody correlation option
        missing_rate = regime_cfg.get("missing_rate", 0.0)
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

            # Structured sensor dropout: if audio is dropped, prosody (derived from audio) is also dropped
            if "audio" in dropped and "prosody" in mods and "prosody" not in dropped:
                dropped.append("prosody")

            # Ensure at least 1 modality remains active
            if len(dropped) == len(mods):
                # When dropping all, choose a sensor other than prosody to keep if audio is missing
                eligible_keeps = [m for m in mods if m != "prosody"] if "audio" in dropped else mods
                keep = self.rng.choice(eligible_keeps)
                dropped.remove(keep)

            # Re-enforce invariant: prosody cannot remain active if audio was dropped
            if "audio" in dropped and "prosody" in mods and "prosody" not in dropped:
                dropped.append("prosody")

            for m in dropped:
                batch["features"][m][i].zero_()
                batch["modality_mask"][m][i] = False

        return batch
