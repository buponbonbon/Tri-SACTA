"""Smoke test for Tri-SACTA modules."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print("Testing imports...")
import torch
import yaml
from src.models.fusion_backbone import MultimodalFusionModel
from src.models.spectral_detector import SpectralShiftDetector
from src.adaptation.tri_sacta import TriSACTA
from src.adaptation.baselines_tta import TENT, SAR, MultimodalTTABaseline
from src.conformal.online_aci import OnlineACI
from src.conformal.static_cp import SplitConformalPredictor, MaskConditionalCP
from src.data.dataset import MultimodalDataset, get_dataloaders
from src.data.shift_generator import ShiftStreamGenerator
from src.utils.metrics import MetricsTracker, VRAMProfiler, LatencyProfiler
from src.utils.stats_test import paired_t_test, wilcoxon_test
print("[PASS] All imports successful!")

print("Testing Model initialization & parameter budget (< 2M)...")
with open("configs/mosei_mvp.yaml", "r") as f:
    config = yaml.safe_load(f)

model = MultimodalFusionModel(config)
total_params, trainable_params = model.get_total_params()
print(f"Total params: {total_params:,} | Trainable params: {trainable_params:,}")
assert trainable_params < 2_000_000, f"Trainable params {trainable_params} exceed 2M budget!"
print("[PASS] Model parameter budget satisfied (< 2M)!")

print("Testing Forward Pass...")
batch_size = 4
features = {
    "visual": torch.randn(batch_size, 512),
    "audio": torch.randn(batch_size, 768),
    "text": torch.randn(batch_size, 768),
    "prosody": torch.randn(batch_size, 88),
}
mask = {m: torch.ones(batch_size, dtype=torch.bool) for m in features}
out = model(features, modality_mask=mask)
assert "logits" in out
assert out["logits"].shape == (batch_size, config["model"]["num_classes"])
print(f"[PASS] Forward pass successful! Output logits shape: {out['logits'].shape}")

print("Testing Spectral Shift Detector...")
dims = {"visual": 512, "audio": 768, "text": 768, "prosody": 88}
detector = SpectralShiftDetector(feature_dims=dims)
synth_train = {m: torch.randn(20, d) for m, d in dims.items()}
detector.register_source_stats(synth_train)
shift_scores = detector.compute_shift_scores(features)
print(f"Shift scores: {shift_scores}")
print("[PASS] SpectralShiftDetector working properly!")

print("Testing Online ACI Conformal Prediction...")
aci = OnlineACI(num_classes=config["model"]["num_classes"], alpha=0.1, config=config["conformal"])
p_sets = aci.predict_set(out["logits"], "1111")
assert len(p_sets) == batch_size
print(f"Sample prediction sets: {p_sets}")
aci.update(out["logits"], torch.tensor([0, 1, 2, 3]), "1111")
print("[PASS] Online ACI conformal prediction sets and updates working!")

print("\n>>> ALL SMOKE TESTS PASSED PERFECTLY! <<<")
