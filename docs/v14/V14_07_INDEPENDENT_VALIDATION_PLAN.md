# V14-07: Independent Validation Strategy for V15

- **Exposed Partitions (Retired from Fresh Confirmation):**
  - `train` (`16,327` utterances): predictor training.
  - `val` (`250` utterances, `29` videos) & `cal` (`250` utterances): V12–V14 development & diagnostics.
  - `test[0:500]`: historical V1–V13.1 test.
  - `eligible_holdout` (`4,131` utterances, `617` videos): **permanently exposed** in V13.2 Phase II — **prohibited** as fresh V15 confirmation.
- **Candidate Independent Datasets for V15 (Metadata Plan Only, Not Downloaded):**
  1. **CMU-MOSI** (`2,199` utterances, `93` videos, 3-class sentiment compatible, requires checking source video overlap against CMU-MOSEI before freezing).
  2. **IEMOCAP / MELD** (multimodal emotion classification, requires retraining lightweight MLP predictor banks on CPU/4GB VRAM — estimated `<15 min` on RTX 2050).
- **Hardware Feasibility:** Frozen MLP banks + lightweight calibrated logistic/MLP value estimators on cached features run on CPU (`<200 MB` RAM) with zero GPU bottleneck.
