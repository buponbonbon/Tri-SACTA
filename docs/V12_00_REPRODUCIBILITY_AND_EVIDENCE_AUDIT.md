# V12.00: Reproducibility, Development-Exposure, and Evidence Inventory

**Author:** Antigravity 3.8 Systems Research & Verification Lead  
**Date:** 2026-11-23  
**Base Commit:** `717d5dfdc1f4831dd481d04a8b7bd7edbe528f04` (`origin/main`)  
**Branch:** `research/v12-systems-contribution-validation`  
**License:** MIT License  

---

## 1. Executive Summary & Research Scope Authorization

In V11, the project reached a formal `FREEZE_TRI_SACTA` decision after proving that test-time adaptation (TTA) via low-rank cross-attention did not produce measurable gains over a static reliability-weighted fusion baseline on CMU-MOSEI.

On 2026-11-23, the repository author officially reopened the project with a **newly authorized research scope**:
- **Pivot:** Reframe Tri-SACTA from an *algorithmic TTA claim* to a **multimodal systems, architecture, and inference orchestration contribution**.
- **Target Venues:** SCImago Q1/Q2 in 2023–2025 (or Q1 in 2025) with preference for **$0 mandatory publication fees** (e.g. *Multimedia Tools and Applications* [MTAP], *Multimedia Systems* [MMSYS], *Computing*).
- **Core Contribution:** The **Budgeted Reliability-Aware Modality Orchestrator (BRMO)** — an inference-time routing engine that dynamically trades off classification accuracy, prediction risk, end-to-end latency, and sensor invocation cost under real-world modality corruption and missingness.

---

## 2. Inventory of Physical Assets & Cache Checksums

All experiments in V12 operate on the authentic CMU-MOSEI feature cache verified in V10 and V11:
- **Audio:** 74-dimensional COVAREP acoustic features.
- **Visual:** 35-dimensional Facet action unit / pose features.
- **Text:** 768-dimensional mDeBERTa-v3-Base contextual embeddings (last hidden state mean pooled).
- **Splits:** 100% video-disjoint and speaker-disjoint across `train` (1,000 samples, 112 videos), `val` (250 samples, 29 videos), `cal` (250 samples, 29 videos), and `test` (500 samples, 57 videos).

### Cryptographic Checksums (SHA-256)
Refer to [results/v12_systems/manifest.json](file:///c:/Users/ASUS/Desktop/Tri-SACTA/results/v12_systems/manifest.json) for full cryptographic hashes:
- `train/text.pt`: `626f2cf800332a266f398b31d85f425702306bf3ffa7ffd46a6980431b476f11`
- `train/audio.pt`: `44a9fde3dd4077c82476febad30d888cbb58a6a938424d695478b6abdd43c628`
- `train/visual.pt`: `f1c12869cce46e0f573c7644aeecbd999140e078c553dcc753d4f52427541d47`
- `train/labels.pt`: `4578b77626938dc85b73d61b36585bf2ee9bcf6c2306ce06497fe91ae6da308c`

---

## 3. Rectification of V11 Comparison Flaw (Priority 0)

In V11 Phase B, an unfair comparison was identified:
- The Text-only Logistic Regression was trained **directly on 3-class sentiment labels** ($s < 0, s = 0, s > 0$), achieving 57.2% accuracy and 0.5425 Macro-F1 on `val`.
- In contrast, the 1.37M parameter `MultimodalFusionModel` was **not retrained on 3 classes**. Its historical 7-class logits were merely mapped: $\{0,1,2\} \to \text{Neg}, \{3\} \to \text{Neu}, \{4,5,6\} \to \text{Pos}$, which artificially constrained its decision boundaries, resulting in 45.6% accuracy and 0.4101 Macro-F1.
- Furthermore, early concatenation Logistic Regression (877D) directly trained on 3 classes had achieved 56.8% accuracy and **0.5477 Macro-F1** (outperforming text-only Macro-F1).

**V12 Requirement:** In Stage 1, `MultimodalFusionModel` will be directly reinitialized and trained with a 3-class output head (`num_classes: 3`) across 3 independent seeds (`42, 100, 2026`) using identical balanced cross-entropy and group-disjoint validation.

---

## 4. Test Set Development-Exposure Quarantine

- The 500-sample `test` split was exposed during exploratory TTA iterations in V7–V9.
- **Strict Quarantine:** Under no circumstances shall hyperparameters, threshold selections, or orchestration policies in V12 be tuned or optimized against `test`.
- All model selection, early stopping, and routing threshold calibration take place strictly on `train`, `val`, and `cal`.

---

## 5. Hardware & Runtime Environment

- **Operating System:** Windows 10 / 11 (build 26200)
- **CPU RAM:** 16.45 GB
- **GPU:** NVIDIA GeForce RTX 2050 Laptop GPU (4.29 GB VRAM, CUDA 12.1, PyTorch 2.5.1+cu121)
- **Constraint Compliance:** Model memory footprint is strictly budgeted to remain $< 100$ MB VRAM, ensuring flawless reproducibility on consumer-grade laptop hardware.
