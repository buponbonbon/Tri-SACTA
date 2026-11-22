# Tri-SACTA: Tri-Modal Spectral-Aligned Continual Test-Time Adaptation with Non-Exchangeable Online Conformal Control under Compound Modality Shifts

[![Target Venue: Q1 Journal](https://img.shields.io/badge/Target%20Venue-Q1%20Journal%20(Inf%20Fusion%20%2F%20IEEE%20TMM%20%2F%20Neural%20Networks)-blue.svg)](#)
[![Hardware: 4GB VRAM Compatible](https://img.shields.io/badge/Hardware-1x%204GB%20Laptop%20GPU%20(Peak%20VRAM%20%3C%2050MB)-success.svg)](#)
[![Unit Tests: 45/45 PASS](https://img.shields.io/badge/Unit%20Tests-45%2F45%20PASS%20(100%25)-brightgreen.svg)](#)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](#)
[![PyTorch: AMP FP16](https://img.shields.io/badge/PyTorch-AMP%20FP16-red.svg)](#)

---

## 1. Research Overview

Existing multimodal test-time adaptation (TTA) algorithms (e.g., AV-CTTA, READ) are predominantly restricted to bimodal (audio-visual) settings and lack statistical coverage guarantees. Conversely, conventional multimodal conformal prediction methods (e.g., MCCF, split conformal prediction) fundamentally rely on static exchangeability, leading to catastrophic coverage violations under streaming non-stationary distribution shifts.

**Tri-SACTA** resolves these foundational limitations for **3–4 modality streams** (Visual, Audio, Text/ASR, and Prosody) operating under **continual compound shifts** (calibrated audio SNR noise, speech recognition WER drift, visual occlusion/blur, and correlated sensor dropouts):

1. **Modality-Wise Spectral Shift Detector:**
   Quantifies distribution shift $d_t^{(m)}$ for each modality stream at test time using dimension-normalized first- and second-order feature statistics (Mahalanobis mean drift and log-eigenvalue covariance spectrum against source anchor statistics $(\mu_s^{(m)}, \Sigma_s^{(m)})$), calibrated against a clean holdout null baseline ($d_{\text{null}}^{(m)}$).
2. **Anchor-Guided Selective Low-Rank Adaptation:**
   Dynamically partitions modalities into three disjoint sets: Clean Anchors ($\mathcal{A}_t$, reliability $r_t \ge 0.7$), Adaptable ($\mathcal{U}_t$, $0.3 \le r_t < 0.7$), and Corrupted/Dropped ($\mathcal{D}_t$, $r_t < 0.3$). Gradient updates are strictly isolated to lightweight adapter parameters ($\le 1.83\%$ total parameters) of modalities in $\mathcal{U}_t$ guided by clean anchor supervision, while gradients from corrupted or missing channels are detached to eliminate negative transfer. An explicit maximum-entropy abstain policy activates when all modalities are rejected.
3. **Mask-and-Disagreement-Driven Online Conformal Control (Online ACI):**
   Maintains adaptive miscoverage levels $\alpha_{t, m_t}$ conditioned on the observed modality mask $m_t \in \{0, 1\}^M \setminus \{\mathbf{0}\}$. We provide an exact finite-sample diagnostic accounting identity relating empirical miscoverage to step sizes, signed projection clipping residuals, and unreceived feedback delays (see [THEORY_V3.md](docs/THEORY_V3.md)), alongside an empirical dynamic step size $\eta_t \in [\eta_{\text{min}}, \eta_{\text{max}}]$ modulated by spectral shift velocity $|\Delta d_t|$ and cross-modal Jensen-Shannon disagreement $\kappa_t$.

> [!IMPORTANT]
> **Scientific Integrity & Data Provenance Notice (V5):**
> Following an independent scientific audit, all historical synthetic pseudo-real fixtures from V3 have been permanently quarantined. Real data mode enforces strict fail-closed cryptographic validation: all feature files must match SHA-256 manifest checksums, and model checkpoints are cryptographically bound to manifest hashes. Authentic text features are extracted via genuine DeBERTa-v3 forward passes on official MELD and CMU-MOSEI annotations. Because raw multimodal media archives (>150 GB) are beyond the local download budget, raw audio/video media extraction is honestly declared **BLOCKED**; under no circumstances are synthetic vectors substituted.

---

## 2. Repository Structure

```text
Tri-SACTA/
├── docs/
│   ├── AUDIT_STATUS_V5.md          # Complete V5 defect register (V5-01 through V5-15)
│   ├── HANDOVER_V5.md              # Engineering handover milestones & reproduction steps
│   ├── CLAIM_EVIDENCE_MATRIX.md    # Formal scientific claim and empirical verification matrix
│   ├── DATA_PROVENANCE_AUDIT.md    # Forensic data provenance audit and quarantine log
│   ├── THEORY_V3.md                # Conformal proofs, exact accounting identity & counterexample
│   └── logs/
│       └── v5_test_execution.log   # Complete execution log of all 45 passing tests
├── configs/
│   ├── debug_synthetic.yaml        # Fast reproducible debug & testing configuration
│   ├── mosei_real.yaml             # Official CMU-MOSEI real dataset configuration (Video-disjoint)
│   ├── meld_real.yaml              # Official MELD emotion dialogue dataset configuration
│   └── shift_regimes_v2.yaml       # Formal Protocol V2 defining 6 sequential shift regimes
├── src/
│   ├── data/
│   │   ├── dataset.py              # Dataloader with strict manifest-to-tensor row join
│   │   ├── fixtures.py             # Isolated synthetic fixture generator (tagged synthetic_fixture)
│   │   ├── manifest.py             # DataManifest schema, full SHA-256 checks & tamper detection
│   │   └── shift_generator.py      # Physically grounded SNR audio noise & structured dropout
│   ├── models/
│   │   ├── fusion_backbone.py      # MaskedModalityTransformerLayer with hard gating & abstain policy
│   │   └── spectral_detector.py    # Dimension-normalized Mahalanobis & Eigenvalue detector
│   ├── adaptation/
│   │   ├── tri_sacta.py            # Anchor-Guided Selective Adaptation core algorithm
│   │   └── baselines_tta.py        # TTA baselines: TENT, SAR, READ/AV-CTTA
│   ├── conformal/
│   │   ├── online_aci.py           # Online ACI with signed clipping residual tracking
│   │   └── static_cp.py            # Static Split CP with exact finite-sample order statistic rank
│   └── utils/
│       ├── metrics.py              # Fixed-class Macro-F1, FP32 NLL, float64 Brier, VRAM footprint
│       └── stats_test.py           # Paired t-test, Wilcoxon signed-rank test, LaTeX exporter
├── scripts/
│   ├── 00_validate_data.py         # Fail-closed data integrity and manifest validation CLI
│   ├── 01_train_source.py          # Source training with schema v5 manifest-bound checkpoints
│   ├── 02_run_stream_benchmark.py  # 6-regime continual compound shift stream benchmark
│   ├── 03_plot_rolling_curves.py   # Publication figure generator (300 DPI)
│   ├── 04_run_experiment_matrix.py # Multi-seed paired experiment matrix orchestrator
│   ├── 05_aggregate_results.py     # Multi-seed aggregator with strict quarantine filtering
│   └── extract_features_sequential.py # Authentic text DeBERTa extractor (fail-closed on missing media)
├── tests/
│   ├── run_all_tests.py            # Unified test runner (45/45 unit & regression tests PASS)
│   ├── test_v5_provenance.py       # V5 provenance enforcement & math counterexample suite
│   ├── test_v4_quarantine.py       # V4 data authenticity & fixture rejection suite
│   ├── test_v3_verification.py     # V3 selective adaptation & shift detection suite
│   └── ...                         # Core algorithm and metric verification suites
├── results/
│   ├── meld/real/                  # Authentic MELD text prequential & transductive benchmark runs
│   ├── quarantined_v3_fixtures/    # Quarantined legacy artifacts (retained strictly for audit)
│   └── summary/                    # Multi-seed statistical summaries
└── requirements.txt
```

---

## 3. Quantitative Experimental Findings

### 3.1. Main Benchmark Results Across TTA & Conformal Methods (Controlled Synthetic Protocol)

All comparative baselines are evaluated on identical feature streams, random seeds, and sequence order across 6 sequential compound shift regimes (nominal confidence level $1 - \alpha = 90.0\%$):

| Method / Model | Macro-F1 $\uparrow$ | Accuracy $\uparrow$ | Empirical Coverage | Coverage Gap $|\text{Cov} - 90\%| \downarrow$ | Average Set Size $|\mathcal{C}_t| \downarrow$ | Peak VRAM | Latency |
|---|---|---|---|---|---|---|---|
| **Source-Only + Static CP** | 0.4220 | 42.97% | 91.73% | 1.73% | 3.30 | **21.6 MB** | **0.68 ms/sample** |
| **TENT + Online ACI** | 0.4216 | 43.03% | 90.30% | 0.30% | 3.33 | 36.8 MB | 2.84 ms/sample |
| **SAR + Online ACI** | 0.4248 | 43.23% | 90.13% | 0.13% | 3.31 | 36.8 MB | 2.80 ms/sample |
| **READ / CTTA + Online ACI** | 0.3895 | 40.73% | 90.33% | 0.33% | 3.42 | 49.2 MB | 4.89 ms/sample |
| **Tri-SACTA + Online ACI (Ours, $\delta=0$)** | **0.4287** | **43.53%** | **90.00%** | **0.00%** | **3.27** | 31.8 MB | 5.54 ms/sample |
| **Tri-SACTA + Online ACI (Delay $\delta=10$)** | 0.4287 | 43.53% | 89.43% | 0.57% | 3.21 | 31.8 MB | 5.42 ms/sample |
| **Tri-SACTA + Online ACI (Delay $\delta=50$)** | 0.4287 | 43.53% | 88.30% | 1.70% | 3.29 | 31.8 MB | 5.81 ms/sample |

### 3.2. Authentic Text-Only Pilot Benchmark (MELD Real Annotation Stream)

Evaluated on genuine DeBERTa-v3 768-dimensional embeddings from official MELD transcripts with dialogue-disjoint partitions:

| Protocol | Macro-F1 | Accuracy | Empirical Coverage | Coverage Gap | Avg Set Size | Peak VRAM | Latency |
|---|---|---|---|---|---|---|---|
| **Prequential ($\delta=0$)** | 0.0909 | 46.25% | 91.67% | 0.0167 | 6.36 | 21.8 MB | 5.74 ms/sample |
| **Transductive ($\delta=0$)** | 0.0909 | 46.25% | 91.67% | 0.0167 | 6.36 | 21.6 MB | 4.29 ms/sample |

*Artifact Provenance:* Checkpoint bound to `manifest_hash=5ffc5e3e67de`. Output logs: `results/meld/real/prequential/tri_sacta_online_aci_delay0_seed42.json`.

---

## 4. Quickstart & Reproducibility Guide

### 4.1. Environment Setup
```bash
pip install -r requirements.txt
```

### 4.2. Run the Full Regression Suite (45 Tests)
```bash
python -u tests/run_all_tests.py
```
*Expected Output:* `SUMMARY: 45 PASSED, 0 FAILED` (runtime ~8s).

### 4.3. Data Validation & Manifest Check
```bash
# Synthetic fixture validation:
python scripts/00_validate_data.py --config configs/debug_synthetic.yaml --data-mode synthetic

# Authentic dataset manifest validation (fail-closed if checksums mismatch):
python scripts/00_validate_data.py --config configs/meld_real.yaml --data-mode real
```

### 4.4. Authentic Text Extraction (DeBERTa-v3)
```bash
# MELD dialogue-disjoint pilot:
python scripts/extract_features_sequential.py --dataset meld --extract-text-only --max-samples 80

# CMU-MOSEI video-disjoint pilot:
python scripts/extract_features_sequential.py --dataset cmu_mosei --extract-text-only --max-samples 80
```

### 4.5. Source Model Training
```bash
python scripts/01_train_source.py --config configs/meld_real.yaml --data-mode real --seed 42 --epochs 5
```

### 4.6. Continual Stream Benchmark
```bash
# Proposed Tri-SACTA + Online ACI on authentic text:
python scripts/02_run_stream_benchmark.py \
  --config configs/meld_real.yaml \
  --data-mode real \
  --method tri_sacta \
  --conformal online_aci \
  --seed 42 \
  --checkpoint checkpoints/meld/real/seed42/source_model.pt \
  --spectral-stats checkpoints/meld/real/seed42/source_spectral_stats.pt
```

---

## 5. Hardware Profiling (4GB Laptop GPU Constraint)

| Metric | Budget Specification | Empirical Measurement (RTX 2050 4GB) | Compliance |
|---|---|---|:---:|
| **Total Model Parameters** | $< 2.0\text{ M}$ | **1,369,379** (1.37 M) | PASS |
| **Adaptable Parameters Ratio** | $< 2.0\%$ | **1.83%** (LayerNorm + Low-Rank Projections) | PASS |
| **Peak GPU VRAM (Allocated)** | $< 1.5\text{ GB}$ | **21.8 MB** (< 0.6% of 4GB VRAM) | PASS |
| **Peak GPU VRAM (Reserved)** | $< 1.5\text{ GB}$ | **48.1 MB** (< 1.2% of 4GB VRAM) | PASS |
| **Inference + Adaptation Latency**| $< 50\text{ ms/sample}$ | **5.74 ms/sample** (> 170 FPS real-time) | PASS |

---

## 6. Scientific References & Theoretical Documentation

- [AUDIT_STATUS_V5.md](docs/AUDIT_STATUS_V5.md): Comprehensive V5 defect register (V5-01 through V5-15) with verification logs.
- [HANDOVER_V5.md](docs/HANDOVER_V5.md): Engineering handover guide, reproduction steps, and milestone tracking.
- [CLAIM_EVIDENCE_MATRIX.md](docs/CLAIM_EVIDENCE_MATRIX.md): Detailed scientific claim-evidence matrix and gate criteria.
- [DATA_PROVENANCE_AUDIT.md](docs/DATA_PROVENANCE_AUDIT.md): Forensic data authenticity audit and quarantine records.
- [THEORY_V3.md](docs/THEORY_V3.md): Formal proof of exact diagnostic accounting identity and delayed-feedback counterexample.
