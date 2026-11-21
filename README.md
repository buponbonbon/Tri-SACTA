# Tri-SACTA: Tri-Modal Spectral-Aligned Continual Test-Time Adaptation with Non-Exchangeable Online Conformal Control under Compound Modality Shifts

[![Target Venue: Q1 Journal](https://img.shields.io/badge/Target%20Venue-Q1%20Journal%20(Inf%20Fusion%20%2F%20IEEE%20TMM%20%2F%20Neural%20Networks)-blue.svg)](#)
[![Hardware: 4GB VRAM Compatible](https://img.shields.io/badge/Hardware-1x%204GB%20Laptop%20GPU%20(Peak%20VRAM%20%3C%2050MB)-success.svg)](#)
[![Unit Tests: 24/24 PASS](https://img.shields.io/badge/Unit%20Tests-24%2F24%20PASS%20(100%25)-brightgreen.svg)](#)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](#)
[![PyTorch: AMP FP16](https://img.shields.io/badge/PyTorch-AMP%20FP16-red.svg)](#)

---

## 1. Research Overview

Existing multimodal test-time adaptation (TTA) algorithms (e.g., AV-CTTA, READ) are predominantly restricted to bimodal (audio-visual) settings and lack statistical coverage guarantees. Conversely, conventional multimodal conformal prediction methods (e.g., MCCF, split conformal prediction) fundamentally rely on static exchangeability, leading to catastrophic coverage violations under streaming non-stationary distribution shifts.

**Tri-SACTA** resolves these foundational limitations for **3–4 modality streams** (Visual, Audio, Text/ASR, and Prosody) operating under **continual compound shifts** (calibrated audio SNR noise, speech recognition WER drift, visual occlusion/blur, and correlated sensor dropouts):

1. **Modality-Wise Spectral Shift Detector:**
   Quantifies distribution shift $d_t^{(m)}$ for each modality stream at test time using dimension-normalized first- and second-order feature statistics (Mahalanobis mean drift and log-eigenvalue covariance spectrum against source anchor statistics $(\mu_s^{(m)}, \Sigma_s^{(m)})$), calibrated against a clean holdout null baseline ($d_{\text{null}}^{(m)}$).
2. **Anchor-Guided Selective Low-Rank Adaptation:**
   Dynamically partitions modalities into three disjoint sets: Clean Anchors ($\mathcal{A}_t$, reliability $r_t \ge 0.7$), Adaptable ($\mathcal{U}_t$, $0.3 \le r_t < 0.7$), and Corrupted/Dropped ($\mathcal{D}_t$, $r_t < 0.3$). Gradient updates are strictly isolated to lightweight adapter parameters ($\le 1.83\%$ total parameters) of modalities in $\mathcal{U}_t$ guided by clean anchor supervision, while gradients from corrupted or missing channels are detached to eliminate negative transfer.
3. **Mask-and-Disagreement-Driven Online Conformal Control (Online ACI):**
   Maintains adaptive miscoverage levels $\alpha_{t, m_t}$ conditioned on the observed modality mask $m_t \in \{0, 1\}^M \setminus \{\mathbf{0}\}$. We provide a mathematically proven Fixed-Step Mask-Conditional Reference Controller guaranteeing exact finite-sample pathwise error tracking $|\bar{e}_{n_g} - \alpha| \le \frac{1+\eta}{\eta \cdot n_g}$ without exchangeability assumptions (see [THEORY_V3.md](file:///c:/Users/ASUS/Desktop/Tri-SACTA/docs/THEORY_V3.md)), alongside an empirical dynamic step size $\eta_t \in [\eta_{\text{min}}, \eta_{\text{max}}]$ modulated by spectral shift velocity $|\Delta d_t|$ and cross-modal Jensen-Shannon disagreement $\kappa_t$ for rapid transition tracking under delayed feedback ($\delta \in \{10, 50\}$ steps).

---

## 2. Repository Structure

```text
Tri-SACTA/
├── docs/
│   ├── AUDIT_STATUS.md             # Complete issue register tracking (AUD-01 to AUD-26)
│   ├── HANDOVER.md                 # Engineering handover milestones and verification report
│   ├── METHOD_SPEC_V2.md           # Formal mathematical specification of Tri-SACTA V2
│   ├── THEORY.md                   # Conformal proofs, asymptotic convergence & error bounds
│   └── EXPERIMENT_PROTOCOL.md      # Prequential evaluation protocols & regime definitions
├── configs/
│   ├── debug_synthetic.yaml        # Fast reproducible debug & testing configuration
│   ├── mosei_real.yaml             # Official CMU-MOSEI real dataset configuration (Speaker-disjoint)
│   ├── meld_real.yaml              # Official MELD emotion dialogue dataset configuration
│   └── shift_regimes_v2.yaml       # Formal Protocol V2 defining 6 sequential shift regimes
├── src/
│   ├── data/
│   │   ├── dataset.py              # Speaker-disjoint dataloader with explicit --data-mode
│   │   ├── manifest.py             # DataManifest schema, checksums, and leakage validator
│   │   └── shift_generator.py      # Physically grounded SNR audio noise & structured dropout
│   ├── models/
│   │   ├── fusion_backbone.py      # MaskedModalityTransformerLayer with key-padding mask
│   │   └── spectral_detector.py    # Dimension-normalized Mahalanobis & Eigenvalue detector
│   ├── adaptation/
│   │   ├── tri_sacta.py            # Anchor-Guided Selective Adaptation core algorithm
│   │   └── baselines_tta.py        # TTA baselines: TENT, SAR, READ/AV-CTTA
│   ├── conformal/
│   │   ├── online_aci.py           # Online ACI with PredictionRecord & rare-mask shrinkage
│   │   └── static_cp.py            # Static Split CP with exact finite-sample order statistic rank
│   └── utils/
│       ├── metrics.py              # Fixed-class Macro-F1, FP32 NLL, float64 Brier, VRAM footprint
│       └── stats_test.py           # Paired t-test, Wilcoxon signed-rank test, LaTeX exporter
├── scripts/
│   ├── 00_validate_data.py         # Fail-closed data integrity and manifest validation CLI
│   ├── 01_train_source.py          # Source training with supervised auxiliary heads
│   ├── 02_run_stream_benchmark.py  # 5-to-6 regime continual compound shift stream benchmark
│   ├── 03_plot_rolling_curves.py   # Publication figure generator (300 DPI)
│   ├── 04_run_experiment_matrix.py # Multi-seed paired experiment matrix orchestrator
│   ├── 05_aggregate_results.py     # Multi-seed statistical aggregator & LaTeX table generator
│   └── extract_features_sequential.py # Stage 1 sequential FP16 extractor (<1.5GB VRAM)
├── tests/
│   ├── run_all_tests.py            # Unified standalone test runner (24/24 unit tests PASS)
│   ├── test_spectral.py            # Spectral shift detector unit tests (T01 - T04)
│   ├── test_masked_fusion.py       # Masked fusion & mask invariance tests (T05 - T07)
│   ├── test_selective_adaptation.py# Selective parameter freezing & gradient gating (T08, T11)
│   ├── test_conformal.py           # Online ACI & finite-sample order statistics (T12, T13, T15)
│   ├── test_data_integrity.py      # Fail-closed loader & deterministic seeding (T19, T21, T23)
│   ├── test_metrics_rigor.py       # Metrics precision & memory footprint tests (T26, T29)
│   ├── test_shift_generator.py     # SNR noise power & structured dropout tests (T25)
│   └── test_feedback_delay.py      # FIFO delayed queue & rare-mask shrinkage tests (T14, T18)
├── results/
│   ├── figures/                    # High-resolution publication figures (300 DPI)
│   ├── legacy/                     # Preserved pre-audit experimental run records
│   ├── summary/                    # Aggregated multi-seed statistics (Markdown & LaTeX tables)
│   └── *.json                      # Detailed per-run benchmark logs and telemetry
└── requirements.txt
```

---

## 3. Quantitative Experimental Findings

All comparative benchmarks are evaluated on identical feature sequences, random seeds, and stream orders across 5 sequential compound shift regimes (600 test utterances, nominal confidence level $1 - \alpha = 90.0\%$):

### 3.1. Main Benchmark Results Across TTA & Conformal Methods

| Method / Model | Macro-F1 $\uparrow$ | Accuracy $\uparrow$ | Empirical Coverage | Coverage Gap $|\text{Cov} - 90\%| \downarrow$ | Average Set Size $|\mathcal{C}_t| \downarrow$ | Peak VRAM | Latency |
|---|---|---|---|---|---|---|---|
| **Source-Only + Static CP** | 0.4220 | 42.97% | 91.73% | 1.73% | 3.30 | **21.6 MB** | **0.68 ms/sample** |
| **TENT + Online ACI** | 0.4216 | 43.03% | 90.30% | 0.30% | 3.33 | 36.8 MB | 2.84 ms/sample |
| **SAR + Online ACI** | 0.4248 | 43.23% | 90.13% | 0.13% | 3.31 | 36.8 MB | 2.80 ms/sample |
| **READ / CTTA + Online ACI** | 0.3895 | 40.73% | 90.33% | 0.33% | 3.42 | 49.2 MB | 4.89 ms/sample |
| **Tri-SACTA + Online ACI (Ours, $\delta=0$)** | **0.4287** | **43.53%** | **90.00%** | **0.00%** | **3.27** | 31.8 MB | 5.54 ms/sample |
| **Tri-SACTA + Online ACI (Delay $\delta=10$)** | 0.4287 | 43.53% | 89.43% | 0.57% | 3.21 | 31.8 MB | 5.42 ms/sample |
| **Tri-SACTA + Online ACI (Delay $\delta=50$)** | 0.4287 | 43.53% | 88.30% | 1.70% | 3.29 | 31.8 MB | 5.81 ms/sample |

### 3.2. Per-Regime Macro-F1 Breakdown (Continual Compound Shift Stream)

| Method | R1: Clean | R2: Audio Shift (SNR 5dB) | R3: Text Drift (WER 35%) | R4: Visual+Missing (30%) | R5: Severe Compound |
|---|---|---|---|---|---|
| **Source-Only** | 0.4887 | 0.4473 | 0.4483 | 0.3314 | 0.3853 |
| **TENT** | 0.4914 | 0.4664 | 0.4484 | 0.3304 | 0.3601 |
| **SAR** | 0.4901 | 0.4678 | 0.4577 | 0.3349 | 0.3645 |
| **READ / CTTA** | 0.4721 | 0.4251 | 0.4087 | 0.3006 | 0.3324 |
| **Tri-SACTA (Ours)** | 0.4845 | **0.4749** (+2.76%) | **0.4672** (+1.89%) | **0.3439** (+4.33% vs READ) | 0.3651 |

### 3.3. Scientific Takeaways
1. **Resistance to Negative Transfer:** Under Regime 4 (severe visual degradation + 30% modality dropout), standard multimodal CTTA (READ) suffers severe degradation down to Macro-F1 = 0.3006 by forcing consistency across corrupted modalities. Tri-SACTA detaches corrupted channels into $\mathcal{D}_t$ and leverages reliable anchors, sustaining Macro-F1 at 0.3439 (+4.33% over READ).
2. **Guaranteed Statistical Validity:** Static Split CP under-covers under compound shifts (Regime 4 coverage = 88.67%, Regime 5 = 87.83%), violating the 90% target due to non-exchangeability. In contrast, Tri-SACTA's Online ACI achieves exact empirical coverage of **90.00%** with a coverage gap of **0.0000%**.
3. **Delayed Feedback Robustness:** Evaluating delayed feedback using immutable historical prediction sets demonstrates graceful degradation: Coverage remains at **89.43%** under delay $\delta=10$, and **88.30%** under delay $\delta=50$.

---

## 4. Quickstart & Reproducibility Guide

### 4.1. Environment Setup
```bash
pip install -r requirements.txt
```

### 4.2. Run the Full Unit Test Suite (24 Tests)
```bash
python tests/run_all_tests.py
```
*Verification:* Validates mathematical mask invariance ($\Delta y = 0.0000$), selective parameter freezing ($\Delta \theta = 0$), finite-sample quantile ranks, calibrated SNR noise, and bounded step size dynamics.

### 4.3. Data Validation
```bash
# Synthetic debug verification (runs immediately):
python scripts/00_validate_data.py --config configs/debug_synthetic.yaml --data-mode synthetic

# Real dataset validation (fail-closed if missing files):
python scripts/00_validate_data.py --config configs/mosei_real.yaml --data-mode real
```

### 4.4. Source Model Training
```bash
python scripts/01_train_source.py --config configs/debug_synthetic.yaml --epochs 10 --seed 42 --data-mode synthetic
```
*Output:* Checkpoint saved to `checkpoints/source_model_synthetic_seed42.pt` (Val Macro-F1 = 50.01%) and anchor statistics to `checkpoints/source_spectral_stats_synthetic_seed42.pt`.

### 4.5. Run Continual Stream Benchmark
```bash
# Tri-SACTA + Online ACI (Proposed):
python scripts/02_run_stream_benchmark.py --method tri_sacta --conformal online_aci --seed 42 --data-mode synthetic

# Delayed feedback ablations (delta = 10, 50):
python scripts/02_run_stream_benchmark.py --method tri_sacta --conformal online_aci --delay 10 --seed 42 --data-mode synthetic
python scripts/02_run_stream_benchmark.py --method tri_sacta --conformal online_aci --delay 50 --seed 42 --data-mode synthetic

# Baselines:
python scripts/02_run_stream_benchmark.py --method source_only --conformal static_cp --seed 42 --data-mode synthetic
python scripts/02_run_stream_benchmark.py --method tent --conformal online_aci --seed 42 --data-mode synthetic
python scripts/02_run_stream_benchmark.py --method sar --conformal online_aci --seed 42 --data-mode synthetic
python scripts/02_run_stream_benchmark.py --method read --conformal online_aci --seed 42 --data-mode synthetic
```

### 4.6. Generate Publication Figures & LaTeX Tables
```bash
# Generate 3 high-resolution publication figures (300 DPI) in results/figures/:
python scripts/03_plot_rolling_curves.py

# Generate multi-seed statistical summary Markdown & LaTeX table in results/summary/:
python scripts/05_aggregate_results.py --input_dir results --output_dir results/summary --data-mode synthetic
```

---

## 5. Hardware Profiling (4GB Laptop GPU Constraint)

| Metric | Budget Specification | Empirical Measurement (RTX 2050 4GB) | Compliance |
|---|---|---|:---:|
| **Total Model Parameters** | $< 2.0\text{ M}$ | **1,369,379** (1.37 M) | PASS |
| **Adaptable Parameters Ratio** | $< 2.0\%$ | **1.83%** (LayerNorm + Low-Rank Projections) | PASS |
| **Peak GPU VRAM (Allocated)** | $< 1.5\text{ GB}$ | **31.8 MB** (< 1.0% total 4GB VRAM) | PASS |
| **Peak GPU VRAM (Reserved)** | $< 1.5\text{ GB}$ | **48.0 MB** (< 1.2% total 4GB VRAM) | PASS |
| **Inference + Adaptation Latency**| $< 50\text{ ms/sample}$ | **5.54 ms/sample** (> 180 FPS real-time) | PASS |

---

## 6. Scientific References & Theoretical Documentation

- [METHOD_SPEC_V2.md](docs/METHOD_SPEC_V2.md): Comprehensive mathematical formulation and parameter policies of Tri-SACTA V2.
- [THEORY.md](docs/THEORY.md): Conformal prediction theoretical framework, asymptotic convergence proofs, and delay bounds.
- [EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md): Rigorous prequential evaluation protocols and 6-regime benchmark specification.
- [AUDIT_STATUS.md](docs/AUDIT_STATUS.md): Complete audit issue register (AUD-01 through AUD-26) with concrete experimental evidence.
- [HANDOVER.md](docs/HANDOVER.md): Detailed engineering handover and milestone documentation.
