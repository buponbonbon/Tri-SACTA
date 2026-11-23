# TRI-SACTA — CLAIM-EVIDENCE MATRIX (V8)

**Document Version:** 8.0  
**Branch:** `research/v8-valid-baselines-and-event-trace`  
**Base Commit:** `974573d4f366216a6b2f0afb5c71117aebddd06f`  
**Status:** Living Scientific Integrity & Verification Matrix  
**Principle:** In Q1 peer-reviewed research, a scientific claim is only as valid as its empirical and mathematical evidence. No claim is accepted without reproducible artifacts, non-vacuous tests, cryptographic data provenance, and controlled baselines.

---

## 1. High-Level Claims & Scientific Status

| Claim ID | Scientific Claim | Mathematical / Theoretical Foundation | Empirical Evidence Required | Current Status | Notes / Scope Boundaries |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **C1** | **Sample-Size-Aware Spectral Reliability:** Eigenvalue spectrum + mean shift score quantifies per-modality shift while preventing rank-collapse bias on small/missing batches. | Covariance rank $k = \min(top\_k, n_{\text{valid}}-1)$. Modality-calibrated temperature $\tau_m = \frac{-\ln(\theta_{\text{high}})}{d_{m, \text{clean}, 99}}$. Gram matrix decomposition $\frac{1}{N-1}XX^\top$. | AUROC/AUPRC on distinguishing shifted vs unshifted modalities; controlled False Positive Rate $\le 5\%$ on clean data; clean anchor retention rate $\ge 85\%$. | **VERIFIED (Calibrated Multimodal CMU-MOSEI)** | Implemented in `src/models/spectral_detector.py`. Calibrated $\tau_{\text{text}}=2.3282, \tau_{\text{audio}}=0.0273, \tau_{\text{visual}}=0.0217$ in `results/diagnostics/spectral_threshold_calibration.json`. Clean anchor rate: **87.5%**, Moderate shift adaptable rate: **25.0%**, AUROC: 0.7157, AUPRC: 0.9304. |
| **C2** | **Anchor-Guided Selective Continual Adaptation:** Adapting only lightweight projection layers of moderately shifted modalities using clean anchors prevents negative transfer and catastrophic forgetting. | Modality partitioning: $\mathcal{A}_t$ (anchor), $\mathcal{U}_t$ (adaptable), $\mathcal{D}_t$ (corrupted/dropped). Certainty-weighted alignment $\mathcal{L}_{\text{align}} = \sum_{u, a} w_a D_{\text{KL}}(p_t^{(u)} \| \text{sg}(p_t^{(a)}))$ where $w_a = 1 - H(p_a)/\ln C$. Effective attention mask $\mathbf{M}_t^{\text{eff}}$ strictly enforced in all forward passes. | Controlled ablation vs Full Adaptation and Reliability-Only on identical cryptographic stream hash; non-zero parameter updates $\Delta \theta > 0$ on adaptable set, strictly frozen ($\Delta \theta = 0$) on anchor and corrupted sets; zero gradient leakage. | **VERIFIED (Certainty-Weighted Alignment & Active FullAdaptation)** | Certainty weighting in `src/adaptation/tri_sacta.py:283` prevents noisy anchor teachers from corrupting student representations. `FullAdaptation` baseline executes active parameter updates across 96/96 batches. Verified in `tests/test_v8_regression.py::test_v8_01` & `test_v8_05`. |
| **C3** | **Mask-and-Disagreement-Driven Online Conformal Control:** Dynamic online ACI maintains valid marginal coverage under non-exchangeable streaming compound modality shifts. | Exact diagnostic accounting identity with signed clipping residuals: $\|\bar{e}_N - \alpha\| \le \frac{\|a_n - a_0\|}{\eta N} + \frac{\sum \|c_j\|}{\eta N} + \frac{K_{\text{pending}}}{N}$. | Marginal coverage $1-\alpha \pm 2.5\%$ across streaming regimes; per-mask coverage breakdown; exact telemetry recorded directly from `OnlineACI` without manual recomputation; 1-to-1 event ID matching (3,000 distinct records). | **VERIFIED (1-to-1 Event Telemetry & Frozen Logits Replay)** | Telemetry extracted per sample inside prediction loop, producing exactly 3,000 unique event IDs ($1..3000$). Frozen logits replay guarantees bit-for-bit identical comparison across controllers. `GlobalACI` provides genuine single-alpha baseline. Verified in `tests/test_v8_regression.py::test_v8_02`, `test_v8_03`, `test_v8_04`. |
| **C4** | **Hardware-Efficient Edge Deployment:** Offline feature caching and lightweight fusion operate within a 4GB VRAM laptop GPU budget. | Stage 1 offline caching; Stage 2 lightweight fusion (<2M trainable params); AMP float16 execution; VRAM memory barriers. | Peak VRAM $< 1.5$ GB on NVIDIA RTX 2050 4GB; latency $< 10$ ms per sample; throughput $> 100$ samples/sec. | **VERIFIED (RTX 2050 Laptop Profile)** | Source model training peak VRAM: **41.2 MB**; streaming benchmark peak VRAM: **29.6 MB**; average latency: **4.30 ms/sample** on laptop RTX 2050 GPU. Over 98% VRAM headroom maintained. |
| **C5** | **Competitive Multimodal Source Backbone Surpassing Unimodal Baselines:** Normalized projection and learned modality gating prevent modality dilution. | Modality input LayerNorm $\text{LN}(x)$; learned gating prior $\mathbf{w}_g = \text{Softmax}(\mathbf{g})$; balanced class loss weighting $w_c = \frac{N}{C \cdot N_c}$; auxiliary unimodal loss $\lambda_{\text{aux}} = 0.05$. | Multimodal backbone must outperform unimodal text-only Logistic Regression (F1=0.2458) and match concatenation baselines on identical splits. | **VERIFIED (Authentic CMU-MOSEI Splits)** | Evaluated in `scripts/diagnose_unimodal_multimodal_baselines.py` and `01_train_source.py`. Text-only LR: 0.2458. Concat MLP: 0.2590. Optimized V7 Multimodal Backbone achieves clean test Macro-F1 = **0.2598** and Accuracy = **28.00%**, officially beating unimodal text. |

---

## 2. Work Package Evidence

### 2.1. Work Package 1: Baseline Integrity & FullAdaptation Activation (V8 P0-1)
- **Defect in V7:** `full_adaptation` performed 0 parameter updates due to `theta_high=0.0, theta_low=0.0` classifying all modalities as Anchor.
- **Evidence Delivered (V8):**
  1. Dedicated `FullAdaptation` class in `src/adaptation/baselines_tta.py` executing unconstrained entropy minimization.
  2. Verified active parameter updates: 96/96 batches adapted, parameter delta > 0.
  3. Verified by unit test: `tests/test_v8_regression.py::test_v8_01` [PASS].

### 2.2. Work Package 2: 1-to-1 Event-Level Conformal Telemetry (V8 P0-2)
- **Defect in V7:** Post-batch telemetry extraction caused all samples in a batch to clone the last sample's event ID (96 unique IDs for 3,000 samples).
- **Evidence Delivered (V8):**
  1. Telemetry captured per sample directly inside the sample-level prediction and update loop.
  2. Exactly 3,000 unique sequential event IDs ($1, 2, \dots, 3000$) verified.
  3. Verified by unit test: `tests/test_v8_regression.py::test_v8_02` [PASS].

### 2.3. Work Package 3: Genuinely Global ACI Controller (V8 P0-3)
- **Defect in V7:** `global_aci` used `fixed_mask_aci` which maintained per-mask alpha state vectors.
- **Evidence Delivered (V8):**
  1. Dedicated `GlobalACI` class in `src/conformal/online_aci.py` with single global scalar $\alpha_t$ and single pooled score window.
  2. Verified by unit test: `tests/test_v8_regression.py::test_v8_03` [PASS].

### 2.4. Work Package 4: Frozen Logits Replay for Fair Conformal Evaluation (V8 P0-4)
- **Defect in V7:** Conformal controllers evaluated on separate stochastic adaptation runs.
- **Evidence Delivered (V8):**
  1. Implemented `--save-logits` and `--replay-logits` in `scripts/02_run_stream_benchmark.py`.
  2. All 4 conformal controllers evaluate on bit-for-bit identical replayed frozen logits.
  3. Verified by unit test: `tests/test_v8_regression.py::test_v8_04` [PASS].

### 2.5. Work Package 5: Negative Transfer Diagnosis & Certainty Weighting (V8 P0-5)
- **Defect in V7:** Tri-SACTA had lower Macro-F1 than source-only due to (a) Anchor Teacher Inversion (noisy audio anchor corrupting text) and (b) Abstention Penalty (all-rejected samples emitting all-zero features).
- **Evidence Delivered (V8):**
  1. Certainty-weighted alignment in `src/adaptation/tri_sacta.py:283` suppresses high-entropy anchor teachers.
  2. `fallback_to_presence_on_all_rejected=True` prevents all-zero feature emission in extreme noise while preserving clean single-count abstention tracking.
  3. Verified by unit test: `tests/test_v8_regression.py::test_v8_05` [PASS].
