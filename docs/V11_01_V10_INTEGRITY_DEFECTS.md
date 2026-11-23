# V11-01: INVESTIGATION OF V10 SCIENTIFIC INTEGRITY DEFECTS & ANOMALIES

**Audit Date:** 2026-11-23  
**Auditor:** Antigravity 3.8 Independent Scientific Auditor  
**Document Code:** `TRI-SACTA-V11-01`  
**Base Commit:** `314364fcf90dbf80a112aadab227397370544cd7`  
**Active Branch:** `research/v11-system-feasibility-go-no-go`

---

## 1. P0-A: Repository Messaging and Claim Retraction

### Historical Defect
Prior to V11, `README.md` prominently advertised historical V8 synthetic fixture results (Macro-F1 0.4287), claimed target submission to Q1 journals, and cited an unverified exploratory performance jump (`0.2111 -> 0.2175`).

### Verification & Resolution
1. **Purged False Claims:** Removed all claims of SOTA test-time adaptation, Q1 readiness, and positive performance deltas.
2. **Explicit Retraction Notice:** Clearly documented that the early exploratory numbers were from uncalibrated preliminary slices and are formally retracted.
3. **Four-Tier Data Isolation:** Formally separated synthetic fixtures (`results/quarantined_v3_fixtures/`), development-exposed test features (`results/v10_audit/`), unexposed train/val splits, and raw media pipelines.
4. **Authoritative Numbers Reconciled:** Prominently stated the canonical seed-42 stream benchmark metrics (Source-Only: 0.2152, Reliability-Only: 0.2156, Tri-SACTA: 0.2147).

---

## 2. P0-B: Forensic Diagnosis of Alignment Invariance ($\Delta_{\text{align}} = 0$)

### Anomaly Identified
In V10, `selective_tta_no_align` ($\lambda_{\text{align}}=0.0$) and `tri_sacta` ($\lambda_{\text{align}}=0.5$) yielded bit-for-bit identical Macro-F1 ($0.21469995$), accuracy ($23.833\%$), and 0 label disagreements across all 3,000 stream events.

### Forensic Investigation
1. **Did Alignment Loss Execute?**
   Yes. Inspecting total parameter deltas reveals:
   - `tri_sacta` ($\lambda_{\text{align}}=0.5$): $\sum \|\Delta \theta\| = 0.275047$
   - `selective_tta_no_align` ($\lambda_{\text{align}}=0.0$): $\sum \|\Delta \theta\| = 0.275715$
   - Difference: $|\Delta_{\theta}| = 0.000668 > 0$.
   The runs were completely independent and computed different parameter updates.
2. **Mechanistic Cause of Discrete Zero Effect:**
   In `src/adaptation/tri_sacta.py`, alignment loss is modulated by anchor certainty:
   $$\text{certainty}_a = \max\left(0, 1 - \frac{H(p_a)}{\ln C}\right)$$
   For $C = 7$ classes, $\ln(7) = 1.9459$. Because the source model has low accuracy (~24%), unimodal anchor predictions are nearly uniform:
   $$\bar{H}(p_{\text{text}}) = 1.9287 \implies \text{certainty}_{\text{text}} = 0.0088$$
   $$\bar{H}(p_{\text{audio}}) = 1.9443 \implies \text{certainty}_{\text{audio}} = 0.0008$$
   $$\bar{H}(p_{\text{visual}}) = 1.9454 \implies \text{certainty}_{\text{visual}} = 0.0003$$
   The certainty weighting scales the alignment loss down by a factor of $10^{-3}$ to $10^{-4}$! The resulting gradient updates are so minute that they do not shift a single discrete argmax prediction across all 3,000 stream predictions.
3. **Verdict:** **`ACTIVATED_NO_MEASURABLE_EFFECT`**. The alignment loss was functionally implemented and executed, but mathematical certainty suppression correctly neutralized the noisy anchor teacher, rendering it completely inert.

---

## 3. P0-C: Mathematical Explanation of Visual Shift AUROC ($0.1578$)

### Anomaly Identified
In `results/v10_audit/diagnostics/rq1_spectral_detection_audit.json`, the spectral shift detector achieved AUROC = 1.0000 on text and 0.9992 on audio, but **0.1578** on visual features.

### Mathematical Root Cause: Finite-Sample Low-Rank Eigenvalue Collapse
The visual features are $D = 35$ dimensions. In streaming evaluation, batches have size $N = 32$.
1. **Clean Test Batches:** Because $N=32 \le D=35$, the empirical covariance matrix $C_b = \frac{1}{31} \tilde{X}^T \tilde{X}$ has rank at most $31$. The tail eigenvalues undergo extreme finite-sample shrinkage, collapsing to $\approx 1.5 \times 10^{-6}$.
   The source reference eigenvalues $\lambda_s$ (computed from $N=1,000$ training samples) have tail eigenvalues of $\approx 0.0025$.
   Computing log-eigenvalue divergence:
   $$(\ln(1.5 \times 10^{-6}) - \ln(0.0025))^2 \approx (-13.39 - (-5.99))^2 = (-7.4)^2 \approx 54.76$$
   This yields a massive clean spectral divergence score ($\text{mean} = 7.0068$).
2. **Shifted / Noisy Batches:** When additive Gaussian noise $\mathcal{N}(0, \sigma^2 I)$ is injected into the features, it adds $\sigma^2 I$ to the empirical covariance matrix.
   The tail eigenvalues are lifted from $10^{-6}$ up to $\sim \sigma^2 \approx 1.0$.
   Because $\ln(1.0) = 0$ is *closer* to $\ln(0.0025) \approx -5.99$ than $-13.39$ was, adding noise **reduces** the spectral divergence:
   $$\text{spectral\_div}_{\text{clean}} = 8.48 \implies \text{spectral\_div}_{\text{noisy}} = 6.71$$
3. **Inversion of Directionality:** The detector assigns **lower** excess shift scores to noisy batches than clean batches ($4.67$ vs $7.01$). Thus, when evaluating AUROC with shifted batches as the positive class ($y=1$), the classifier is inverted, producing AUROC = $0.1578 < 0.50$.
4. **Verdict:** The visual detector failure is not an indexing bug, but a fundamental limitation of unregularized log-eigenvalue divergence when $N \le D$.

---

## 4. P0-D: Test Execution Rigor

All test suites were executed live on the system. All 82 test functions passed without mocks or bypasses:
- Core regression suites (T01–T23, AUD, R3, R4, V5, V6, V7, V8, V9, V10, V11): **82 PASSED, 0 FAILED**.
- Tests confirm mathematical invariance, zero split leakage, and cryptographic consistency.
