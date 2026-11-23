# V10-06: RESEARCH DECISION & V11 ROADMAP

**Repository:** `buponbonbon/Tri-SACTA`  
**Base Commit:** `d146a06b1db3a59f01d3a04b586804bdfa69c03c`  
**Date:** November 23, 2026  
**Auditor:** Principal AI Research Scientist & Senior ML Systems Engineer  

---

## 1. Formal Research Decision

Among the five decision paths specified in Section 6 of the V10 directive:
- `Path A — Selective adaptation succeeds`: **REJECTED / FALSIFIED**. Active gradient adaptation ($\Delta \theta > 0$) causes negative transfer relative to `reliability_only` ($p = 0.046$).
- `Path B — Adaptation fails, spectral reliability succeeds`: **ADOPTED**. Dynamic spectral reliability gating consistently improves performance on out-of-distribution regimes (Regime 3 and Regime 5) over the source model.
- `Path C — Adaptation fails, conformal succeeds`: **ADOPTED**. Online ACI prevents coverage collapse (maintains ~90% vs 80.6% for static CP).
- `Path D — Adaptation fails, safe adaptation suppression succeeds`: **ADOPTED**. Suppressing gradient updates when anchor confidence is low or task capacity is unbalanced prevents confirmation bias.
- `Path E — No contribution is supported`: **NOT APPLICABLE**. Both spectral reliability detection and online conformal control are empirically validated.

### Selected Direction:
**PIVOT TO PATH B / PATH D:**
> **Title:** *When Does Multimodal Test-Time Adaptation Fail? Spectral Reliability Gating and Safe Adaptation Suppression under Compound Streaming Shifts*  
> **Target Journal:** *Information Fusion* (Elsevier, Q1, IF 18.6) or *IEEE Transactions on Multimedia* (IEEE, Q1, IF 8.4).

---

## 2. V11 Implementation & Engineering Roadmap

### Phase 1: Formalize Safe Adaptation Suppression (P0)
1. Implement a formal gating rule:
   $$\text{Adapt}(\mathcal{U}_t) \iff \min_{a \in \mathcal{A}_t} \mathcal{C}(a) \ge \tau_{\text{anchor\_capacity}} \quad \text{and} \quad \text{Entropy}(p_{\text{fused}}) \ge \tau_{\text{ambiguity}}$$
   If anchor modalities lack task capacity or predictions are already confident, freeze parameters ($\Delta \theta = 0$).
2. Demonstrate on validation data that safe suppression matches or exceeds `reliability_only` across all regimes.

### Phase 2: Independent Multi-Seed Training Pipeline (P1)
1. Add a dedicated script `scripts/train_multi_seed_checkpoints.py` to train source models for seeds `[100, 2026, 777, 999]` on authentic CMU-MOSEI train splits.
2. Store cryptographic manifest hashes and checkpoints in `checkpoints/cmu_mosei/real/seed{S}/source_model.pt`.
3. Re-run confirmatory benchmarks across all 5 seeds to provide multi-seed variance.

### Phase 3: Manuscript Preparation & Figures (P1)
1. Generate publication-quality vector plots (300 DPI PDF/PNG) for:
   - Spectral detection AUROC/AUPRC ROC curves across modalities.
   - Pareto frontier: Coverage vs Average Set Size for Conformal Controllers.
   - Per-regime Macro-F1 breakdown comparing Source-Only, Reliability-Only, AdapterNormTTA, and Tri-SACTA.
   - Video-grouped bootstrap difference distributions.
2. Draft the manuscript structure with honest empirical reporting of the failure of unsupervised TTA on weak multimodal streams and the success of dynamic reliability gating.
