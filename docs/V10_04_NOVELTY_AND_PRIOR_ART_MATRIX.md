# V10-04: NOVELTY & PRIOR ART COMPARISON MATRIX

**Repository:** `buponbonbon/Tri-SACTA`  
**Base Commit:** `d146a06b1db3a59f01d3a04b586804bdfa69c03c`  
**Date:** November 23, 2026  
**Auditor:** Principal AI Research Scientist & Adversarial Q1 Reviewer  

---

## 1. Targeted Literature Landscape

| Method / Reference | Venue / Year | Target Domain | Core Mechanism | Critical Limitations Relevant to Tri-SACTA |
|---|---|---|---|---|
| **TENT** (Wang et al.) | ICLR 2021 | Unimodal Vision | Entropy minimization on batch normalization parameters. | Fails under severe class imbalance and multimodal compound shifts; causes confirmation bias when initial accuracy is low (<30%). |
| **SAR** (Niu et al.) | ICLR 2023 | Unimodal Vision | Sharpness-aware entropy minimization with noisy sample filtering. | Uses scalar sample entropy threshold; assumes all modalities shift uniformly; lacks cross-modal awareness. |
| **AV-CTTA / READ** (Hong et al., Liu et al.) | CVPR / ECCV 2023–2024 | Bimodal Audio-Visual | Cross-modal feature alignment between audio and visual streams under corruptions. | Restricted to 2 modalities; assumes audio and visual have balanced task capacities; lacks distribution-shift statistical guarantees. |
| **Adaptive Conformal Inference (ACI)** (Gibbs & Candès) | NeurIPS 2021 | General Streaming | Online scalar tracking $\alpha_{t+1} = \alpha_t + \eta (\alpha - e_t)$. | Evaluates on single scalar; assumes exchangeable or slowly drifting unimodal error sequence; no multimodal missingness conditioning. |
| **Mask-Conditional CP (MCCF)** (Authors et al.) | Stat / ML 2023 | Static Multimodal | Split conformal prediction stratified by observed modality availability mask. | Assumes static exchangeability; coverage drops by ~10% under continual streaming compound shifts. |

---

## 2. Novelty Matrix of Proposed Tri-SACTA Contributions

| Contribution | Closest Prior Art | Mathematical Difference | Difference in Assumptions | Nature of Novelty | Necessary Experiment | Q1 Reviewer Publication Risk |
|---|---|---|---|---|---|---|
| **1. Sample-Size Matched Spectral Detector** | Covariance Frobenius norm drift; Mahalanobis mean drift | Computes log-eigenvalue spectrum via Gram matrix $(X X^\top)/(N-1)$ truncated at $k = \min(top\_k, D, N-1)$ to prevent small-batch rank collapse. | Prior art assumes $N > D$; Tri-SACTA operates under small streaming batches ($N \ll D$). | **GENUINE METHODOLOGICAL NOVELTY** | AUROC/AUPRC on high-dimensional text/audio under small batches ($N=16, 32$). Verified in `results/v10_audit/diagnostics/rq1_spectral_detection_audit.json`. | Low risk on high-dim modalities; Moderate risk on low-dim features ($D=35$). |
| **2. Dynamic Reliability Gating without Adaptation (`reliability_only`)** | Static late fusion; Softmax attention gating | Computes dynamic reliability $r_t^{(m)} = \exp(-\tau_m d_t^{(m)})$, gating cross-modal attention via `effective_mask` and reliability pooling weights without weight updates. | Traditional attention assumes static attention weights; Tri-SACTA modulates attention dynamically based on test-time distribution drift. | **HIGH PRACTICAL & EMPIRICAL VALUE** | Controlled ablation vs Source-Only on OOD regimes (R3 ASR drift, R5 compound shift). | Very low risk; highly defensible because it consistently beats Source-Only and TTA baselines. |
| **3. Online Anchor-Guided Adapter TTA** | AV-CTTA; READ; TENT | Tri-partitioning into Anchor $\mathcal{A}_t$, Adaptable $\mathcal{U}_t$, Corrupted $\mathcal{D}_t$; certainty-weighted KL alignment $w_a D_{\text{KL}}(p_u \parallel p_a)$. | Assumes anchor modality has high task accuracy to serve as a reliable teacher. | **FALSIFIED / METHODOLOGICALLY FLAWED ON WEAK MODALITIES** | Matched comparison vs `reliability_only`. Result: Statistically significant degradation ($p=0.046$). | **FATAL REJECTION RISK IF CLAIMED AS A GENERAL WINNER.** Must be presented as a failure analysis / safe adaptation suppression. |
| **4. Mask-Conditioned Online Conformal Inference under Delay** | Online ACI (Gibbs & Candès); Delayed ACI | Implements parallel mask-specific and global scalar tracking with FIFO queue maturation for feedback delay $\delta \in \{0, 10, 50\}$. | Explicitly models non-exchangeable multimodal missingness patterns and observation latency. | **INCREMENTAL BUT RIGOROUS** | Pareto frontier vs Fixed ACI and MCCF across delays $\delta \in \{0, 10, 50\}$. | Low risk if claimed as an empirical streaming framework rather than a new mathematical theorem. |

---

## 3. Recommended Reframing for Q1 Journal Submission

Attempting to publish Tri-SACTA as a paper claiming *"Online test-time gradient adaptation beats all baselines"* will be **immediately rejected** by any adversarial Q1 reviewer who requests the ablation against `reliability_only` ($\Delta \theta = 0$).

Instead, reframing the paper around **Path B / Path D**:
> **"When Does Multimodal Test-Time Adaptation Fail? Spectral Reliability Gating and Safe Adaptation Suppression under Compound Streaming Shifts"**
- **Core Narrative:**
  1. We establish why conventional multimodal test-time adaptation (entropy minimization and anchor alignment) suffers from confirmation bias and negative transfer when anchor modalities have low discriminability ($p=0.046$ degradation).
  2. We propose a dimension-calibrated spectral shift detector operating on small streaming batches ($N \ll D$) that achieves near-perfect shift detection (AUROC > 0.999 on text/audio).
  3. We demonstrate that dynamic spectral reliability gating and channel isolation achieves superior out-of-distribution robustness (+0.0023 in ASR drift, +0.0095 in severe compound shift) **without updating model weights**.
  4. We provide online conformal control under feedback delay, maintaining valid coverage under non-exchangeable streaming distribution shifts where static methods suffer a 9.4% coverage violation.

This reframed narrative turns an algorithmic negative result into a **major, rigorously supported scientific insight** that Q1 reviewers will appreciate.
