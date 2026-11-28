# Tri-SACTA V13: Oracle Headroom Analysis & Error Complementarity Report

**Date:** 2026-11-24  
**Split Evaluated:** `val` (N=250 samples, 29 unique video clusters, 0 overlap with train/cal/test)  
**Seeds Evaluated:** 42, 100, 2026 (Predictor bank with 7 dedicated task-matched MLPs)  
**Protocol Reference:** `configs/v13_utility_protocol.yaml`  

---

## 1. Executive Summary & Gate 1 Decision

The primary objective of Gate 1 was to determine whether **adaptive modality acquisition has sufficient predictive headroom ($\Delta \text{Macro-F1} \ge 0.05$) over the best static modality subset** to justify the development and deployment of an expected-utility sequential orchestration policy.

### **Gate 1 Verdict: PASS (GO FOR EXPECTED-UTILITY ORCHESTRATION)**
* **Best Static Modality Subset:** `TA` (Cost = 2.2, Macro-F1 = $0.5764 \pm 0.0071$, Accuracy = $0.5853 \pm 0.0100$)
* **Unconstrained Oracle Upper Bound:** Macro-F1 = $0.8811 \pm 0.0234$, Accuracy = $0.8853 \pm 0.0210$, Mean Cost = $1.15 \pm 0.02$
* **Point Headroom ($\Delta \text{Macro-F1}$):** **$+0.3047$ (+30.47 F1 points)**
* **Clustered Video-Grouped Bootstrap 95% CI ($B=1,000$ clusters):** **$[+0.2545, +0.3499]$**
* **Cost Efficiency Gain:** The Oracle achieves an F1 of $0.8811$ at an average cost of only $1.15$, representing a **$-47.6\%$ cost reduction** compared to static `TA` ($2.2$) and **$-67.1\%$ cost reduction** compared to `TAV` ($3.5$).

The empirical headroom exceeds the pre-registered threshold ($+0.05$) by more than $6\times$, with the entire 95% confidence interval strictly bounded above $+0.25$. This confirms beyond statistical doubt that different modality subsets contain strongly complementary diagnostic information for different video utterances.

---

## 2. Static Modality Subsets Performance

Each subset model was trained under task-matched conditions using early stopping on validation loss across 3 random seeds:

| Subset | Modalities Included | Dimension | Cost | Validation Accuracy (Mean $\pm$ Std) | Validation Macro-F1 (Mean $\pm$ Std) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`T`** | Text | 768 | 1.0 | $0.5827 \pm 0.0038$ | $0.5681 \pm 0.0052$ |
| **`A`** | Audio | 74 | 1.2 | $0.4213 \pm 0.0197$ | $0.3862 \pm 0.0054$ |
| **`V`** | Visual | 35 | 1.3 | $0.3907 \pm 0.0068$ | $0.3842 \pm 0.0036$ |
| **`TA`** | Text + Audio | 842 | 2.2 | **$0.5853 \pm 0.0100$** | **$0.5764 \pm 0.0071$** |
| **`TV`** | Text + Visual | 803 | 2.3 | $0.5480 \pm 0.0057$ | $0.5442 \pm 0.0067$ |
| **`AV`** | Audio + Visual | 109 | 2.5 | $0.4053 \pm 0.0082$ | $0.3953 \pm 0.0016$ |
| **`TAV`** | All Modalities | 877 | 3.5 | $0.5787 \pm 0.0019$ | $0.5678 \pm 0.0030$ |

### Observations:
1. **Dominance of Text:** Text-containing subsets (`T`, `TA`, `TV`, `TAV`) dramatically outperform non-text subsets (`A`, `V`, `AV`), achieving $\sim 0.57$ vs $\sim 0.39$ Macro-F1.
2. **Negative Fusion in Full Tri-Modal:** Always-on full fusion (`TAV`) suffers slight negative interference relative to `TA` ($0.5678$ vs $0.5764$), while consuming $1.6\times$ higher compute/bandwidth cost.
3. **Audio-Text Synergy:** `TA` achieves the highest static Macro-F1 ($0.5764$), confirming that acoustic features complement textual semantics for sentiment classification.

---

## 3. Cost-Constrained Oracle Frontier

The Oracle evaluates the optimal subset selection per sample under strict budget ceilings $B$:

| Budget Limit ($B$) | Feasible Subsets | Oracle Cost | Oracle Accuracy | Oracle Macro-F1 (Mean $\pm$ Std) |
| :---: | :---: | :---: | :---: | :---: |
| **$1.0$** | $\{T\}$ | 1.000 | 0.5827 | $0.5681 \pm 0.0052$ |
| **$1.5$** | $\{T, A, V\}$ | 1.055 | 0.8173 | $0.8067 \pm 0.0215$ |
| **$2.0$** | $\{T, A, V\}$ | 1.055 | 0.8173 | $0.8067 \pm 0.0215$ |
| **$2.2$** | $\{T, A, V, TA\}$ | 1.083 | 0.8413 | $0.8344 \pm 0.0146$ |
| **$2.5$** | $\{T, A, V, TA, TV, AV\}$ | 1.135 | 0.8787 | $0.8742 \pm 0.0243$ |
| **$3.0$** | $\{T, A, V, TA, TV, AV\}$ | 1.135 | 0.8787 | $0.8742 \pm 0.0243$ |
| **$3.5$** | All 7 Subsets | 1.152 | 0.8853 | **$0.8811 \pm 0.0234$** |

### Key Frontier Insights:
* Allowing the system to select between unimodal baselines $\{T, A, V\}$ (Budget $1.5$) jumps Macro-F1 from $0.5681 \to 0.8067$ (+23.86 F1 points) with only an average cost of $1.055$!
* Adding bi-modal alternatives (`TA`, `TV`, `AV`) increases Macro-F1 to $0.8742$ at an average cost of $1.135$.
* The full 7-subset unconstrained oracle tops out at $0.8811$ with cost $1.152$.

---

## 4. Error Complementarity & Rescue Dynamics

The $7 \times 7$ error complementarity matrix measures the conditional rescue probability:
$$P(S_{\text{rescuer}} \text{ is correct} \mid S_{\text{primary}} \text{ is incorrect})$$

### Rescue Rates when Primary Predictor is Incorrect:

| Primary Subset | Rescuer: `T` | Rescuer: `A` | Rescuer: `V` | Rescuer: `TA` | Rescuer: `TV` | Rescuer: `AV` | Rescuer: `TAV` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`T`** | — | **37.68%** | **33.87%** | 16.93% | 20.47% | **35.48%** | 22.68% |
| **`A`** | 54.95% | — | 34.61% | 51.95% | 52.06% | 25.59% | 51.74% |
| **`V`** | 54.73% | 37.81% | — | 54.09% | 46.88% | 25.10% | 50.36% |
| **`TA`** | 16.37% | 33.15% | 32.60% | — | 22.52% | 32.20% | 23.07% |
| **`TV`** | 26.58% | 38.63% | 28.33% | 28.91% | — | 28.03% | 21.84% |
| **`AV`** | 54.67% | 27.60% | 23.35% | 52.65% | 45.28% | — | 47.50% |
| **`TAV`** | 23.41% | 33.85% | 28.15% | 24.37% | 16.13% | 25.94% | — |

### Scientific Significance of Complementarity:
When the text modality alone fails (which happens on $\approx 41.7\%$ of utterances), **Audio alone correctly classifies $37.68\%$ of those failures**, and **Visual alone correctly classifies $33.87\%$ of them**!
This provides the definitive mathematical explanation for why sequential modality acquisition can succeed: the failure modes of unimodal audio and visual models are weakly correlated with text failures, creating substantial rescue opportunities if the router can accurately identify when text is uncertain.

---

## 5. Next Implementation Phase: Expected-Utility Sequential Acquisition

With Gate 1 unambiguously passed, we now implement the **Expected-Utility Sequential Modality Acquisition Policy** (`src/service/utility_router.py`):

1. **Sequential State Transitions:**
   * Step 0: Acquire base modality $S_0 = \{T\}$ (Cost: 1.0).
   * Evaluate initial uncertainty $U(S_0)$ via entropy $H(p_T)$ and margin $1 - \max_c p_T(c)$.
2. **Marginal Expected Utility:**
   * For candidate modality $m \in \{A, V\} \setminus S$:
     $$\Delta \hat{R}(m \mid S) = \text{Estimated Risk Reduction}$$
     $$\text{Utility}(m \mid S) = \frac{\Delta \hat{R}(m \mid S)}{\text{Cost}(m)}$$
3. **Policy-Matched Conformal Calibration:**
   * Calibrate risk on the exact deployed sequential trajectory on the `cal` split ($N=250$, 46 video clusters).
4. **Benchmarking Against Baselines:**
   * Confidence Routing, Fixed Fallback, Matched-Budget Random, Static Subsets, and Always-On Fusion across all 7 protocol workloads.
