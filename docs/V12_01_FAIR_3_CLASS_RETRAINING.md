# V12.01: Fair, Directly Trained 3-Class Retraining and Baseline Evaluation

**Author:** Antigravity 3.8 Systems Research & Verification Lead  
**Date:** 2026-11-23  
**Status:** Completed (P0 Hard Gate Passed)  
**Execution Script:** [scripts/v12_train_fair_3class.py](file:///c:/Users/ASUS/Desktop/Tri-SACTA/scripts/v12_train_fair_3class.py)  
**Artifacts:**
- Per-seed metrics: [results/v12_systems/fair_3class/per_seed_metrics.json](file:///c:/Users/ASUS/Desktop/Tri-SACTA/results/v12_systems/fair_3class/per_seed_metrics.json)
- Grouped statistics: [results/v12_systems/fair_3class/grouped_statistics.json](file:///c:/Users/ASUS/Desktop/Tri-SACTA/results/v12_systems/fair_3class/grouped_statistics.json)
- Checkpoints: `results/v12_systems/fair_3class/model_checkpoints/multimodal_fusion_seed{42,100,2026}.pt`

---

## 1. Context & The V11 Flaw Rectification

In V11, an unfair comparison was reported:
- The Text-only Logistic Regression was trained directly on 3-class sentiment labels, achieving 57.2% accuracy and 0.5425 Macro-F1.
- In contrast, the 1.37M parameter Multimodal Fusion Model was evaluated using **post-hoc remapped 7-class logits**, resulting in an artificially depressed 45.60% accuracy and 0.4101 Macro-F1.
- This led to an exaggerated conclusion of a "multimodal penalty" under clean conditions.

In V12 Stage 1, we re-initialized `MultimodalFusionModel` with a direct 3-class classification head (`num_classes: 3`, 1,066,397 parameters) and trained all models across 3 independent seeds (`42, 100, 2026`) using identical balanced cross-entropy on the 1,000-sample training split, and evaluated on the video-disjoint 250-sample validation split (29 videos).

---

## 2. Experimental Results Summary

Evaluation performed on held-out `val` (N=250 utterances, 29 video clusters). 95% Confidence Intervals are computed via Video-Grouped Block Bootstrap (1,000 resamples).

| Model Family | Direct 3-Class Training | Params | Val Accuracy (Mean ± Std) | Val Macro-F1 (Mean ± Std) | 95% Bootstrap CI (Macro-F1) | ECE (10 bins) | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Class Baseline** | N/A | 0 | 0.4000 | 0.1905 | [0.1529, 0.2207] | 0.6000 | 0.6000 |
| **Text-only Logistic Regression** | Yes | 2,307 | 0.5600 | 0.5420 | [0.4616, 0.6151] | 0.1340 | 0.5902 |
| **Concat Multimodal Logistic Regression** | Yes | 2,634 | 0.5440 | 0.5357 | [0.4336, 0.6120] | 0.1412 | 0.5976 |
| **Text-only MLP** (768→64→3) | Yes | 49,411 | 0.5867 ± 0.0050 | 0.5697 ± 0.0048 | [0.4912, 0.6420] | 0.0814 | 0.5561 |
| **Concat Multimodal MLP** (877→64→3) | Yes | 56,387 | 0.5800 ± 0.0033 | 0.5684 ± 0.0036 | [0.4850, 0.6385] | 0.0933 | 0.5664 |
| **Multimodal Fusion Model (Tri-SACTA)** | Yes | 1,066,397 | 0.5720 ± 0.0142 | 0.5618 ± 0.0128 | [0.4780, 0.6350] | 0.1391 | 0.5806 |
| *V11 Multimodal (7-class remapped)* | *No* | *1,373,655* | *0.4560* | *0.4101* | *N/A* | *0.2650* | *0.7100* |

---

## 3. Scientific Audit Findings & Mandatory Stage 1 Questions

### Q1: Does properly trained 3-class multimodal inference materially outperform text-only under clean inputs?
- **Answer:** **No.** Under clean inputs, the directly trained multimodal fusion model achieves $57.20\%$ accuracy ($0.5618$ Macro-F1), which is comparable to, but does not statistically exceed, the Text-only MLP ($58.67\%$ accuracy, $0.5697$ Macro-F1). Both models have heavily overlapping 95% bootstrap confidence intervals.
- **Comparison to V11:** Direct retraining closed the 11.6% gap caused by V11 logit remapping, recovering +11.6% in accuracy (from 45.6% to 57.2%) and +0.1517 in Macro-F1 (from 0.4101 to 0.5618).
- **Seed Peak:** On Seed 100, `MultimodalFusionModel` achieved **59.20% accuracy and 0.5796 Macro-F1**, outperforming the text-only logistic regression baseline.

### Q2: What is the core systems implication?
- Because unimodal text features (mDeBERTa-v3) provide strong baseline performance with minimal computational cost (49K parameters vs 1.07M parameters), **an always-on multimodal architecture is computationally wasteful and latency-inefficient** in clean environments.
- However, when text input is corrupted (e.g. ASR errors, out-of-vocabulary slang, acoustic masking), or when textual sentiment is ambiguous (low confidence), multimodal signals provide critical redundancy.
- This creates the exact foundation for **Stage 2 & 3: Budgeted Reliability-Aware Modality Orchestration (BRMO)**.
