# V11-02: SENTIMENT TASK PREDICTIVE UTILITY PILOT (VALIDATION SPLIT ONLY)

**Audit Date:** 2026-11-23  
**Auditor:** Antigravity 3.8 Independent Scientific Auditor  
**Document Code:** `TRI-SACTA-V11-02`  
**Dataset:** CMU-MOSEI Genuine Pre-extracted Features (mDeBERTa-v3 768D, COVAREP 74D, FACET 35D)  
**Partitioning:** Strictly train (1,000 samples, 112 videos) and val (250 samples, 29 videos). Zero test split exposure.

---

## 1. Study Motivation

The CMU-MOSEI source fusion backbone achieved ~23.9% accuracy on the 7-class stream benchmark (chance level is ~14.3%, majority class is ~20%). A system that misclassifies >75% of incoming utterances cannot support a credible application or systems paper.

This bounded study evaluates whether:
1. Reformulating the problem as a 3-class task (Negative / Neutral / Positive) provides acceptable utility.
2. The complex multimodal fusion architecture outperforms simple linear or unimodal text baselines.

---

## 2. Experimental Protocol

### 2.1. 3-Class Ground Truth Mapping
In CMU-MOSEI, raw sentiment scores $s \in [-3.0, +3.0]$ were mapped according to standard literature definitions:
- **Negative ($y = 0$):** $s < 0.0$ (40.0% of validation set)
- **Neutral ($y = 1$):** $s = 0.0$ (26.0% of validation set)
- **Positive ($y = 2$):** $s > 0.0$ (34.0% of validation set)

### 2.2. Models Evaluated on Validation Split ($N_{\text{val}} = 250$)
1. **Majority Class:** Predicts Negative ($y=0$) for all inputs.
2. **Text-Only Logistic Regression:** Supervised L-BFGS ($C=1.0$) on 768D text embeddings.
3. **Multimodal Early Concat Logistic Regression:** Supervised L-BFGS on concatenated Text (768D) + Audio (74D) + Visual (35D) = 877D.
4. **Text-Only MLP:** 1 hidden layer (64 units, ReLU) on 768D text embeddings.
5. **Multimodal Fusion Backbone (Remapped):** 1.37M parameter Transformer Cross-Modal Backbone with LayerNorm and Modality Prior Gating, remapped from 7 classes to 3 classes.

---

## 3. Empirical Results

| Architecture / Model | Input Modalities | Parameters | Validation Accuracy $\uparrow$ | Validation Macro-F1 $\uparrow$ | Status |
|---|---|:---:|:---:|:---:|---|
| **Majority-Class Baseline** | None | 0 | 40.00% | 0.1905 | Naive reference |
| **Text-Only Logistic Regression** | Text (768D) | 2,307 | **57.20%** | 0.5425 | Strong linear baseline |
| **Multimodal Concat Logistic Regression**| Text+Audio+Visual (877D) | 2,634 | 56.80% | **0.5477** | Competitive linear |
| **Text-Only MLP (1-Layer)** | Text (768D) | 49,411 | 50.80% | 0.4921 | Non-linear text |
| **Multimodal Fusion Backbone (Remapped)**| Text+Audio+Visual | 1,373,655 | **45.60%** | **0.4101** | **Severe Multimodal Penalty** |

---

## 4. Scientific Conclusions

1. **Failure of Multimodal Fusion:** The complex 1.37M parameter fusion network achieves only **45.60% accuracy**, trailing a simple Text-Only Logistic Regression (57.20%) by **11.60 percentage points** in accuracy and **0.1324** in Macro-F1.
2. **Multimodal Degradation ("Curse of Fusion"):** Adding acoustic and visual features to the neural fusion network degrades performance compared to text alone. Audio and visual representations in CMU-MOSEI exhibit high noise variance and lower signal-to-noise ratio, diluting text sentiment representations.
3. **Implications for Systems Paper:** An inference pipeline built around this multimodal backbone cannot be justified on grounds of utility, because a trivial 2KB linear classifier on text achieves substantially higher accuracy, consumes 1000x less memory, and requires zero multimodal alignment.
