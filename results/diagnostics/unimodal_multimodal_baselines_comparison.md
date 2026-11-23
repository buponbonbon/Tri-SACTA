# Unimodal and Multimodal Baselines Comparative Evaluation

**Dataset:** CMU-MOSEI Authentic (N_train=1000, N_val=250, N_test=500)

| Architecture / Baseline | Modality Scope | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|---|
| Majority-Class Predictor | None | 20.00% | 0.0476 | 0.0667 |
| Logistic Regression (Unweighted) | Text-only (768d) | 32.20% | 0.2410 | 0.3126 |
| **Logistic Regression (Balanced)** | **Text-only (768d)** | **28.40%** | **0.2458** | **0.2936** |
| 2-Layer MLP (Balanced) | Text-only (768d) | 32.00% | 0.2443 | 0.3069 |
| Logistic Regression (Balanced) | Audio-only (74d) | 10.60% | 0.1018 | 0.1213 |
| 2-Layer MLP (Balanced) | Audio-only (74d) | 13.80% | 0.1015 | 0.1257 |
| Logistic Regression (Balanced) | Visual-only (35d) | 15.80% | 0.1409 | 0.1666 |
| 2-Layer MLP (Balanced) | Visual-only (35d) | 15.80% | 0.1415 | 0.1542 |
| **Logistic Regression (Balanced)** | **Concat Early Fusion (877d)** | **29.40%** | **0.2590** | **0.3029** |
| 2-Layer MLP (Balanced) | Concat Early Fusion (877d) | 28.60% | 0.2590 | 0.2739 |
| **Gated Late Fusion (Balanced)** | **Multimodal (T+A+V)** | **30.00%** | **0.2755** | **0.2935** |
| Multimodal Transformer (V6 Checkpoint) | Multimodal (T+A+V) | 21.40% | 0.1865 | 0.2003 |
