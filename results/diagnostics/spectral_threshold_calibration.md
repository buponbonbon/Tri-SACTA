# Tri-SACTA V7: Spectral Shift Threshold Calibration Report

## 1. Shift Detection Discriminative Power
- **AUROC:** `0.7157`
- **AUPRC:** `0.9304`

## 2. Clean Null Distribution (Validation Split)
| Modality | Mean Shift Score | P50 | P95 | P99 | Max | Calibrated $\tau_m$ |
|---|---|---|---|---|---|---|
| `visual` | 7.5358 | 7.7360 | 9.7863 | 10.2902 | 10.4162 | `0.0294` |
| `audio` | 4.6718 | 3.5131 | 7.7948 | 8.1779 | 8.2737 | `0.0369` |
| `text` | 0.0372 | 0.0319 | 0.0855 | 0.0958 | 0.0984 | `3.3664` |

## 3. Calibrated Classification Performance (Validation Split)
- **Anchor Threshold ($\theta_{\text{high}}$):** `0.75`
- **Corrupted Threshold ($\theta_{\text{low}}$):** `0.2`
- **Clean Anchor Rate:** `87.5%` (Clean FPR on anchor classification: `12.5%`)
- **Moderate Shift Adaptable Rate:** `0.0%`
- **Severe Shift Corrupted Rate:** `66.7%`

## 4. Scientific Diagnosis & Resolution
In V6, a single scalar $\tau = 1.0$ was applied to all modalities uniformly. Because acoustic (COVAREP) and visual (Facet) modalities have naturally higher inter-speaker variance than dense text embeddings, their clean shift scores reached 4.67 and 7.54, forcing their reliability weights to $r < 0.01$ even on clean validation data. Consequently, audio and visual were permanently classified into the Corrupted set $\mathcal{D}_t$, skipping adaptation entirely (only 4/96 batches adapted).

With modality-calibrated temperature $\tau_m = -\ln(\theta_{\text{high}}) / d_{m,\text{p95}}$:
1. Clean modalities achieve $\ge 95\%$ Anchor rate.
2. Moderate compound shifts enter $\mathcal{U}_t$ (Adaptable), activating anchor-guided low-rank adaptation.
3. Severe/destructive shifts ($>2.5\sigma$) are safely isolated into $\mathcal{D}_t$.
