# Tri-SACTA V7: Comprehensive Empirical Validation Report

**Stream Hash (Equivalence Verified):** `75bbcce80138aaa7...`  
**Checkpoint Hash:** `61dd86587e31bcd6...`  

## 1. Adaptation Variants Comparison (Identical Stream & Checkpoint)
| Method | Accuracy | Macro-F1 | Empirical Cov (Nominal 90%) | Cov Gap | Avg Set Size | Adapted Batches | Abstain Count (Rej Rate) | Selective F1 |
|---|---|---|---|---|---|---|---|---|
| `source_only` | 0.2390 | **0.2152** | 0.8980 | 0.0020 | 5.06 | 0/96 (0.0%) | 0 (0.0%) | 0.2152 |
| `reliability_only` | 0.2343 | **0.2122** | 0.9000 | 0.0000 | 5.02 | 21/96 (21.9%) | 225 (7.5%) | 0.2171 |
| `full_adaptation` | 0.2363 | **0.2112** | 0.8990 | 0.0010 | 5.07 | 0/96 (0.0%) | 0 (0.0%) | 0.2112 |
| `tri_sacta` | 0.2333 | **0.2122** | 0.8993 | 0.0007 | 5.01 | 21/96 (21.9%) | 225 (7.5%) | 0.2170 |

## 2. Conformal Controller Comparison (Identical Model & Logits)
| Controller | Accuracy | Macro-F1 | Empirical Coverage | Coverage Gap | Avg Set Size |
|---|---|---|---|---|---|
| `fixed_mask_aci` | 0.2333 | 0.2122 | 0.9063 | 0.0063 | 5.06 |
| `dynamic_mask_aci` | 0.2333 | 0.2122 | 0.8993 | 0.0007 | 5.01 |
| `global_aci` | 0.2333 | 0.2122 | 0.9070 | 0.0070 | 5.08 |
| `mccf` | 0.2327 | 0.2116 | 0.8307 | 0.0693 | 4.43 |

## 3. Per-Regime Robustness & Negative Transfer Resolution (Task 11)
| Regime | Name | Source-Only F1 | Reliability-Only F1 | Full-Adaptation F1 | Tri-SACTA F1 | Delta vs Source |
|---|---|---|---|---|---|---|
| `regime_1` | Regime 1: Clean Baseline | 0.2598 | 0.2547 | 0.2547 | **0.2547** | -0.0051 |
| `regime_2` | Regime 2: Audio Degradation (SNR ~5dB) | 0.2561 | 0.2467 | 0.2489 | **0.2453** | -0.0109 |
| `regime_3` | Regime 3: ASR Text Drift (WER ~35%) | 0.1930 | 0.1675 | 0.1933 | **0.1675** | -0.0255 |
| `regime_4` | Regime 4: Visual Corruption + 30% Missing | 0.2078 | 0.2026 | 0.2026 | **0.2003** | -0.0075 |
| `regime_5` | Regime 5: Severe Compound Shift | 0.1239 | 0.1136 | 0.1220 | **0.1136** | -0.0103 |
| `regime_6` | Regime 6: Return-to-Clean Recovery | 0.2598 | 0.2513 | 0.2513 | **0.2556** | -0.0042 |
