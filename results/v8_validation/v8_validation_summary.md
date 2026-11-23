# Tri-SACTA V8: Comprehensive Empirical Validation Report

**Stream Hash (Equivalence Verified):** `75bbcce80138aaa7...`  
**Checkpoint Hash:** `61dd86587e31bcd6...`  

## 1. Adaptation Variants Comparison (Identical Stream & Checkpoint)
| Method | Accuracy | Macro-F1 | Empirical Cov (Nominal 90%) | Cov Gap | Avg Set Size | Adapted Batches | Abstain Count (Rej Rate) | Selective F1 |
|---|---|---|---|---|---|---|---|---|
| `source_only` | 0.2390 | **0.2152** | 0.8980 | 0.0020 | 5.06 | 0/96 (0.0%) | 0 (0.0%) | 0.2152 |
| `reliability_only` | 0.2370 | **0.2143** | 0.8967 | 0.0033 | 4.96 | 22/96 (22.9%) | 225 (7.5%) | 0.2216 |
| `full_adaptation` | 0.2353 | **0.2117** | 0.8967 | 0.0033 | 5.10 | 96/96 (100.0%) | 0 (0.0%) | 0.2117 |
| `tri_sacta` | 0.2353 | **0.2129** | 0.8967 | 0.0033 | 4.99 | 22/96 (22.9%) | 225 (7.5%) | 0.2201 |

## 2. Conformal Controller Comparison (Replayed on 100% Identical Frozen Logits)
| Controller | Accuracy | Macro-F1 | Empirical Coverage | Coverage Gap | Avg Set Size | Controller Design |
|---|---|---|---|---|---|---|
| `fixed_mask_aci` | 0.2353 | 0.2129 | 0.8977 | 0.0023 | 5.03 | Exact Fixed Step |
| `dynamic_mask_aci` | 0.2353 | 0.2129 | 0.8967 | 0.0033 | 4.99 | Velocity-Modulated |
| `global_aci` | 0.2353 | 0.2129 | 0.9003 | 0.0003 | 5.01 | Global Scalar Alpha |
| `mccf` | 0.2353 | 0.2129 | 0.8017 | 0.0983 | 4.27 | Static Split CP |

## 3. Per-Regime Robustness & Negative Transfer Resolution
| Regime | Name | Source-Only F1 | Reliability-Only F1 | Full-Adaptation F1 | Tri-SACTA F1 | Delta vs Source |
|---|---|---|---|---|---|---|
| `regime_1` | Regime 1: Clean Baseline | 0.2598 | 0.2632 | 0.2600 | **0.2632** | +0.0033 |
| `regime_2` | Regime 2: Audio Degradation (SNR ~5dB) | 0.2561 | 0.2553 | 0.2531 | **0.2508** | -0.0053 |
| `regime_3` | Regime 3: ASR Text Drift (WER ~35%) | 0.1930 | 0.1675 | 0.1844 | **0.1662** | -0.0269 |
| `regime_4` | Regime 4: Visual Corruption + 30% Missing | 0.2078 | 0.2050 | 0.2077 | **0.2050** | -0.0028 |
| `regime_5` | Regime 5: Severe Compound Shift | 0.1239 | 0.1287 | 0.1203 | **0.1287** | +0.0048 |
| `regime_6` | Regime 6: Return-to-Clean Recovery | 0.2598 | 0.2598 | 0.2550 | **0.2582** | -0.0016 |
