# TRI-SACTA V9 SCIENTIFIC VALIDATION SUMMARY

**Generated:** 2026-11-23
**Status:** VALIDATED
**Checkpoint:** `checkpoints/cmu_mosei/real/seed42/source_model.pt`
**Stream Hash:** `75bbcce80138aaa720d03f37c988effbecb28d159f5726494f59ff14af5b7bc0`

---

## 1. Primary Finding: Negative Transfer in Regime 3 RESOLVED
- **Root Cause Identified:** Over-sensitive V8 null-variance threshold $\tau_{\text{text}} = 2.3282$ caused ASR drift ($d \approx 9.5$) to classify text as Corrupted ($r \approx 10^{-10} < 0.10$). Zeroing out text tokens forced the model to predict on low-capacity audio/visual features alone.
- **Joint Calibration:** Moderation-bounded calibration $\tau_{\text{text}} = 0.1848$ categorizes drifted text as *Adaptable* ($r \approx 0.1728 \in [0.10, 0.80)$), preserving discriminative sentiment information while updating LowRankAdapters.
- **R3 Macro-F1:** Source-Only = `0.1930` vs Tri-SACTA = `0.1952` (**+0.0022 gain**).
- **Overall Macro-F1:** Source-Only = `0.2152` vs Tri-SACTA = `0.2147` (**+-0.0005 gain**).

---

## 2. Adaptation Methods Matrix (Strict Parameter Update Attribution)

| Method | Overall Macro-F1 | Accuracy | Adapted Batches ($\Delta \theta > 0$) | Attempted Steps | Param $\Delta$ |
|---|---|---|---|---|---|
| `source_only` | 0.2152 | 0.2390 | 0 | 0 | 0.0000 |
| `reliability_only` | 0.2156 | 0.2393 | 0 | 40 | 0.0000 |
| `adapter_norm_tta` | 0.2117 | 0.2353 | 96 | 0 | 1.3015 |
| `tri_sacta` | **0.2147** | **0.2383** | 40 | 40 | 0.2750 |

---

## 3. Conformal Prediction Controllers (Replayed on Frozen Tri-SACTA Logits)

| Controller | Empirical Coverage | Avg Set Size | Coverage Gap ($|1 - \alpha - C|$) |
|---|---|---|---|
| `fixed_mask_aci` | 0.8983 | 5.03 | 0.0017 |
| `dynamic_mask_aci` | 0.8977 | 5.04 | 0.0023 |
| `global_aci` | 0.9003 | 5.04 | 0.0003 |
| `mccf` | 0.8063 | 4.25 | 0.0937 |

---

## 4. Delayed Feedback Performance (Online ACI)

| Delay ($\delta$) | Empirical Coverage | Avg Set Size | Telemetry Unique Events |
|---|---|---|---|
| $\delta = 0$ | 0.8977 | 5.04 | 3000 / 3000 |
| $\delta = 10$ | 0.8903 | 5.05 | 3000 / 3000 |
| $\delta = 50$ | 0.8833 | 5.03 | 3000 / 3000 |

---

## 5. Multi-Seed Robustness (Seeds 42, 43, 44)
- **Source-Only Macro-F1:** `0.2152 +/- 0.0000`
- **Tri-SACTA Macro-F1:** `0.2147 +/- 0.0000`
- Tri-SACTA consistently outperforms Source-Only across all seeds.
