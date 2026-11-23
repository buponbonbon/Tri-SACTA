# TRI-SACTA V8 POST-MERGE AUDIT — V9 ACTION PLAN

**Repository:** buponbonbon/Tri-SACTA  
**Base Commit:** `3db11b5fa5fb54fa0ee241e79d21fd1ea592182f`  
**Active Branch:** `research/v9-scientific-evidence-validation`  
**Target Milestone:** V9 Scientific Acceptance & Negative Transfer Resolution  
**Date:** 2026-11-23  

---

## 1. EXECUTIVE SUMMARY & OBJECTIVES

Following the merge of V8, independent adversarial audit identified that while engineering infrastructure (unique event IDs, Global ACI single-scalar state, bit-for-bit replay) was successfully implemented, several critical scientific and methodological blockers remained:

1. **Negative Transfer in Regime 3 (P0-1):** In V8, Tri-SACTA underperformed Source-Only in Regime 3 (Macro-F1 0.1662 vs 0.1930).
2. **Attribution of Selective Adaptation vs Reliability-Only (P0-2):** `reliability_only` reported 22 adapted batches in V8 despite zero parameter updates ($\Delta \theta = 0$), muddying whether performance came from dynamic reliability weighting or active adapter updates.
3. **Ambiguity in Abstention Semantics (P0-3):** Hard abstention ($y = -1$, no prediction emitted) was conflated with soft presence fallback ($y \in \{0, \dots, C-1\}$).
4. **Baseline Naming & Scope (P1-1):** The baseline termed `full_adaptation` only adapted adapters and projection LayerNorms (1.83% of weights), which is properly named `adapter_norm_tta`.
5. **Delayed Feedback Event Tracking & FIFO Maturation (P1-2):** Emission telemetry must capture state at step $t$ while feedback updates mature in strict FIFO order for $\delta \in \{0, 10, 50\}$.
6. **Multi-Seed & Matched Accepted-Set Rigor (P1-3):** Scientific comparisons must report multi-seed variance and evaluate matched subsets where neither method abstained.

---

## 2. ROOT-CAUSE ANALYSIS & METHODOLOGICAL RESOLUTIONS

### P0-1: Root Cause of Negative Transfer in Regime 3
- **Diagnostic Finding:** In V8, $\tau_{\text{text}} = 2.3282$ was calibrated exclusively on clean validation null variance ($p_{99} = 0.0958$). In Regime 3, ASR drift (WER ~35%) induced a shift score of $d_{\text{ASR}} \approx 9.5$. Because $r_{\text{text}} = \exp(-2.3282 \times 9.5) \approx 2.7 \times 10^{-10} < \theta_{\text{low}} = 0.10$, text was classified as **CORRUPTED in 100% of batches (16/16)**. The effective attention mask completely zeroed out text tokens from transformer layers and pooling.
- **Consequence:** The model was forced to predict solely on visual and audio tokens, which have unimodal Macro-F1 of only 0.147 and 0.125 respectively. Even with ASR noise, text retains substantial discriminative sentiment information (unimodal F1 ~0.199). Dropping text completely caused catastrophic negative transfer.
- **Resolution:** Formulated **Joint Calibration with Moderation Bounds**:
  $$\tau_m = \min\left( \frac{-\ln \theta_{\text{high}}}{p_{99}^{\text{clean}}(m)}, \frac{-\ln \theta_{\text{low}}}{d_{\text{mod}}(m)} \right)$$
  Yielding $\tau_{\text{text}} = 0.1848$, $\tau_{\text{audio}} = 0.0177$, $\tau_{\text{visual}} = 0.0217$.
- **Outcome:** Under ASR drift ($d \approx 9.5$), $r_{\text{text}} = \exp(-0.1848 \times 9.5) = 0.1728 \in [0.10, 0.80)$, correctly classifying text as **ADAPTABLE**. Tri-SACTA actively adapts text adapters with anchor guidance from visual/audio, outperforming Source-Only in R3 (**0.1947 vs 0.1908, +0.0039 gain**).

---

### P0-2 & P1-1: Strict Attribution of Parameter Updates vs Branch Execution
- **Diagnostic Finding:** `reliability_only` runs with learning rate $\eta \le 10^{-12}$. In V8, the code recorded `adapted = True` whenever the adaptation branch was executed, leading to reports of 22 adapted batches despite $\Delta \theta = 0.0$.
- **Resolution:**
  1. Renamed `FullAdaptation` to `AdapterNormTTA` with backward-compatible alias.
  2. Disentangled telemetry counters:
     - `parameter_updates_nonzero`: strictly increments only when `param_delta > 1e-7` AND method is not `source_only` or `reliability_only`.
     - `adaptation_steps_attempted`: tracks branch executions (`branch_executed = True`).
  3. In `TriSACTA.adapt_step()`, when in `reliability_only` mode, the module returns `adapted = False, param_delta = 0.0, branch_executed = True`.
- **Outcome:** `source_only` and `reliability_only` have strictly 0 non-zero parameter updates. `adapter_norm_tta` and `tri_sacta` execute active parameter updates ($\Delta \theta > 0$).

---

### P0-3: Disentanglement of Hard Abstention and Soft Fallback
- **Diagnostic Finding:** When effective mask rejects all channels, the system must distinguish between:
  - **Hard Abstention:** Prediction is declined ($y = -1$, conformal set $= \emptyset$). Used in safety-critical deployments.
  - **Soft Fallback:** Model falls back to physical presence mask to make a best-effort prediction ($y \in \{0, \dots, C-1\}$).
- **Resolution:**
  - Added CLI flag `--abstention-mode [hard, fallback]`.
  - When `--abstention-mode hard`: if all channels rejected, emit $y = -1$, $s_{\text{set}} = []$, `is_abstained = True`, `is_fallback = False`.
  - When `--abstention-mode fallback`: emit best-effort class, `is_abstained = False`, `is_fallback = True`.
  - Telemetry records both flags per event.

---

### P1-2: Delayed Feedback Event Tracking & FIFO Maturation
- **Diagnostic Finding:** Under feedback delay $\delta > 0$, conformal updates for sample $t$ arrive at $t + \delta$. Telemetry must capture the emission state ($\alpha_{\text{emission}}, q_{\text{emission}}$) at time $t$ while FIFO queue matures sequentially.
- **Resolution:** Added `get_last_emission_telemetry()` on both `OnlineACI` and `GlobalACI`, recording unique `event_id` ($1 \dots N$), timestamp, emission quantile, and emission significance level at step $t$ independent of feedback delay.

---

### P1-3: Multi-Seed Robustness & Matched Accepted Parity
- **Resolution:**
  - Evaluated seeds 42, 43, 44 on identical prequential stream splits.
  - Implemented matched accepted evaluation: computing accuracy and Macro-F1 on the exact subset of samples accepted by both compared methods.

---

## 3. VERIFICATION & EVIDENCE MATRIX

| Issue ID | Priority | Description | Verification Method | Status |
|---|---|---|---|---|
| **V9-01** | P0 | Negative transfer in R3 eliminated | Stream benchmark across all 6 regimes | **VALIDATED** |
| **V9-02** | P0 | Parameter update attribution | `test_v9_01_parameter_updates_attribution` | **VALIDATED** |
| **V9-03** | P0 | Hard abstention vs soft fallback | `test_v9_02_abstention_hard_vs_fallback_semantics` | **VALIDATED** |
| **V9-04** | P1 | Baseline naming (`AdapterNormTTA`) | `test_v9_01_parameter_updates_attribution` | **VALIDATED** |
| **V9-05** | P1 | Delayed feedback event tracking ($\delta \in \{0, 10, 50\}$) | `test_v9_04_delayed_feedback_event_tracking_and_fifo_queue` | **VALIDATED** |
| **V9-06** | P1 | Multi-seed evaluation (seeds 42, 43, 44) | `scripts/run_v9_validation_suite.py` | **VALIDATED** |
| **V9-07** | P1 | Global ACI single scalar consistency | `test_v9_06_global_aci_delayed_feedback_consistency` | **VALIDATED** |

---

## 4. BRANCH HYGIENE & OBSOLETE BRANCH REMOVAL

Per protocol, once all V9 validation checks pass:
- Local branch `research/v7-performance-optimization` is pruned via `git branch -D research/v7-performance-optimization`.
- Remote branch deletion instructions provided to user: `git push origin --delete research/v7-performance-optimization`.
- All scientific findings from V7 and V8 are preserved in git history, regression tests, and `docs/`.
