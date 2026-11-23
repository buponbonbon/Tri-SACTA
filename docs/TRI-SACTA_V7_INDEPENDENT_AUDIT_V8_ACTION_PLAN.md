# TRI-SACTA V7 INDEPENDENT SCIENTIFIC AUDIT — V8 ACTION PLAN

**Repository:** buponbonbon/Tri-SACTA  
**Audited Commit:** `974573d4f366216a6b2f0afb5c71117aebddd06f`  
**Audit Date:** 2026-11-23  
**Branch to Create:** `research/v8-valid-baselines-and-event-trace`

---

## EXECUTIVE SUMMARY

V7 introduced several genuine improvements (source backbone Macro-F1 0.2598, calibrated spectral temperatures, effective_mask fix, single-count abstention, stream equivalence hashing). However, independent adversarial review has identified five P0 blockers that invalidate the comparative experimental claims.

---

## P0 BLOCKERS

### P0-1: `full_adaptation` performs ZERO parameter updates

**Root cause:** In `scripts/02_run_stream_benchmark.py:218-222`, `full_adaptation` is initialized with:
```python
full_config["tta"]["theta_high"] = 0.0
full_config["tta"]["theta_low"]  = 0.0
```
In `TriSACTA._classify_modalities`, any modality with `r >= theta_high` (i.e., r >= 0.0) is classified as **Anchor**, not Adaptable. With both thresholds at 0, all present modalities enter `anchor_set`, leaving `adaptable_set` empty. Since `adapt_step` requires both `adaptable_set` and `anchor_set` to be non-empty, NO gradient updates occur.

**Evidence:** `adaptation_counts.total_adapted_batches = 0` for `full_adaptation` in V7 results.

**Fix:** Implement `full_adaptation` as an unconditional full-parameter gradient update (entropy minimization on ALL parameters, using all present modalities), without mask classification.

---

### P0-2: Event-level conformal telemetry has only 96 unique event IDs out of 3,000 records

**Root cause:** `OnlineACI.predict_set()` increments `self._global_t` and stores `records[self._global_t]`. Then `update()` calls `observe(current_id=self._global_t)` — but `self._global_t` has ALREADY been incremented by subsequent `predict_set()` calls for other samples in the same batch. For batch size > 1, `update(current_id=self._global_t)` tries to observe a record ID that belongs to the LAST sample, not the current one. For bsz=1 loops, `predict_set` and `update` are called alternately per sample — but the loop in `02_run_stream_benchmark.py:397-415` calls predict_set for ALL i in bsz, then update for ALL i in bsz, so `self._global_t` after the predict loop points to the last emitted record. `update` then repeatedly observes the same final record.

**Evidence:** V7 validation shows only 96 `event_id` values (one per batch) rather than 3,000 (one per sample).

**Fix:** `predict_set` must return the `record_id` for each sample. `update()` must accept an explicit `record_id` argument. The benchmark loop must pass the correct `record_id` to `update`.

---

### P0-3: `global_aci` is NOT a genuinely global controller

**Root cause:** `global_aci` sets `min_mask_support=999999` and `controller="fixed_mask_aci"`. This forces pooled-window quantile computation (global fallback) but maintains PER-MASK alpha state (`self.alphas[mask_key]`). Different mask keys therefore diverge toward different alpha values. A truly global ACI must maintain a SINGLE global alpha level and a single global score window.

**Fix:** Add a `GlobalACI` subclass (or mode) that uses a single alpha scalar and single pooled window for all mask types.

---

### P0-4: Conformal controller comparison uses DIFFERENT model states for each variant

**Root cause:** `run_v7_validation_suite.py` calls `02_run_stream_benchmark.py` once per conformal variant, each time loading the same checkpoint but running full Tri-SACTA adaptation. Because adaptation is stochastic (depends on minibatch order and optimizer state), each conformal run sees DIFFERENT model weights during streaming. Scientific comparison of conformal controllers must freeze model logits.

**Fix:** Add a `--logits-replay-file` mode to `02_run_stream_benchmark.py` that (a) saves logits on a reference run and (b) replays frozen logits for conformal comparisons.

---

### P0-5: Tri-SACTA Macro-F1 < source-only in ALL SIX regimes

**Root cause (hypotheses, to be verified empirically):**
1. **Regime 1 (Clean):** Tri-SACTA F1=0.2547 vs source-only F1=0.2598. Adaptation is active on clean data (21 batches). The clean anchor should suppress adaptation entirely. Hypothesis: some batches are misclassified as having adaptable modalities on clean data, causing harmful gradient updates.
2. **Regimes 2-5:** Adaptation consistently hurts. Likely causes: (a) entropy minimization on 7-class imbalanced CMU-MOSEI converges to lower-entropy but wrong predictions; (b) alignment loss drives adaptable modality toward anchor distribution, but on CMU-MOSEI the text (anchor) has limited discriminability; (c) the adaptation learning rate (1e-4) is too high relative to the amount of adaptation signal (small adaptable_set).
3. **All regimes:** The selective abstention (225 samples rejected) may be removing hardest samples but computing Macro-F1 on BOTH abstained and non-abstained predictions together.

**Fix:** Diagnose with per-regime confusion matrices and adaptation frequency histograms. Implement controlled ablation: Tri-SACTA with lr=0 as baseline, then sweep lr and lambda_align.

---

## P1 BLOCKERS

### P1-1: Spectral calibration contradiction — V6 vs V7 clean anchor rate

V7 claims clean anchor retention rose from 33.3% (V6) to 87.5% after temperature calibration, yet Tri-SACTA adapts 21/96 batches in Regime 1 (clean). If clean anchor rate is 87.5%, almost all batches should have anchor modalities (text in A) with no adaptable modalities. The fact that 21 clean batches trigger adaptation indicates significant clean data is still being misclassified into the adaptable set.

**Fix:** Provide per-regime adaptation frequency breakdown and per-batch modality classification histograms.

---

### P1-2: Abstention semantics are correct but selective metrics are inconsistent

Selective Macro-F1 (computed on non-abstained samples) is not directly comparable to source-only Macro-F1 (computed on all samples). The 225 rejected samples are likely the hardest (all modalities corrupted), and removing them inflates selective metrics.

**Fix:** Report three metrics: (1) all-sample Macro-F1, (2) selective Macro-F1, (3) coverage-normalized Macro-F1 weighted by rejection rate.

---

## MANDATORY DELIVERABLES

1. `docs/TRI-SACTA_V7_INDEPENDENT_AUDIT_V8_ACTION_PLAN.md` (this file)
2. `docs/AUDIT_STATUS_V8.md` — Complete defect register with resolution evidence
3. `docs/HANDOVER_V8.md` — Technical handover with root-cause analysis and reproducible commands
4. `docs/CLAIM_EVIDENCE_MATRIX.md` — Updated to v8.0
5. `tests/test_v8_regression.py` — Regression tests for all P0 fixes
6. `results/v8_validation/v8_validation_summary.json` — Empirical validation with verified results
7. Updated `README.md` in English reflecting V8 status

**Do not auto-merge. Do not fabricate results. Submit PR for independent review.**
