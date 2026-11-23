# V10-00: REPOSITORY FORENSIC INVENTORY & AUDIT

**Repository:** `buponbonbon/Tri-SACTA`  
**Base Commit (V9):** `d146a06b1db3a59f01d3a04b586804bdfa69c03c`  
**Working Branch:** `research/v10-independent-scientific-validation`  
**Audit Protocol:** Global Multi-Agent Scientific Integrity Protocol (Antigravity 3.8 Autonomous Research Team)  
**Date:** November 23, 2026  

---

## 1. Executive Summary

This forensic audit inspects the exact state of the repository following the V9 pull request (`#9`), reconciles discrepancies between text claims and committed experimental summaries, verifies data lineage, and categorizes every scientific claim according to strict empirical evidence standards.

---

## 2. Git & Commit Provenance

- **Main Branch Head:** `3db11b5fa5fb54fa0ee241e79d21fd1ea592182f` (Merge PR #8).
- **V9 PR (#9) Commit:** `d146a06b1db3a59f01d3a04b586804bdfa69c03c` on branch `research/v9-scientific-evidence-validation`.
- **Working V10 Branch:** `research/v10-independent-scientific-validation` branched cleanly from `d146a06`.
- **Working Tree State:** Completely clean prior to V10 investigation. No uncommitted local work lost. No branches deleted or force-pushed.

---

## 3. Checkpoint & Model Artifact Forensic Inventory

| Checkpoint Path | Dataset | Split / Mode | Seed | Parameters | File Size | SHA-256 Checksum (first 16) | Status |
|---|---|---|---|---|---|---|---|
| `checkpoints/cmu_mosei/real/seed42/source_model.pt` | CMU-MOSEI | Real | 42 | 1,071,537 | 4.3 MB | `61dd86587e31bcd6` | **VERIFIED AUTHENTIC** |
| `checkpoints/cmu_mosei/real/seed42/source_spectral_stats.pt` | CMU-MOSEI | Real | 42 | N/A | 2.4 MB | `76bb21e90ef8d91c` | **VERIFIED AUTHENTIC** |
| `checkpoints/meld/real/seed42/source_model.pt` | MELD | Real | 42 | 1,373,655 | 5.5 MB | `2e080cb966885ebc` | Text-only features (no raw audio/video features) |
| `checkpoints/quarantined_v3_fixtures/` | Various | Synthetic | 42 | Various | Various | Various | **PERMANENTLY QUARANTINED** |

> [!WARNING]
> **Multi-Seed Checkpoint Gap:**
> In `checkpoints/cmu_mosei/real/`, ONLY `seed42/` exists. No pre-trained source model checkpoints exist for seeds 100, 2026, 777, 999 (or 43, 44). Any claim that multi-seed confirmatory evaluation has been completed across independent checkpoints was an artifact of conditional fallback in V9. Seed 42 is the sole verified authentic source model checkpoint currently available.

---

## 4. Test Suite Forensic Audit

- Running `python tests/run_all_tests.py` executes 70 tests across 15 suites:
  - 100% Pass rate (70/70 PASSED, 0 FAILED).
  - Runtime: ~20 seconds on CPU.
- **Auditor Distinction (P0-03):**
  - All 70 tests verify **engineering invariants** (e.g., parameter delta $>0$ when active, non-empty prediction sets, valid dictionary shapes, no NaN outputs, 1-to-1 event ID uniqueness).
  - None of the 70 tests establish **scientific superiority** or statistical significance of the Tri-SACTA adaptation algorithm over baselines on real benchmark distributions.

---

## 5. Artifact & Results Mapping

- `results/v9_validation/v9_validation_summary.json`:
  - Contains exact point estimates for `source_only`, `reliability_only`, `adapter_norm_tta`, `tri_sacta`.
  - Stream hash: `75bbcce80138aaa720d03f37c988effbecb28d159f5726494f59ff14af5b7bc0`.
  - Frozen logits tensor: `results/v9_validation/tri_sacta_stream_logits.pt` (141 KB).
- `results/diagnostics/spectral_threshold_calibration.json`:
  - Contains calibrated temperatures: $\tau_{\text{text}} = 0.1848, \tau_{\text{audio}} = 0.0177, \tau_{\text{visual}} = 0.0217$.
- Discrepancy identified between narrative text (claiming F1=0.2175 vs 0.2111) and committed JSON (recording F1=0.2147 vs 0.2152), investigated in detail in `V10_01_CLAIM_EVIDENCE_RECONCILIATION.md`.
