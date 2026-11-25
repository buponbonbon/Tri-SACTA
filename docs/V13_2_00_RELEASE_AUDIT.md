# Tri-SACTA V13.2: Release-Candidate Forensic Audit & Defect Inventory

**Audit Version:** 13.2 (Antigravity 3.8 Contract)  
**Audit Date:** 2026-11-24 UTC  
**Active Branch:** `research/v13.2-final-systems-validation-release-candidate`  
**Base Commit:** `e71147fc511c5d50c218ded1fdad96953ae15b90`

---

## 1. Context & Forensic Findings from Independent Audit

The independent review of Tri-SACTA V13.1 identified critical inconsistencies between Markdown prose and underlying JSON/code implementations. While V13.1 successfully demonstrated that Expected Utility reduces algebraically to transformed confidence thresholds, several operational and data defects persisted:

1. **P0 Data Hash Contradiction:**  
   - `docs/V13_1_01_PROVENANCE_AND_INDEPENDENT_SPLIT.md` cited `950662d...` for `mosei.pkl`, while `provenance_audit.json` and the physical file had `90701ad...`.
   - Markdown reported label distribution `245 / 167 / 588` for train, while the physical tensor had `390 / 238 / 372`.
2. **P0 Video-Derived Speaker Proxy:**  
   - CMU-MOSEI speaker IDs were constructed via `spk_id = f"mosei_spk_{vid[:8]}"`, making "0 shared speakers" an artifact of "0 shared videos".
3. **P0 Evaluator Budget Ignored:**  
   - `scripts/v13_1_evaluate_policies.py` unconditionally passed `cost_budget=3.5` inside the evaluation loop, overriding heterogeneous workload budgets.
4. **P0 Baseline Transport Asymmetry:**  
   - `DistributedCoordinator` forced Always-On and Static TA to execute a redundant initial Text forward pass, artificially inflating their latency.
5. **P0 Modulo Indexing in Socket Server:**  
   - `ModalitySocketServer` executed `sample_idx = int(req["sample_idx"]) % total_samples`, silently serving wrong samples on out-of-bounds queries.
6. **P0 Verifier Hardcoded PASS:**  
   - `scripts/v13_1_verify_gates.py` hardcoded gate variables (`g0_pass=True`, `g8_pass=True`), excluded G5 from `all_gates_pass`, and printed `ALL GATES PASS` even when G5 was not run.

---

## 2. V13.2 Rectification & Verification Summary

Every identified defect was resolved through rigorous code modification and regression testing:

| Defect Area | V13.1 Vulnerability | V13.2 Rectification | Verification Evidence |
| :--- | :--- | :--- | :--- |
| **Provenance** | Stale hash & label typos in Markdown. | Exact `hashlib.sha256()` verification; physical count reconciliation (`390/238/372`). | [v13_2_audit_provenance.py](../scripts/v13_2_audit_provenance.py) |
| **Identity** | Claimed speaker independence via video proxy. | Schema updated: `video_group_id`, `speaker_id_verified: null`, `speaker_overlap_status: UNKNOWN`. | [V13_2_01_PROVENANCE_RECONCILIATION.md](../docs/V13_2_01_PROVENANCE_RECONCILIATION.md) |
| **Budget Flow** | `cost_budget=3.5` passed unconditionally. | Per-query `w_budgets[i]` propagated; infeasible initial budget rejects with `-1` and 0 cost. | [test_v13_2_evaluator_regression.py](../tests/test_v13_2_evaluator_regression.py) |
| **Transport** | Always-On executed redundant Text forward pass. | Optimized direct paths: Always-On runs direct TAV; Static TA runs direct TA. | [test_v13_2_runtime_safety.py](../tests/test_v13_2_runtime_safety.py) |
| **Socket Safety** | Modulo indexing on sample IDs. | Strict bounds check; invalid/out-of-bounds IDs return `400 Bad Request`. | [test_v13_2_runtime_safety.py](../tests/test_v13_2_runtime_safety.py) |
| **Gate Verifier** | Hardcoded boolean passes and cosmetic summary. | Recomputes metrics from raw traces; passes 7 adversarial corruption tests. | [test_v13_2_verifier_adversarial.py](../tests/test_v13_2_verifier_adversarial.py) |

---

## 3. Pre-Holdout Freeze Contract Status

- Phase I has completed all required developmental benchmarks, statistical analyses, systems measurements, and adversarial verifier tests.
- **Strict Boundary Enforced:** The fresh holdout pool of 4,131 utterances (617 video clusters) remains completely unaccessed and unexposed.
- **Phase I Status:** `READY_FOR_HOLDOUT_REVIEW`.
