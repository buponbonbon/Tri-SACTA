# V10-01: CLAIM-EVIDENCE RECONCILIATION & NUMERICAL PROVENANCE

**Repository:** `buponbonbon/Tri-SACTA`  
**Base Commit:** `d146a06b1db3a59f01d3a04b586804bdfa69c03c`  
**Date:** November 23, 2026  
**Auditor:** Independent Data Integrity & Statistical Auditor  

---

## 1. The Central Numerical Contradiction (P0-01)

In the V9 documentation and experimental artifacts, two conflicting sets of numbers were reported for the primary streaming benchmark on CMU-MOSEI real data (Seed 42):

| Document / Artifact | Source-Only Macro-F1 | Reliability-Only Macro-F1 | Tri-SACTA Macro-F1 | Tri vs Source Delta | Claimed Verdict |
|---|---:|---:|---:|---:|---|
| `results/v9_validation/v9_validation_summary.json` (Committed) | **0.215190** | **0.215562** | **0.214700** | **-0.000490** | Tri-SACTA underperforms Source-Only and Reliability-Only |
| `results/v9_validation/v9_validation_summary.md` (Committed) | **0.2152** | **0.2156** | **0.2147** | **-0.0005** | Negative transfer in R3 resolved, overall delta -0.0005 |
| `docs/HANDOVER_V9.md` (Section 2.1) | **0.2111** | **0.2139** | **0.2175** | **+0.0064** | Claims Tri-SACTA beats Source-Only by +0.0064 |
| `docs/AUDIT_STATUS_V9.md` (Table 1) | **0.2111** | N/A | **0.2175** | **+0.0064** | Claims Tri-SACTA beats Source-Only by +0.0064 |
| `docs/CLAIM_EVIDENCE_MATRIX.md` (v9.0) | **0.2111** | N/A | **0.2175** | **+0.0064** | Claims Tri-SACTA beats Source-Only by +0.0064 |

---

## 2. Forensic Provenance Investigation

To establish the origin of this discrepancy, the auditor inspected execution logs, scratch scripts, dataloader definitions, and git commit history:

### 2.1 Origin of Set A: 0.2111 (Source) vs 0.2175 (Tri-SACTA)
- During early V9 exploration, the engineering team executed a calibration diagnostic script (`scripts/calibrate_spectral_thresholds.py` and temporary evaluation loops) to determine the effect of lowering $\tau_{\text{text}}$ from 2.3282 to 0.1848.
- In that preliminary diagnostic script:
  - Mini-batch evaluation used a non-standard batch sequence or partial dataloader slices.
  - The script logged:
    - R1: Source=0.2427, Tri=0.2568
    - R2: Source=0.2690, Tri=0.2549
    - R3: Source=0.1908, Tri=0.1947
    - R4: Source=0.2002, Tri=0.2043
    - R5: Source=0.1337, Tri=0.1426
    - R6: Source=0.2344, Tri=0.2538
    - Overall: Source=**0.2111**, Tri-SACTA=**0.2175** (+0.0064).
- The narrative summaries in `HANDOVER_V9.md` and `AUDIT_STATUS_V9.md` copied these exploratory numbers into the draft documentation *before* the formal, end-to-end benchmark suite was executed.

### 2.2 Origin of Set B: 0.2152 (Source) vs 0.2156 (Rel) vs 0.2147 (Tri-SACTA)
- When `scripts/run_v9_validation_suite.py` was authored and executed, it invoked the official benchmark runner:
  ```bash
  python scripts/02_run_stream_benchmark.py --config configs/mosei_real.yaml \
      --method [source_only, reliability_only, adapter_norm_tta, tri_sacta] \
      --conformal online_aci --protocol prequential --data-mode real --seed 42 \
      --checkpoint checkpoints/cmu_mosei/real/seed42/source_model.pt \
      --output_dir results/v9_validation --abstention-mode fallback
  ```
- This benchmark strictly iterated through all 500 test utterances across all 6 regimes (3,000 stream events), computing the official SHA-256 stream hash:
  `75bbcce80138aaa720d03f37c988effbecb28d159f5726494f59ff14af5b7bc0`.
- The actual execution output from `02_run_stream_benchmark.py` produced:
  - Source-only: Macro-F1 = **0.215190**, Acc = **0.239000**
  - Reliability-only: Macro-F1 = **0.215562**, Acc = **0.239333**
  - AdapterNormTTA: Macro-F1 = **0.211750**, Acc = **0.235333**
  - Tri-SACTA: Macro-F1 = **0.214700**, Acc = **0.238333**
- The script automatically dumped this output to `results/v9_validation/v9_validation_summary.json`.
- However, the narrative markdown documentation was not reconciled with the executed JSON, creating the contradiction flagged by the external audit.

---

## 3. Authoritative Scientific Finding

1. **Set B (0.2152 vs 0.2156 vs 0.2147) is the AUTHORITATIVE, REPRODUCIBLE benchmark result.**
2. Set A (0.2111 vs 0.2175) was a preliminary exploration artifact and is hereby **DEPRECATED and RETRACTED**.
3. **Scientific Implication:**
   - Active parameter adaptation in Tri-SACTA does **NOT** achieve a +0.0064 gain over Source-Only.
   - On the full 3,000-event stream, Tri-SACTA (0.2147) slightly underperforms Source-Only (0.2152) by -0.0005, and underperforms Reliability-Only (0.2156) by -0.0009.
   - However, in **Regime 3 (Severe ASR Drift)**, both Reliability-Only (**0.1953**) and Tri-SACTA (**0.1952**) outperform Source-Only (**0.1930**), confirming that spectral calibration successfully prevented text dropout.
   - In **Regime 5 (Severe Compound Shift)**, Reliability-Only (**0.1334**) and Tri-SACTA (**0.1334**) outperform Source-Only (**0.1239** by +0.0095).

---

## 4. Multi-Seed Claim Reconciliation (P0-02)

- **Claim in V9 narrative:** Evaluated seeds 42, 43, 44.
- **Auditor Fact-Check:**
  - `checkpoints/cmu_mosei/real/` contains only `seed42/source_model.pt`.
  - The V9 script had logic:
    ```python
    multi_seeds = [42]
    for cand_s in [43, 44]:
        if os.path.exists(f"checkpoints/cmu_mosei/real/seed{cand_s}/source_model.pt"):
            multi_seeds.append(cand_s)
    ```
  - Because checkpoints for 43 and 44 did not exist on disk, the loop ran only for `[42]`.
  - The summary JSON honestly recorded: `"multi_seed": { "seeds": [42] }`.
  - But narrative text claimed multi-seed robustness.
- **Correction:** Tri-SACTA V10 explicitly records that confirmatory multi-seed training was **NOT RUN** for seeds 43/44 due to missing independently trained checkpoints. Seed 42 is the single evaluated pilot checkpoint.
