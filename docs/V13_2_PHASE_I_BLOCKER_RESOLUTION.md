# Tri-SACTA V13.2: Phase I Blocker Resolution & Discrepancy Ledger

Audited revision: `1cd239e2b4fc1ef7e5cdd8c213cd71acf5ac022e` -> Scientific Source SHA: `f51a9edc819623b10339c9b0c8232a85b09643a9` -> Manifest Digest: `b5e004f17c75dcfeff65bdb9f51841ef2c1deecdc427c036f904a8db3e34d008`

| Blocker | Original Defect | Implemented Fix & Verification |
| :--- | :--- | :--- |
| **#1 Freeze Incomplete** | Manifest lacked 21 checkpoint hashes, sample IDs, code/lock hashes, executable protocol. | Two-commit immutable freeze via [v13_2_audit_provenance.py](../scripts/v13_2_audit_provenance.py) + [v13_2_freeze_protocol.py](../scripts/v13_2_freeze_protocol.py) locking 21 checkpoints, `4131` canonical IDs, code tree, `requirements-lock.txt`, and [v13_2_run_phase2_holdout.py](../scripts/v13_2_run_phase2_holdout.py). |
| **#2 G5 Passed Empty JSON** | `verify_gates` accepted `{}` in `freeze_manifest.json`. | Strict schema, 21 checkpoint byte verification, code tree verification, detached `.sha256` & external `--pinned-manifest-sha256` checks (`F01`–`F04`). |
| **#3 G3 Ignored Static_TA** | G3 only checked `AlwaysOnFusion`. | Joint intersection over BOTH `AlwaysOn_Direct_Serial/Parallel` AND `Static_TA_Direct` (`S01`, `S02`). |
| **#4 G2 Ignored Baselines/Grid** | G2 only checked candidate F1. | Verifies full `84,000`-row grid, zero duplicate keys, exact cross-policy pairing, domain checks, and all 16 policy summaries (`E01`–`E08`). |
| **#5 G4 Trusted JSON** | G4 read JSON percentage directly. | Recomputes F1, total/mean app bytes, linear p50/p95/p99, useful-on-time, acceptance, client/server agreement & fault suite directly from CSV (`T01`–`T06`). |
| **#6 G6 Hardcoded NOT_RUN** | G6 returned static NOT_RUN without FS/ledger checks. | Checks `holdout_access_ledger.json`, scans prohibited confirmatory/cache directories (`H03`), and enforces `--approval` preflight in gated runner (`H01`, `H02`, `H04`, `H05`). |
| **#7 Docs Mismatched Artifacts** | Old `V13_2_03` / `README` cited `7.85 ms` seq p95 & `113,696 B` Static TA instead of actual `23.88 ms` & `110,030 B` (non-persistent run). | Replaced hand-edited tables with [v13_2_generate_reports.py](../scripts/v13_2_generate_reports.py) reading canonical JSON/CSV directly and disclosing all regressions (`R01`). |
| **#8 Missing Parallel/OneShot TCP** | Socket benchmark had 4 serial/clean modes only. | Added `AlwaysOn_Direct_Parallel`, `OneShot_ConfidenceRouting_0.55`, `BudgetAdmittedMaxSubset`, and `MatchedBudgetRandom_1.45` across all 7 workloads $\times$ 3 seeds (`42,000` TCP queries + sensitivity + 9 fault scenarios). |
| **#9 Resource Metric Ambiguity** | Hypothesis cited application-protocol bytes while G3 tested nominal cost. | Protocol amended (Section 3.5): measured application-protocol bytes primary across 7 workloads $\times$ 3 seeds; nominal cost secondary simulation (`S05`). |
| **#10 Novelty Unestablished** | Prior matrix compared only Wu 2026 / RegCal-Net / DeeBERT. | Added A2MT (`arXiv:2211.05039`), DynMM (`CVPRW 2023`), SemARC (`arXiv:2610.07466`), MoA-Off/MSAO (`arXiv:2509.16995`, `2604.02945`), SPINN (`MobiCom 2020`); honestly classified `novelty_status = OVERLAP_IDENTIFIED`. |

## Discrepancy Reconciliation vs Pre-Repair Snapshot (`1cd239e`)
- Old `1cd239e` distributed benchmark covered only 250 clean `val` queries (seed 42, non-persistent connects), where actual CSV values were Static_TA = `110,030 B`, mean `6.10 ms`, p95 `21.86 ms`; Sequential = `66,308 B`, mean `4.79 ms`, p95 `23.88 ms`. Old docs erroneously listed `113,696 B` and `7.85 ms`.
- Repaired benchmark runs persistent keep-alive TCP across all **7 workloads $\times$ 3 seeds (`5,250` queries/policy)** (`AlwaysOn_Direct_Parallel` `525.5 B/q`, `Static_TA_Direct` `276.3 B/q`, `OneShot_CR0.55` `155.2 B/q`, `SequentialConfidence` `133.1 B/q`) PLUS separate non-persistent sensitivity runs.
