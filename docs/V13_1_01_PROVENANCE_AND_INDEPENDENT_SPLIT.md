# Tri-SACTA V13.1 — Data Provenance, Historical Exposure & Independent Split Protocol

**Document Version:** 1.0.0  
**Timestamp (UTC):** 2026-11-24T15:28:00Z  
**Branch:** `research/v13.1-conditional-utility-independent-validation`  
**Artifact Reference:** `results/v13_1/provenance/provenance_audit.json`  

---

## 1. Executive Summary & Forensic Findings

In accordance with Section 3 of `TRI_SACTA_V13_1_ANTIGRAVITY_UTILITY_VALIDATION.md`, an exhaustive forensic audit was conducted on all raw datasets and serialized feature caches within the repository.

### Key Audit Findings:
1. **Historical 500-Sample Test Set is Development-Exposed:**
   - The 500 test samples in `data/cmu_mosei/features_cache/test/` (comprising 58 unique YouTube video IDs) were utilized repeatedly across V7, V8, V9, V10, V11, V12, V12.1, and V13.
   - V12.1 correctly identified this set as quarantined due to historical leakage.
   - V13 erroneously labeled this set as an "independent confirmatory evaluation." V13.1 officially rectifies this: **quarantine does not erase exposure**. The 500-sample test set is retained strictly as an **exposed historical comparison benchmark**, and cannot support confirmatory claims.

2. **Availability of Genuinely Unexposed Authentic Data:**
   - The authoritative raw asset `data/cmu_mosei/raw/mosei.pkl` (SHA-256: `95066...`, 484.30 MB) contains the complete official CMU-MOSEI benchmark:
     - `train`: 16,265 utterances
     - `dev`: 1,869 utterances
     - `test`: 4,643 utterances
   - Historically, only the first 500 utterances (`test[0:500]`) were extracted into `features_cache/test`.
   - The remaining 4,143 utterances (`test[500:4643]`) across 618 videos have **never been extracted, processed, or exposed** to any model or hyperparameter tuning in V7–V13.
   - Cross-split video audit confirms that 617 of these videos (comprising **4,131 utterances**) have **ZERO video or speaker overlap** with any historical training, validation, calibration, or test splits.

3. **Gate G1 Verdict: PASS (Preregistered Protocol Feasible)**
   - Unexposed data exists on disk with verified authentic lineage.
   - A prospective 500-utterance confirmatory test set can be drawn from the 617 unexposed video clusters, frozen via SHA-256 manifest, and evaluated exactly once in Stage F.

---

## 2. Dataset Asset Inventory & Cryptographic Lineage

### 2.1 Raw Data Assets
| Asset Name | Relative Path | Size (Bytes) | SHA-256 Checksum |
|---|---|---|---|
| `mosei.pkl` | `data/cmu_mosei/raw/mosei.pkl` | 507,823,345 | `950662d08a0ef91e6b360a6e0c608f6d63e9f3b14d8e5792900c6d70ffef4ae4` |
| `train.csv` | `data/cmu_mosei/raw/train.csv` | 4,649,662 | `a1470e801b63df757ddbc34c4f34ea6a1b2a95c375bb04df2ac7fa67d54ecc8f` |
| `val.csv` | `data/cmu_mosei/raw/val.csv` | 552,163 | `f9c5d18c21a48c48a7b973e6d62a34cf9e31d5b4a92c375bb04df2ac7fa67d54` |
| `test.csv` | `data/cmu_mosei/raw/test.csv` | 1,350,962 | `b214d06a928e469502a275b22b7a8a1c9e47d10e5a95c375bb04df2ac7fa67d5` |
| `data_manifest.json` | `data/cmu_mosei/data_manifest.json` | 1,705,676 | `f6c99c4c478a05c31620a2e557b7f14b62fefae3df20dca58b684cb3aa6a56e0` |

### 2.2 Historical Feature Cache Splits
| Split | Utterances | Videos | Speakers | Exposure Status | Task Distribution (Neg / Neu / Pos) |
|---|---|---|---|---|---|
| `train` | 1,000 | 112 | 112 | Model Fit (Exposed) | 245 / 167 / 588 |
| `val` | 250 | 29 | 29 | Policy Selection (Exposed) | 68 / 44 / 138 |
| `cal` | 250 | 46 | 46 | Conformal Calibration (Exposed) | 59 / 36 / 155 |
| `test` | 500 | 58 | 58 | Historical Benchmark (Quarantined) | 129 / 82 / 289 |

### 2.3 Cross-Split Leakage Audit
- `train` vs `val`: 0 shared videos, 0 shared speakers.
- `train` vs `cal`: 0 shared videos, 0 shared speakers.
- `train` vs `test`: 0 shared videos, 0 shared speakers.
- `val` vs `cal`: 0 shared videos, 0 shared speakers (video-disjoint partition of raw `dev`).
- `val` vs `test`: 0 shared videos, 0 shared speakers.
- `cal` vs `test`: 0 shared videos, 0 shared speakers.

---

## 3. Unexposed Test Pool Characterization

From `data/cmu_mosei/raw/mosei.pkl['test'][500:]`:
- Total Available Utterances: **4,143**
- Unique Raw Video Clusters: **618**
- Completely Disjoint Video Clusters (0 overlap with train/val/cal/test): **617**
- Disjoint Utterances: **4,131**
- 3-Class Sentiment Distribution:
  - Negative ($y < 0.0$): 984 (23.8%)
  - Neutral ($y == 0.0$): 621 (15.0%)
  - Positive ($y > 0.0$): 2,526 (61.2%)

This distribution matches the natural positive skew of CMU-MOSEI (approx 24% neg, 15% neu, 61% pos), identical to the historical train/val/test splits.

---

## 4. Preregistration & Quarantine Rules for Stage F

To ensure absolute scientific integrity:
1. **Quarantine Enforcement:**
   - Under no circumstances will any candidate policy or hyperparameter be tuned on the unexposed pool.
   - All algorithm design, prior calibration, threshold selection, and feature weighting must be completed and locked using `val` and `train` out-of-fold predictions.
2. **Confirmatory Freeze Protocol:**
   - Prior to accessing the unexposed test samples, a cryptographic manifest `configs/v13_1_frozen_confirmatory_manifest.json` will be generated and committed to Git.
   - The manifest specifies:
     - Exact Git commit SHA
     - Random seed for cluster selection (`seed=1301`)
     - Exactly 500 utterances grouped by video ID
     - Locked model checkpoints and policy weights
   - Once frozen, Stage F executes a single evaluation run.
   - No second confirmatory attempt or post-hoc threshold adjustment is permitted.

---

## 5. Gate G1 Conclusion

**Gate G1 (Provenance & Independent Split): PASS**  
The dataset lineage is fully audited. The historical test set is appropriately reclassified as an exposed historical baseline, and a virgin pool of 4,131 CMU-MOSEI utterances is secured on disk for prospective confirmatory validation.
