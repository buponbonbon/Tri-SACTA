# Tri-SACTA V13.2: Provenance Reconciliation, Data Lineage & Exposure Audit

**Audit Timestamp:** 2026-11-24 UTC  
**Active Branch:** `research/v13.2-final-systems-validation-release-candidate`  
**Base Commit:** `e71147fc511c5d50c218ded1fdad96953ae15b90`  
**Execution Script:** [v13_2_audit_provenance.py](../scripts/v13_2_audit_provenance.py)  
**Output Artifact:** [provenance_audit.json](../results/v13_2/provenance/provenance_audit.json)

---

## 1. Executive Summary & Root Cause Analysis

An independent scientific audit identified contradictions between the textual Markdown documentation in V13.1 and the underlying programmatic artifacts. Specifically:
1. The SHA-256 hash reported for `mosei.pkl` in `docs/V13_1_01_PROVENANCE_AND_INDEPENDENT_SPLIT.md` (`950662d...`) did not match the JSON artifact or the physical file on disk (`90701ad...`).
2. The 3-class label distribution tables in Markdown (e.g., `245 / 167 / 588` for train) contradicted the JSON artifact (`390 / 238 / 372`).
3. Historical claims of "0 shared speakers" were based on a synthetic string prefix `spk_id = f"mosei_spk_{vid[:8]}"`, which is a video-derived proxy rather than a verified human speaker identifier.

### Mathematical & Forensic Reconciliation
- **On-Disk Reality:** The on-disk file `data/cmu_mosei/raw/mosei.pkl` (507,823,345 bytes) has SHA-256 `90701ad5bfa8767ca253af729c1d346e9a8d259ed24f16de113c845ffb60d90e`. The V13.1 JSON artifact [provenance_audit.json](../results/v13_1/provenance/provenance_audit.json) accurately recorded this hash. The Markdown file suffered from a stale copy-paste transcription error.
- **Label Distribution Reality:** The data loader [loader.py](../src/data/loader.py), model checkpoints, and evaluation scripts all use the continuous sentiment thresholding standard in CMU-MOSEI benchmarks:
  $$\text{class} = \begin{cases} 0 \ (\text{Negative}), & \text{raw\_sentiment} < 0.0 \\ 1 \ (\text{Neutral}), & \text{raw\_sentiment} = 0.0 \\ 2 \ (\text{Positive}), & \text{raw\_sentiment} > 0.0 \end{cases}$$
  Under this exact formula, the physical tensors yield the counts below. The Markdown numbers were transcribed from an obsolete legacy draft and are hereby formally superseded.

| Split | Utterances | Video Groups | Old V13.1 Markdown (Obsolete) | Verified Physical Reality (JSON & Code) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `train` | 1,000 | 112 | 245 / 167 / 588 | **390 / 238 / 372** | **RECONCILED** |
| `val` | 250 | 29 | 68 / 44 / 138 | **100 / 65 / 85** | **RECONCILED** |
| `cal` | 250 | 46 | 59 / 36 / 155 | **43 / 66 / 141** | **RECONCILED** |
| `test` (hist.) | 500 | 58 | 129 / 82 / 289 | **213 / 117 / 170** | **RECONCILED** |
| `unexposed pool` | 4,131 | 617 | 984 / 621 / 2526 | **1,132 / 900 / 2,099** | **RECONCILED** |

---

## 2. Physical Asset Inventory and Checksums

All hashes were recomputed directly via `hashlib.sha256()` on the active file system:

### 2.1 Raw Data Assets
| Asset Name | Relative Path | Size (Bytes) | Verified SHA-256 Checksum |
| :--- | :--- | :---: | :--- |
| `mosei.pkl` | `data/cmu_mosei/raw/mosei.pkl` | 507,823,345 | `90701ad5bfa8767ca253af729c1d346e9a8d259ed24f16de113c845ffb60d90e` |
| `train.csv` | `data/cmu_mosei/raw/train.csv` | 4,649,662 | `e548cbf003c6ad84c2ba941c0e7626b953864f9632c745cb2171da4bc4411e40` |
| `val.csv` | `data/cmu_mosei/raw/val.csv` | 552,163 | `6fb286af708543aa8c3bab193803fee78d36acf96a7fe07d401a21a0d3773778` |
| `test.csv` | `data/cmu_mosei/raw/test.csv` | 1,350,962 | `8723f3b6f1aac552dc6c2d2472d8e1329cf8f7b3b13ca51e3fa13b797b702b71` |
| `data_manifest.json` | `data/cmu_mosei/data_manifest.json` | 1,705,676 | `cb77d74b463aa754c562d228c785a1d516ef1d0f4c331459e1c09b6675f74f15` |

### 2.2 Cached Split Tensors
Each split directory under `data/cmu_mosei/features_cache/{train, val, cal, test}/` contains:
- `text.pt`: 768-D float32 tensor
- `audio.pt`: 74-D float32 tensor (COVAREP)
- `visual.pt`: 35-D float32 tensor (FACET)
- `labels.pt`: 1-D long tensor
- `meta.pt`: Serialized utterance metadata dictionary

---

## 3. Speaker Identity Defect Correction

> [!WARNING]
> **Speaker Proxy Limitation Disclosure:**  
> The CMU-MOSEI dataset raw annotations provide YouTube video IDs and timestamp boundaries, but do NOT provide verified human speaker biometrics or unique speaker identifiers.  
> Earlier iterations constructed `spk_id = f"mosei_spk_{vid[:8]}"`, which was a deterministic 1:1 transformation of the video ID prefix. Consequently, claims of "0 shared speakers" were an artifact of video disjointness.  
> Tri-SACTA V13.2 formally amends the identity schema:
> - `video_group_id`: YouTube video ID (cluster key for all block bootstrap resamples).
> - `speaker_id_verified`: `null` (unverified).
> - `speaker_proxy`: `f"mosei_spk_{vid[:8]}"` (legacy proxy).
> - `speaker_overlap_status`: `"UNKNOWN"`.
> 
> Statistical independence guarantees are strictly claimed at the **video group** level, not at the human speaker level.

---

## 4. Candidate Unexposed Test Pool Verification

To ensure strict scientific integrity before Phase II:
- The raw test set in `mosei.pkl` contains 4,643 utterances.
- 500 utterances were previously extracted into the historical `test` cache and evaluated across V7–V13.
- An additional 12 utterances in `raw_test` belong to the same 58 video groups as those 500 items and are excluded to prevent intra-video cluster leakage.
- Exactly **4,131 utterances across 617 unique video clusters** remain untouched, with **ZERO video overlap** with any historical split (`train`, `val`, `cal`, or `test`).
- **Phase I Isolation Guarantee:** No model inference, feature extraction, or threshold selection has been performed on these 4,131 samples. Only metadata was inspected for eligibility audit.
