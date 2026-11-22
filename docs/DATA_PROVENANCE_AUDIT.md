# FORENSIC DATA PROVENANCE & AUTHENTICITY AUDIT (TRI-SACTA V5)

**Audit Date:** 2026-11-22  
**Auditor:** Agent 3 (Independent Data Integrity Auditor) & Agent 6 (Adversarial Q1 Reviewer)  
**Branch:** `research/v5-provenance-enforcement`  
**Reference Commit:** Base `52ad9b21f66bfd9667baffaf259fe56b8112cb41` $\to$ V5 HEAD  

---

## 1. Executive Summary & Forensic Audit of V5 Enforcement

In V5, data provenance enforcement was systematically audited and hardened against statistical pollution, data leakage, and silent synthetic fallbacks.

### Key Forensic Hardening Actions:
1. **Aggregator Quarantine Filtering (V5-01):** `scripts/05_aggregate_results.py` automatically detects and excludes any result file residing under quarantined paths (`quarantined_v3_fixtures/`, etc.) or records tagged with `"quarantined": True` or `"is_synthetic_fixture": True`.
2. **Complete Segregation of Fixtures (V5-02):** Removed all legacy random generation functions (`build_pilot_real_cache()`, `--generate-pilot-cache`) from `scripts/extract_features_sequential.py`. Synthetic fixture generation is isolated to `src/data/fixtures.py` and tagged as `synthetic_fixture`. The real extractor strictly fails closed (`FileNotFoundError`) if raw media or annotations are missing.
3. **Cryptographic Manifest & Checksum Verification (V5-03):** Removed the 50-row cutoff. `validate_split_integrity` computes and verifies SHA-256 hashes across 100% of items and referenced feature tensors. Added path traversal defenses (`validate_safe_path`) and canonical whole-manifest hashing.
4. **Strict Manifest-to-Tensor Row Join (V5-04):** Dataloader enforces a 1-to-1 join on `sample_id`, `speaker_id`, and `mapped_label`, failing closed if any sample mismatch or count discrepancy occurs.
5. **Cryptographic Checkpoint Binding (V5-05):** Checkpoints and spectral statistics files embed schema v5 provenance headers (`manifest_hash`, `dataset_name`, `data_mode`, `seed`), enforced by the benchmark runner.
6. **MELD Dialogue-Disjoint Splitting:** Resolved local dialogue ID overlap between MELD train and dev splits by prepending split identifiers (`meld_{split}_diag_{id:04d}`). Partitioned dev set into dialogue-disjoint validation (even dialogues) and calibration (odd dialogues) splits.

---

## 2. Quarantine Register

The following artifacts remain strictly quarantined and excluded from all empirical real-mode reporting:

| Quarantined Artifact Path | Origin | Original False Label | Quarantine Status | Audit Reason |
|---|---|---|---|---|
| `data/quarantined_v3_fixtures/cmu_mosei/` | V3 `extract_features_sequential.py` | "verified real CMU-MOSEI cache" | **BANNED FROM REAL BENCHMARK** | Random Gaussian tensors (`randn`), artificial `vid_trn_*` IDs |
| `data/quarantined_v3_fixtures/meld/` | V3 `extract_features_sequential.py` | "verified real MELD cache" | **BANNED FROM REAL BENCHMARK** | Random Gaussian tensors (`randn`), artificial `spk_meld_*` IDs |
| `checkpoints/quarantined_v3_fixtures/cmu_mosei/` | V3 `01_train_source.py` | "source_model.pt (real)" | **BANNED FROM REAL BENCHMARK** | Trained on random synthetic tensors |
| `results/quarantined_v3_fixtures/cmu_mosei/` | V3 `02_run_stream_benchmark.py` | "real prequential benchmark" | **RETRACTED** | 12.08% accuracy reflecting random noise evaluation |
| `results/quarantined_v3_fixtures/summary_cmu_mosei_real_prequential.md` | V3 `05_aggregate_results.py` | "real summary" | **RETRACTED** | Misleading publication table entry |

---

## 3. Authentic Dataset Acquisition & Provenance Verification

Authentic annotations are verified on disk with SHA-256 integrity:

### 3.1. Official MELD Dataset (Declare-Lab)
- **Source Repository:** `https://github.com/declare-lab/MELD`
- **Citation:** Poria et al., "MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations", ACL 2019.
- **Acquired Files & Checksums:**
  - `data/meld/raw/train_sent_emo.csv`: 9,989 utterances, 260 speakers, 1,038 dialogues.  
    **SHA-256:** `d2fa2d6529cfb9f9392e22c814b7261d763bc5cba799309cbffc9a418beaf687`
  - `data/meld/raw/dev_sent_emo.csv`: 1,109 utterances, 47 speakers, 114 dialogues.  
    **SHA-256:** `2e89c6f8aa1819389233633198007a8fc52cff7ebbb7d612e4307a049d5c58ca`
  - `data/meld/raw/test_sent_emo.csv`: 2,610 utterances, 100 speakers, 280 dialogues.  
    **SHA-256:** `8d37103938f7a933f78d65ebda9ce8034db6d953ceeb9304325a743c39121c76`
- **Split Disjointness:** Dialogue-disjoint (Train dialogues 0–1038; Dev partitioned into even-dialogue Val and odd-dialogue Cal; Test dialogues 0–279).
- **V5 Authentic Manifest:** `data/meld/data_manifest.json` (240 genuine items, Canonical Manifest Hash: `5ffc5e3e67de`).

### 3.2. Official CMU-MOSEI Dataset (MultimodalSDK / HF)
- **Source Repository:** `https://huggingface.co/datasets/vintp/CMU-Mosei-text`
- **Citation:** Zadeh et al., "Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph", ACL 2018.
- **Acquired Files & Checksums:**
  - `data/cmu_mosei/raw/train.csv`: 16,274 utterances, 2,244 distinct YouTube videos.  
    **SHA-256:** `e548cbf003c6218d6db3163155799c7d42878c772cb5f69be84a6c2f7dc22791`
  - `data/cmu_mosei/raw/val.csv`: 1,861 utterances, 298 distinct YouTube videos.  
    **SHA-256:** `6fb286af7085777a9446d3283be313c4c95f13c631b091f0915f06117eb8fe6d`
  - `data/cmu_mosei/raw/test.csv`: 4,653 utterances, 675 distinct YouTube videos.  
    **SHA-256:** `8723f3b6f1aae7528e100f9593a105eb0eb0e461b2fc80c80088ad1779ba3a94`
- **Video Disjointness (0% Cross-Split Overlap):**
  - Train vs Val video overlap: 0 videos (0.0%).
  - Train vs Test video overlap: 0 videos (0.0%).
  - Val vs Test video overlap: 0 videos (0.0%).
- **V5 Authentic Manifest:** `data/cmu_mosei/data_manifest.json` (240 genuine items, Canonical Manifest Hash: `4c111d27053d`).

---

## 4. Hardware Constraints & Operational Status

In accordance with Mandatory Rule 3 of the Scientific Integrity Protocol:
> *"If required data, dependencies, permissions or computational resources are unavailable, report BLOCKED. Do not invent substitutes to satisfy acceptance criteria."*

| Asset | Local Status | Download Requirement | Constraint / Blocker | Scientific Resolution |
|---|---|---|---|---|
| **Text Annotations & ASR Transcripts** | **AVAILABLE** | Verified on disk | None | Encoded via genuine DeBERTa-v3 forward passes (`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`) |
| **MELD Raw Audio / Video** | **BLOCKED** | `MELD.Raw.tar.gz` (10.87 GB) | Local disk budget & turn execution timeout | Declared `BLOCKED`. No random Gaussian vectors or synthetic placeholders substituted. |
| **CMU-MOSEI Computational Sequences** | **BLOCKED** | OpenFace2 & COVAREP CSDs (~15 GB) | Network bandwidth & storage | Declared `BLOCKED`. |

**Scientific Integrity Statement:** No synthetic noise or random vectors are used to simulate missing multimodal media in real mode. Real-data results are strictly reported on the verified authentic text modality until full raw multimodal assets are acquired.
