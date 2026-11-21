# FORENSIC DATA PROVENANCE & AUTHENTICITY AUDIT (TRI-SACTA V4)

**Audit Date:** 2026-11-21  
**Auditor:** Agent 3 (Independent Data Integrity Auditor) & Agent 6 (Adversarial Q1 Reviewer)  
**Branch:** `research/v4-data-authenticity`  
**Reference Commit:** `e09978e` $\to$ V4 HEAD  

---

## 1. Executive Summary & Forensic Findings on V3 Pseudo-Real Artifacts

An adversarial inspection of the V3 release identified a **Critical Scientific Integrity Failure**:
1. The method `build_pilot_real_cache()` in `scripts/extract_features_sequential.py` previously generated synthetic random Gaussian vectors (`np.random.randn`) and random discrete integers (`np.random.randint`) for labels, while labeling the output directory as `data/cmu_mosei/` and `data/meld/` with metadata marked `1.0-verified-cache`.
2. Both manifests contained exactly 320 synthetic samples with identical synthetic video IDs (`vid_trn_000` to `vid_trn_009`) and synthetic speaker IDs (`spk_cmu_mosei_trn_000`, `spk_meld_trn_000`).
3. All 640 manifest records had empty `feature_files` and `feature_checksums`.
4. As a direct consequence, the "real-data" benchmark in V3 exhibited chance-level accuracy (12.08% vs 14.28% random guessing on 7 classes), 0.1167 Macro-F1, and an inflated average prediction set size of 6.27 / 7 classes.

**Auditor Action:** All pseudo-real artifacts have been immediately **quarantined**, stripped of any scientific status, and banned from real benchmark evaluation.

---

## 2. Quarantine Register

The following artifacts have been relocated to quarantine storage and marked with immutable `QUARANTINE_NOTICE.md` disclaimers:

| Quarantined Artifact Path | Origin | Original False Label | Quarantine Status | Audit Reason |
|---|---|---|---|---|
| `data/quarantined_v3_fixtures/cmu_mosei/` | V3 `extract_features_sequential.py` | "verified real CMU-MOSEI cache" | **BANNED FROM REAL BENCHMARK** | Random Gaussian tensors (`randn`), artificial `vid_trn_*` IDs |
| `data/quarantined_v3_fixtures/meld/` | V3 `extract_features_sequential.py` | "verified real MELD cache" | **BANNED FROM REAL BENCHMARK** | Random Gaussian tensors (`randn`), artificial `spk_meld_*` IDs |
| `checkpoints/quarantined_v3_fixtures/cmu_mosei/` | V3 `01_train_source.py` | "source_model.pt (real)" | **BANNED FROM REAL BENCHMARK** | Trained on random synthetic tensors |
| `results/quarantined_v3_fixtures/cmu_mosei/` | V3 `02_run_stream_benchmark.py` | "real prequential benchmark" | **RETRACTED** | 12.08% accuracy reflecting random noise evaluation |
| `results/quarantined_v3_fixtures/summary_cmu_mosei_real_prequential.md` | V3 `05_aggregate_results.py` | "real summary" | **RETRACTED** | Misleading publication table entry |

---

## 3. Authentic Dataset Acquisition & Provenance Verification

In V4, genuine dataset acquisition was implemented in [scripts/acquire_authentic_data.py](file:///c:/Users/ASUS/Desktop/Tri-SACTA/scripts/acquire_authentic_data.py). Authentic annotations have been downloaded, verified, and logged:

### 3.1. Official MELD Dataset (Declare-Lab)
- **Source Repository:** `https://github.com/declare-lab/MELD`
- **License / Citation:** Poria et al., "MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations", ACL 2019.
- **Acquired Files & Checksums:**
  - `data/meld/raw/train_sent_emo.csv`: 9,989 utterances, 260 speakers, 1,038 dialogues.  
    **SHA-256:** `d2fa2d6529cfb9f9392e22c814b7261d763bc5cba799309cbffc9a418beaf687` (Size: 1,119,290 bytes)
  - `data/meld/raw/dev_sent_emo.csv`: 1,109 utterances, 47 speakers, 114 dialogues.  
    **SHA-256:** `2e89c6f8aa1819389233633198007a8fc52cff7ebbb7d612e4307a049d5c58ca` (Size: 121,640 bytes)
  - `data/meld/raw/test_sent_emo.csv`: 2,610 utterances, 100 speakers, 280 dialogues.  
    **SHA-256:** `8d37103938f7a933f78d65ebda9ce8034db6d953ceeb9304325a743c39121c76` (Size: 294,526 bytes)
- **Speaker & Dialogue Verification:**
  - Speakers: Genuine Friends TV show character identities (`Chandler`, `Joey`, `Monica`, `Phoebe`, `Rachel`, `Ross`, etc.).
  - Emotion labels (7 classes): `neutral: 0, surprise: 1, fear: 2, sadness: 3, joy: 4, disgust: 5, anger: 6`.
  - Splits: Dialogue-disjoint (Train dialogues 0–1038, Dev dialogues 0–113, Test dialogues 0–279).

### 3.2. Official CMU-MOSEI Dataset (MultimodalSDK / HF)
- **Source Repository:** `https://huggingface.co/datasets/vintp/CMU-Mosei-text`
- **License / Citation:** Zadeh et al., "Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph", ACL 2018.
- **Acquired Files & Checksums:**
  - `data/cmu_mosei/raw/train.csv`: 16,274 utterances, 2,244 distinct YouTube videos.  
    **SHA-256:** `e548cbf003c6218d6db3163155799c7d42878c772cb5f69be84a6c2f7dc22791` (Size: 4,649,662 bytes)
  - `data/cmu_mosei/raw/val.csv`: 1,861 utterances, 298 distinct YouTube videos.  
    **SHA-256:** `6fb286af7085777a9446d3283be313c4c95f13c631b091f0915f06117eb8fe6d` (Size: 552,163 bytes)
  - `data/cmu_mosei/raw/test.csv`: 4,653 utterances, 675 distinct YouTube videos.  
    **SHA-256:** `8723f3b6f1aae7528e100f9593a105eb0eb0e461b2fc80c80088ad1779ba3a94` (Size: 1,350,962 bytes)
- **Video Disjointness Verification (0% Cross-Split Overlap):**
  - Train vs Val video overlap: **0 videos** (0.0%).
  - Train vs Test video overlap: **0 videos** (0.0%).
  - Val vs Test video overlap: **0 videos** (0.0%).
  - Real sentiment scores $\in [-3, +3]$ mapped to 7 standard ordinal classes. Real clean transcripts and real ASR transcripts with word recognition errors.

---

## 4. Hardware Constraints & Current Blockers (Task 12)

In accordance with Mandatory Rule 3 of the Scientific Integrity Protocol:
> *"If required data, dependencies, permissions or computational resources are unavailable, report BLOCKED. Do not invent substitutes to satisfy acceptance criteria."*

### Current Operational Status:

| Asset | Local Status | Download Requirement | Constraint / Blocker | Scientific Resolution |
|---|---|---|---|---|
| **Text Annotations & ASR Transcripts** | **AVAILABLE** | Verified on disk | None | Genuinely encoded via DeBERTa-v3 forward passes |
| **MELD Raw Audio / Video** | **BLOCKED** | `MELD.Raw.tar.gz` (10.87 GB) | Local disk budget & turn execution timeout | Require offline download script; do NOT invent fake audio |
| **CMU-MOSEI Computational Sequences** | **BLOCKED** | OpenFace2 & COVAREP CSDs (~15 GB) | Network bandwidth & storage | Require external drive / pre-extracted cache; do NOT fabricate random features |

**Statement of Integrity:** Under no circumstances will synthetic noise be injected to simulate the un-downloaded 10-15GB audio/video files. Real multimodal benchmarks that evaluate acoustic/visual sensor shifts on full raw media are flagged as **BLOCKED** until external raw assets are mounted or downloaded.
