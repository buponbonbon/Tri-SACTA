# V11-03: RAW-INPUT END-TO-END PIPELINE FEASIBILITY AUDIT

**Audit Date:** 2026-11-23  
**Auditor:** Antigravity 3.8 Independent Scientific Auditor  
**Document Code:** `TRI-SACTA-V11-03`  
**Scope:** Investigation of feasibility for a raw-input ingestion path (raw video/audio $\to$ prediction)

---

## 1. Inventory of Raw Media Assets

A complete filesystem scan of the workspace was conducted:
- Raw video files (`.mp4`, `.avi`, `.mkv`): **0 found**
- Raw audio files (`.wav`, `.mp3`, `.flac`, `.m4a`): **0 found**
- Status: **`RAW_DATA_UNAVAILABLE`**.

The repository stores only offline pre-extracted feature tensors in `.pt` format (`data/cmu_mosei/features_cache/`), originally extracted by the CMU Multimodal SDK / previous academic benchmarks.

---

## 2. Extraction Toolchain Audit

To determine whether raw audio/video clips could be ingested if acquired, the runtime environment was audited for required extraction toolchains:

| Required Extractor | Target Modality & Dimensionality | Software Package | Environment Status | Feasibility Impact |
|---|---|---|:---:|---|
| **COVAREP** | Audio (74D Voice Quality / Prosody) | MATLAB / C++ binary | **NOT INSTALLED** | Fatal: Cannot extract 74D acoustic features |
| **FACET / OpenFace** | Visual (35D Facial Action Units) | OpenFace C++ / Proprietary | **NOT INSTALLED** | Fatal: Cannot extract 35D visual AU features |
| **Whisper / Wav2Vec2** | Text/ASR (Transcription Alignment) | OpenAI Whisper / HuggingFace | **NOT INSTALLED** | Fatal: Cannot generate automated ASR transcripts |
| **FFmpeg / Librosa** | Media decoding and reslicing | `ffmpeg`, `librosa` | **NOT INSTALLED** | Fatal: Cannot decode video containers or audio streams |
| **Status Verdict** | — | — | — | **`FEATURE_INCOMPATIBLE`** |

---

## 3. Forward Pass vs True End-to-End Latency

A common misconception in the historical Tri-SACTA documentation was reporting forward pass speed on cached `.pt` tensors (~4.32 ms/sample) as "real-time throughput (>230 FPS)".

In a genuine end-to-end media pipeline:
1. Video decoding & frame extraction: ~15–30 ms/sec of video.
2. OpenFace/FACET landmark detection: ~80–150 ms/sec on CPU.
3. Audio decoding & COVAREP feature extraction: ~50–100 ms/sec.
4. Speech-to-text ASR (e.g. Whisper-small): ~120–250 ms/sec.
5. Neural fusion forward pass: ~4.3 ms/sample.

The neural model inference represents less than **2%** of the true end-to-end wall-clock latency! Claiming real-time streaming capability based solely on the 4.3 ms cached tensor forward pass while lacking the front-end extraction stack is a major scientific misrepresentation.

---

## 4. Phase C Verdict

**Verdict:** **`BLOCKED (RAW_DATA_UNAVAILABLE & FEATURE_INCOMPATIBLE)`**.  
A genuine raw-input-to-prediction pipeline cannot be implemented without:
1. Licensing and downloading raw CMU-MOSEI video files (>150 GB).
2. Setting up MATLAB/C++ COVAREP and OpenFace extraction containers.
3. Retraining the backbone if modern neural extractors (e.g. Whisper, HuBERT, CLIP) are substituted.
This is far outside the bounded scope of Tri-SACTA and constitutes a completely different multi-month research initiative.
