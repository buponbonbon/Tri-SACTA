# V12.04: Raw Media Ingestion Diagnosis and Extractor Compatibility

**Author:** Antigravity 3.8 Systems Research & Verification Lead  
**Date:** 2026-11-23  
**Status:** Completed & Diagnosed  
**Artifact:** [results/v12_systems/raw_media/stage_timings.json](file:///c:/Users/ASUS/Desktop/Tri-SACTA/results/v12_systems/raw_media/stage_timings.json)  
**Profiling Script:** [scripts/v12_profile_systems_runtime.py](file:///c:/Users/ASUS/Desktop/Tri-SACTA/scripts/v12_profile_systems_runtime.py)

---

## 1. Physical Extraction Reality: Legacy vs Modern Pipelines

A critical finding established across V10, V11, and V12 audits is the fundamental distinction between **pre-extracted feature evaluation** and **raw multimedia streaming**:

1. **The Legacy Pipeline (CMU-MOSEI Standard):**
   - **Audio (74D):** Extracted via the `COVAREP` MATLAB toolbox (glottal flow parameters, harmonic-to-noise ratios, MFCCs). COVAREP relies on legacy MATLAB binary dependencies that are non-trivial to execute headlessly in modern containerized Linux/Windows microservices.
   - **Visual (35D):** Extracted via `FACET` (iMotions commercial software). FACET is proprietary closed-source commercial software and is legally unavailable for redistribution or automated container building.
   - **Text (768D):** Extracted via mDeBERTa-v3-Base tokenization and transformer feature projection.

2. **The Modern Extractor Incompatibility Trap (Track N vs Track L):**
   - Attempting to replace COVAREP with `wav2vec2` or FACET with `MediaPipe / OpenFace` produces an immediate, catastrophic feature distribution mismatch when fed into models trained on COVAREP/FACET.
   - Any deployment using modern open extractors constitutes an entirely new model training protocol (**Track N**), requiring retraining all baselines, re-calibrating all thresholds, and establishing new benchmark splits.

---

## 2. End-to-End Pipeline Stage Timing Breakdown

Profiling was conducted on an NVIDIA GeForce RTX 2050 Laptop GPU (16 GB RAM, 6 CPU cores, Windows 11):

| Pipeline Stage | Implementation | Wall-Clock Time | % of End-to-End Ingestion |
| :--- | :--- | :---: | :---: |
| **Video Demux & Frame Decoding** | OpenCV (mp4v, 30 fps, 90 frames) | 46.33 ms | ~6.6% |
| **Audio Decoding & Acoustic FFT** | NumPy / SciPy signal processing | 2.43 ms | ~0.3% |
| **Text Tokenization & Lookup** | Tokenizer & DeBERTa embedding lookup | 1.83 ms | ~0.3% |
| **Neural Forward Pass (Multimodal)**| PyTorch GPU cross-attention backbone | 13.03 ms (warm) / 653 ms (cold) | ~92.8% (cold) |
| **Total Pipeline (Cold)** | End-to-end ingestion | 704.52 ms | 100.0% |

### Key Systems Insight
In earlier literature, claiming $> 200$ FPS inference based strictly on a $\sim 4.5$ ms neural forward pass is a misleading representation of real-world multimedia systems. When full container demuxing, video frame decoding, and audio buffer management are accounted for, end-to-end throughput is bounded by multimedia I/O and preprocessing pipelines.
