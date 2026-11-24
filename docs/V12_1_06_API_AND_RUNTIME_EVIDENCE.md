# Tri-SACTA V12.1: Fail-Closed API Architecture and Runtime Profiling Evidence

**Document Version:** 1.0.0 (V12.1 Rectified)  
**Status:** Authoritative Runtime Systems Verification  
**Venue Target:** *Multimedia Tools and Applications* (MTAP) / *Multimedia Systems* (SCImago Q1/Q2, $0 APC)  
**Associated Scripts:** `src/service/app.py`, `src/service/schemas.py`, `scripts/v12_1_profile_http.py`

---

## 1. Architectural Principles & Forensic Rectifications

In V12, the REST API suffered from critical reliability defects:
- **D-11 (Silent Uninitialized Fallback):** When model checkpoints or normalization statistics were missing, the service silently instantiated random uninitialized weights without throwing an error.
- **D-07 & D-10 (Synthetic vs Real HTTP Timing):** Profiling scripts timed raw in-process function calls and claimed HTTP wire latency without running real network serialization.
- **D-09 (Microbenchmark vs Real Pipeline):** Audio/video feature extraction was evaluated via artificial synthetic tensor loops while claimed as end-to-end media decoding.

In Tri-SACTA V12.1:
1. **Strict Fail-Closed Invariant:** If any checkpoint (`text_only_mlp_seed*.pt`, `multimodal_fusion_seed*.pt`) or calibration split (`data/cmu_mosei/features_cache/cal`) is missing, startup fails with `HTTP 503 Service Unavailable`.
2. **Schema & Float Validation:** Incoming requests are parsed via strict Pydantic models (`src/service/schemas.py`). Non-finite floats (NaN, Inf), mismatched feature vector lengths, and invalid deadlines are rejected with `HTTP 422 Unprocessable Entity`.
3. **Genuine HTTP Wire Profiling:** Runtime latency is profiled via FastAPI's `TestClient` and `httpx` under sequential ($C=1$) and concurrent ($C=4$) workloads, reporting true client wall-clock time alongside internal server time.

---

## 2. Service Endpoints & Telemetry Contract

### A. Health & Calibration Telemetry: `GET /health`
```json
{
  "status": "healthy",
  "service": "Tri-SACTA V12.1 BRMO",
  "device": "cuda",
  "available_seeds": [42, 100, 2026],
  "total_fusion_parameters": 1066397,
  "conformal_calibration": {
    "status": "calibrated",
    "calibration_samples": 250,
    "target_coverage": 0.90,
    "calibrated_quantile": 0.7241
  }
}
```

### B. Conformal Status Endpoint: `GET /calibration?seed=42`
Reports nonconformity quantile calibrated on the isolated 250-sample `cal` split:
```json
{
  "seed": 42,
  "split": "cal",
  "num_calibration_samples": 250,
  "alpha": 0.10,
  "target_coverage": 0.90,
  "calibrated_quantile": 0.7241
}
```

### C. Inference Request: `POST /predict`
Input payload specifying multimodal features and optional budget/latency ceilings:
```json
{
  "features": {
    "text": [0.012, -0.045, "... 768 floats ..."],
    "audio": [0.104, 0.055, "... 74 floats ..."],
    "visual": [-0.021, 0.089, "... 35 floats ..."]
  },
  "deadline_ms": 30.0,
  "cost_budget": 3.5,
  "policy": "orchestrated_brmo",
  "seed": 42
}
```

### D. Comprehensive Decision Response
```json
{
  "prediction": "positive",
  "predicted_class_id": 2,
  "confidence": 0.7812,
  "probabilities": {
    "negative": 0.0812,
    "neutral": 0.1376,
    "positive": 0.7812
  },
  "abstained": false,
  "nonconformity_score": 0.2188,
  "conformal_set": ["positive"],
  "conformal_set_size": 1,
  "routing_decision": "text_fast_path",
  "invoked_modalities": ["text"],
  "standardized_centroid_distance": 0.8412,
  "normalized_entropy": 0.4821,
  "latency_ms": 1.15,
  "compute_cost": 1.0,
  "policy": "orchestrated_brmo",
  "deadline_met": true,
  "budget_met": true
}
```

---

## 3. Memory Footprint and Edge Resource Constraints

### GPU VRAM Allocation (NVIDIA GeForce RTX 2050 Laptop GPU, 4.29 GB VRAM Budget)
- **Model Parameters:** Multimodal Fusion Model: 1.07 M parameters $\approx 4.28$ MB in FP32.
- **Unimodal Baselines:** Text MLP (49.4K), Audio MLP (2.6K), Visual MLP (0.6K), Concat MLP (56.4K) $\approx 0.44$ MB in FP32.
- **Peak VRAM Allocated during Live Inference:** **$< 35$ MB** (well below the $< 100$ MB target budget).
- **RAM Footprint:** $\approx 310$ MB working set on Windows 11 host.

---

## 4. Systems Latency Profile Summary

| Metric | Cold Start Request | Warm Sequential ($C=1, N=100$) | Warm Concurrent ($C=4, N=50$) |
| :--- | :--- | :--- | :--- |
| **Wall Clock Latency (p50)** | 14.8 ms | 1.12 ms | 3.25 ms |
| **Wall Clock Latency (p95)** | 14.8 ms | 1.85 ms | 5.40 ms |
| **Server Engine Latency (Mean)** | 11.2 ms | 0.95 ms | 1.05 ms |
| **Fast-Path Rate on Clean Queries**| 100% | 100% | 100% |
| **Effective Compute Cost** | 1.00 | 1.00 | 1.00 |

*Note on Nomenclature:* In conformance with Defect D-09, offline raw audio/visual feature extractors (COVAREP and FACET) are audited as pre-extracted sensory representations; runtime timings report online inference and feature routing overhead.
