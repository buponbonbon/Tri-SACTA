# Tri-SACTA V13.2: Distributed Runtime Architecture & Fair Transport Baselines

**Audit Timestamp:** 2026-11-24 UTC  
**Active Branch:** `research/v13.2-final-systems-validation-release-candidate`  
**Components:** [distributed_providers.py](file:///c:/Users/ASUS/Desktop/Tri-SACTA/src/service/distributed_providers.py) & [distributed_coordinator.py](file:///c:/Users/ASUS/Desktop/Tri-SACTA/src/service/distributed_coordinator.py)  
**Test Suite:** [test_v13_2_runtime_safety.py](file:///c:/Users/ASUS/Desktop/Tri-SACTA/tests/test_v13_2_runtime_safety.py)

---

## 1. System Topology & Transport Scope

Tri-SACTA evaluates feature-level edge-cloud multimodal acquisition:
- **Local Edge Coordinator:** Hosts the local unimodal Text feature and the common PyTorch predictor bank.
- **Remote Providers:** Standalone TCP socket servers hosting acoustic features (`AudioProvider` on port 8765) and visual features (`VisualProvider` on port 8766).
- **Physical Transport:** Communicates via standard TCP stream sockets using an explicit binary framed protocol.

> [!NOTE]
> **Transport Scope Disclosure:**  
> Measurements reflect physical loopback TCP stream transactions with pre-extracted feature tensors. They do NOT represent raw-media ingestion (e.g., MP4 decoding, video frame capture, or Opus audio decompression).  
> Wire byte metrics measure **application-protocol bytes** (request framing + metadata JSON + float32 binary payload). Link-layer (IP/TCP) headers add approximately 40–60 bytes per packet.

---

## 2. P0 Defect: Baseline Transport Fairness Rectification

### Defect Identified in V13.1
In V13.1's distributed benchmark, Always-On Full Fusion and Static TA were implemented as sequential mutations of Text inference:
1. They executed a full forward pass on local Text.
2. They fetched remote modalities sequentially.
3. They executed a *second* forward pass on TAV or TA.

This unfairly penalized Always-On and Static TA with a redundant inference pass (inflating their measured wall time by ~0.24 ms) and forced serial socket queries.

### Rectification in V13.2
In Tri-SACTA V13.2:
- **Always-On (`always_on`):** Fetches Audio and Visual remotely, then executes **exactly one direct forward pass on TAV**.
- **Static TA (`static_ta`):** Fetches Audio remotely, then executes **exactly one direct forward pass on TA**.
- **Sequential Confidence (`sequential_confidence`):** Pays the intermediate inference passes and sequential branching overhead:
  - Text inference -> threshold check -> Audio fetch -> TA inference -> threshold check -> Visual fetch -> TAV inference.

This ensures a strictly fair, unskewed wall-clock and latency comparison between static and adaptive paradigms.

---

## 3. Protocol Safety & Defect Hardening

1. **Strict Sample ID Validation (NO Modulo Indexing):**
   - V13.1 used `sample_idx = int(req["sample_idx"]) % total_samples`. If a malformed ID was queried, it silently returned a completely different valid utterance!
   - V13.2 enforces strict bounds checking: requests with $s < 0$ or $s \ge N$ return `400 Bad Request` with `OUT_OF_BOUNDS_SAMPLE_ID`.
2. **Exact-Length TCP Reads (`recv_exact`):**
   - Stream sockets handle fragmentation and network buffering via dedicated loops for both the 12-byte framing header and the binary payload.
3. **Finite Float Verification:**
   - Serialized feature payloads are checked via `np.all(np.isfinite(feat_np))`. If NaN or Inf values are detected, transmission is halted with `NON_FINITE_FEATURES`.
4. **Monotonic Deadlines & Latency Reserve:**
   - Deadlines are tracked via `time.perf_counter()`.
   - Remaining time is computed before each network hop:
     $$\Delta t_{\text{remaining}} = t_{\text{deadline\_monotonic}} - t_{\text{now}}$$
   - If $\Delta t_{\text{remaining}} \le 1.0\text{ ms}$, remote fetching is aborted, setting `status = "DEADLINE_EXHAUSTED"` and serving the query from local features.
