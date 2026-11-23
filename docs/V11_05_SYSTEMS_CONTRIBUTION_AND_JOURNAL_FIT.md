# V11-05: SYSTEMS CONTRIBUTION AUDIT AND JOURNAL-FIT ANALYSIS

**Audit Date:** 2026-11-23  
**Auditor:** Antigravity 3.8 Independent Scientific Auditor  
**Document Code:** `TRI-SACTA-V11-05`  
**Mandatory Publication Constraints:**
1. SCImago Q1 or Q2 throughout 2023, 2024, 2025 OR Q1 in 2025.
2. Verified $0 mandatory publication fee (Subscription route with no mandatory page/color/submission charges).
3. No fabricated acceptance rates or probabilities without publisher-official sources.

---

## 1. Candidate Venues Detailed Assessment

### 1.1. Multimedia Tools and Applications (MTAP) — Springer Nature
- **SJR Qualification:**
  - 2023: Q1 (Media Technology, Software)
  - 2024: Q1 (Media Technology)
  - 2025: Q1 (Media Technology)
  - **Status: ELIGIBLE**
- **Mandatory Fees:**
  - Hybrid Journal: Open Access APC is $3,690 / €2,890 / £2,490 (optional).
  - Traditional Subscription Route: **$0 Mandatory APC**.
  - Color in print / page charges: Color figures free in electronic publication.
  - **Status: VERIFIED $0 MANDATORY FEE (SUBSCRIPTION ROUTE)**
- **Official Acceptance Rate:** Not publicly disclosed by Springer Nature on journal homepage.
- **Review Outlook for Current Tri-SACTA:**
  - **Desk-Reject Risk: HIGH.**
  - **Root Cause:** MSA papers in MTAP require rigorous empirical validation across multiple datasets (e.g. MOSI, MOSEI, CH-SIMS). A submission based on an offline-cached feature set where multimodal fusion underperforms text-only baselines and where TTA updates degrade accuracy will be rejected for lack of demonstrated utility.

### 1.2. Multimedia Systems (MMSYS) — Springer Nature
- **SJR Qualification:**
  - 2023–2025: Q1 (Computer Science Applications / Media Technology)
  - **Status: ELIGIBLE**
- **Mandatory Fees:**
  - Hybrid Journal: Traditional Subscription Route: **$0 Mandatory APC**.
- **Review Outlook for Current Tri-SACTA:**
  - **Desk-Reject Risk: VERY HIGH.**
  - **Root Cause:** Direct collision with `RegCal-Net` (published in *Multimedia Systems* in September 2026, DOI `10.1007/s00530-026-02653-8`). MMSYS demands deep systems/architectural innovation or superior empirical accuracy.

### 1.3. Signal, Image and Video Processing (SIVP) — Springer Nature
- **SJR Qualification:**
  - 2023–2025: Q2 (Signal Processing, Computer Vision)
  - **Status: ELIGIBLE**
- **Mandatory Fees:** Traditional Subscription Route: **$0 Mandatory APC**.
- **Review Outlook for Current Tri-SACTA:**
  - **Desk-Reject Risk: HIGH.**
  - **Root Cause:** SIVP focuses on signal-level innovations. Tri-SACTA operates on pre-extracted feature embeddings, not raw signal processing.

### 1.4. Computing — Springer Nature
- **SJR Qualification:**
  - 2023: Q1, 2024: Q1, 2025: Q2
  - **Status: ELIGIBLE**
- **Review Outlook for Current Tri-SACTA:**
  - Requires substantial software engineering, reproducibility, or distributed systems contributions. Wrapping a pre-extracted neural model in a pipeline script does not meet the systems contribution threshold.

---

## 2. Summary of Acceptance Risk Across All Scenarios

| Submission Scenario | MTAP (Springer Q1) | Multimedia Systems (Springer Q1) | SIVP (Springer Q2) | Computing (Springer Q2) |
|---|:---:|:---:|:---:|:---:|
| **S0: As-Is Current Snapshot** | Desk-Reject (100%) | Desk-Reject (100%) | Desk-Reject (100%) | Desk-Reject (100%) |
| **S1: Web/CLI Pipeline Wrapper**| Desk-Reject (>90%) | Desk-Reject (>95%) | Desk-Reject (>90%) | Desk-Reject (>85%) |
| **S2: Safe Suppression Narrative**| Reject (Lack Novelty vs RegCal-Net)| Reject (Duplicate vs RegCal-Net) | Reject (Pre-extracted Features) | Reject (Not Systems Scope) |

**Conclusion:** Neither a thin software wrapper nor a re-badged failure analysis paper can achieve acceptance in eligible Q1/Q2 venues without fundamentally retraining the backbone on authentic modern representations and solving the multimodal penalty.
