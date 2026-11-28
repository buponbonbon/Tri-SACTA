# Tri-SACTA V13.2: Prior Art Overlap Analysis, Novelty Assessment & Evidence Ledger

**Novelty Status:** `OVERLAP_IDENTIFIED`  
**Source Artifact:** [novelty_assessment.json](../results/v13_2/validation/novelty_assessment.json)  
**Target Venues (Conditional, Subscription $0 Mandatory APC Track):** *Multimedia Tools and Applications* (Springer SJR 2024/2025 ~0.777–0.798, Q1 Media Technology / Q2 CS Networks; hybrid $0 mandatory APC subscription vs optional Gold OA ~$3,190) or *Multimedia Systems* (Springer SJR ~0.669; hybrid $0 mandatory APC vs optional Gold OA ~$3,490). *No acceptance guarantee claimed.*

## 1. Verified Prior Art Matrix

| Work | Identifier | Verified Text Scope | Action Space & Method | Overlap & Scope Differentiation |
| :--- | :--- | :--- | :--- | :--- |
| A2MT (Kossen et al., TMLR 2023) | `arXiv:2211.05039` | Published TMLR paper & OpenReview ablations | Sequential multimodal temporal modality acquisition via Perceiver IO + RL | Already demonstrated sequential multimodal acquisition and observed RL policies often collapse to simple marginal-cost heuristics. |
| DynMM (Xue & Marculescu, CVPRW 2023) | `DOI:10.1109/CVPRW59228.2023.00256` | CVPRW proceedings metadata & abstract/repo | Dynamic instance-wise modality/fusion path selection on multimodal benchmarks including CMU-MOSEI | Establishes adaptive modality gating on CMU-MOSEI; Tri-SACTA differs only by decoupling providers across TCP sockets. |
| SemARC (Mohapatra et al., 2026 preprint) | `arXiv:2610.07466` | arXiv metadata & summary (full PDF not ingested locally) | Sequential modality aggregator (SeMA) + Adaptive Runtime Controller (ARC) with marginal utility priors & Q-learning | Close methodological overlap on sequential fusion + marginal utility priors under missingness; Tri-SACTA contributes distributed TCP protocol byte accounting and fault handling rather than novel fusion/RL. |
| MoA-Off (Yang et al., 2025 preprint) & MSAO (Yang et al., 2026 preprint) | `arXiv:2509.16995 / arXiv:2604.02945` | arXiv cs.DC abstracts/metadata (full PDF not ingested locally) | Edge-cloud modality complexity/sparsity-aware offloading with speculative execution for MLLMs | Substantial overlap on adaptive edge-cloud modality offloading under bandwidth/latency constraints; Tri-SACTA narrows scope to pre-extracted sentiment MLPs with fail-closed TCP socket fault injection. |
| SPINN (Laskaridis et al., ACM MobiCom 2020) | `DOI:10.1145/3372224.3419194` | ACM MobiCom published paper metadata | Progressive early-exit edge-cloud neural inference under dynamic network conditions and deadlines | Already established progressive confidence early exit over real edge-cloud networks under latency deadlines. |
| Wu et al. (Electronics 2026) & RegCal-Net (Multimedia Systems 2026) | `DOI:10.3390/electronics15163624 / DOI:10.1007/s00530-026-02653-8` | Journal metadata & abstracts | Centralized missing-modality evidential/register fusion | Orthogonal model-architecture works; insufficient alone to establish distributed runtime novelty. |

## 2. Explicit Negative Disclosures & Limitations

1. **Algorithmic Novelty:** None claimed. `SequentialConfidence_0.55_0.45` is standard softmax-max confidence thresholding, retained over `ConditionalUtilityRouter` after empirical defeat.
2. **Tail-Latency & One-Shot Trade-Offs:** `Static_TA_Direct` achieves lower persistent p95/p99 latency, and `OneShot_ConfidenceRouting_0.55` outperforms sequential routing on non-persistent TCP handshakes while achieving nearly matched bytes (`+14.3%` byte difference, CI `[+12.5%, +15.8%]`).
3. **Single-Host Loopback Scope:** Measurements cover localhost loopback TCP socket application-protocol framing bytes and wall time, NOT WAN links, two-device wireless deployments, hardware energy, or raw video/audio codecs.
4. **No Verified Human Speaker IDs:** `speaker_id_verified = null`, `speaker_overlap_status = UNKNOWN`. Disjointness is strictly guaranteed at YouTube `video_group_id` cluster granularity.
5. **Pre-Holdout Status:** Phase II confirmatory results are `NOT_RUN` (`PENDING_AUTHOR_APPROVAL`).
