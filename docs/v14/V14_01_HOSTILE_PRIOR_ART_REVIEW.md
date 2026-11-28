# V14-01: Hostile Prior-Art Review

| Work | Venue & Year | State & Action Space | Cost / Network / Deadline | Overlap & Unresolved Gap |
|---|---|---|---|---|
| **DynMM** (Xue & Marculescu) | CVPRW 2023 | Input-dependent modality & fusion gating | Compute/FLOPs loss; no TCP RTT or deadlines | Subsumes static/compute-only modality skipping; does **not** model asymmetric harm `H(s,a)` under sequential vs parallel TCP deadlines |
| **A2MT** (Kossen et al.) | TMLR 2023 | Temporal active multimodal acquisition (Perceiver IO) | Nominal acquisition cost; no parallel-vs-sequential RTT trade-off | Subsumes sequential active feature acquisition; does **not** treat option value vs parallel round-trip latency under service deadlines |
| **SPINN** (Laskaridis et al.) | MobiCom 2020 | Early-exit CNN partitioning (device/cloud) | Network bandwidth & deadline-aware progressive inference | Subsumes progressive edge-cloud early exiting; single-modality CNN partitioning rather than multimodal subset-bank rescue/harm & parallel acquisition |
| **SeMARC** | arXiv / preprint (venue unverified as NeurIPS) | Sequential modality aggregator + adaptive runtime controller | Runtime resource constraints | High conceptual overlap on sequential runtime control; lacks explicit rescue-harm risk decomposition & joint sequential-parallel option-value formulation |
| **Classical AFA / Cascades** (Viola-Jones; Shim et al. JAFA; Janisch et al. RL-AFA) | Classic / AAAI / JMLR | Sequential feature-group acquisition MDPs | Feature acquisition cost | Establishes POMDP/CMDP sequential acquisition; typically assumes additive/monotone feature value or ignores parallel-vs-sequential RTT option value |

**Rejected Novelty Claims:** Renaming confidence routing, adding a scalar cost penalty, replacing thresholds with an MLP, or applying existing AFA to CMU-MOSEI.
