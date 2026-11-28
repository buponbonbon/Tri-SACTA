# Tri-SACTA V13.1 — Reachable Sequential Oracle & Headroom Diagnostic

**Document Version:** 1.0.0  
**Timestamp (UTC):** 2026-11-24T15:35:00Z  
**Branch:** `research/v13.1-conditional-utility-independent-validation`  
**Artifact References:**
- `results/v13_1/oracle/reachable_oracle_summary.json`
- `results/v13_1/oracle/oracle_frontier.csv`
- `results/v13_1/oracle/gain_decomposition.json`

---

## 1. Oracle Definitions: Free-Choice vs. Reachable Sequential

Tri-SACTA V13 evaluated an unconstrained, label-aware oracle that picked the cheapest modality subset that correctly classified each sample. That oracle could select unimodal Audio ($A$, cost 1.2) or unimodal Visual ($V$, cost 1.3) without paying for Text ($T$, cost 1.0).

In any real sequential acquisition architecture:
1. All queries begin with the base anchor modality ($T$, cost 1.0).
2. If the initial prediction is accepted, the query terminates with cost 1.0.
3. If an expansion is initiated, already-acquired modalities cannot be refunded. Transitioning to Audio incurs cumulative cost $1.0 + 1.2 = 2.2$ ($TA$). Transitioning to Visual incurs cumulative cost $1.0 + 1.3 = 2.3$ ($TV$). Transitioning to both incurs cumulative cost $1.0 + 1.2 + 1.3 = 3.5$ ($TAV$).

In V13.1, we formalize the **Reachable Sequential Oracle**:
- For query $i$:
  - If $T$ is correct: stops at $T$ (cost 1.0).
  - If $T$ is incorrect: searches allowable reachable expansions $\{TA, TV, TAV\}$.
  - If any reachable expansion is correct, it chooses the lowest-cumulative-cost expansion ($TA$ first, then $TV$, then $TAV$).
  - If none rescues the error, it terminates at $T$ (cost 1.0) to avoid wasted budget.

---

## 2. Experimental Headroom Results (Validation Split, 3 Seeds)

Evaluated across all 250 validation utterances $\times$ 3 training seeds ($N = 750$ paired evaluation instances):

| System / Policy | Macro-F1 (Mean $\pm$ Std) | Accuracy (Mean $\pm$ Std) | Mean Cost | Headroom vs Best Static ($TA$) | 95% Clustered CI |
|---|---|---|---|---|---|
| Static $T$ | $0.5681 \pm 0.0071$ | $0.5827 \pm 0.0065$ | 1.0000 | -0.0083 | [-0.0210, +0.0045] |
| Static $A$ | $0.3862 \pm 0.0124$ | $0.4213 \pm 0.0151$ | 1.2000 | -0.1902 | [-0.2241, -0.1560] |
| Static $V$ | $0.3842 \pm 0.0089$ | $0.3907 \pm 0.0112$ | 1.3000 | -0.1922 | [-0.2215, -0.1630] |
| Static $TA$ (Best Static) | **$0.5764 \pm 0.0052$** | **$0.5853 \pm 0.0049$** | 2.2000 | **0.0000** | [0.0000, 0.0000] |
| Static $TV$ | $0.5442 \pm 0.0068$ | $0.5480 \pm 0.0055$ | 2.3000 | -0.0322 | [-0.0480, -0.0165] |
| Static $AV$ | $0.3953 \pm 0.0141$ | $0.4053 \pm 0.0160$ | 2.5000 | -0.1811 | [-0.2150, -0.1470] |
| Static $TAV$ | $0.5678 \pm 0.0061$ | $0.5787 \pm 0.0058$ | 3.5000 | -0.0086 | [-0.0230, +0.0058] |
| **Free-Choice Oracle (V13)** | $0.8811 \pm 0.0041$ | $0.8853 \pm 0.0039$ | 1.1521 | +0.3047 | [+0.2545, +0.3499] |
| **Reachable Sequential Oracle (V13.1)** | **$0.7288 \pm 0.0048$** | **$0.7347 \pm 0.0045$** | **1.2209** | **+0.1524** | **[+0.1254, +0.1846]** |

---

## 3. Transition Gain & Error Decomposition

Across all 750 query instances ($250 \times 3$ seeds):
- **$T$ Correct Initially:** 437 instances (58.27%) $\to$ Terminated at $T$ with zero incremental cost (cost 1.0).
- **$T$ Error Initially:** 313 instances (41.73%):
  - **Rescued by $TA$:** 53 instances (16.93% of errors) $\to$ Acquired Audio (cumulative cost 2.2).
  - **Rescued by $TV$:** 42 instances (13.42% of errors) $\to$ Acquired Visual (cumulative cost 2.3).
  - **Rescued by $TAV$:** 19 instances (6.07% of errors) $\to$ Acquired Audio + Visual (cumulative cost 3.5).
  - **Unrescuable:** 199 instances (63.58% of errors) $\to$ Remained at $T$ (cost 1.0).

Total Errors Rescued: **114** out of 313 (36.42% overall rescue rate).  
Mean Reachable Oracle Cost: **1.2209** nominal units (a $44.5\%$ cost reduction compared to static $TA$'s 2.2 units, while gaining $+0.1524$ Macro-F1).

---

## 4. Reachable Oracle Budget Frontiers

Under varying budget ceilings $B$:
| Budget Ceiling $B$ | Reachable Modalities | Reachable Oracle Macro-F1 | Mean Cost | Headroom vs Best Static ($TA$) |
|---|---|---|---|---|
| $B = 1.0$ | $T$ | 0.5681 | 1.0000 | -0.0083 |
| $B = 1.5$ | $T$ | 0.5681 | 1.0000 | -0.0083 |
| $B = 2.0$ | $T$ | 0.5681 | 1.0000 | -0.0083 |
| $B = 2.2$ | $T, TA$ | 0.6388 | 1.0848 | +0.0624 |
| $B = 2.3$ | $T, TA, TV$ | 0.7045 | 1.1520 | +0.1281 |
| $B = 2.5$ | $T, TA, TV$ | 0.7045 | 1.1520 | +0.1281 |
| $B = 3.5$ | $T, TA, TV, TAV$ | 0.7288 | 1.2209 | +0.1524 |

---

## 5. Gate A Verdict

**Gate A (Headroom & Diagnostic Feasibility): PASS**  
1. **Reachable Headroom:** The reachable sequential oracle achieves a Macro-F1 headroom of **+0.1524** over the best static subset ($TA$), with a 95% clustered video block bootstrap interval of **[+0.1254, +0.1846]**.
2. **Confidence Bound:** The lower confidence bound ($+0.1254$) is strictly positive and well exceeds the preregistered threshold ($0.05$).
3. **Sufficient Observed Events:** 114 rescue events and 199 unrescuable error events were observed, providing an empirical sample to train and calibrate conditional transition estimators in Stage B.
