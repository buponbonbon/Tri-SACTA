# Tri-SACTA V13.2: Prior Art, Systems Differentiation & Claim-Evidence Ledger

**Audit Timestamp:** 2026-11-24 UTC  
**Active Branch:** `research/v13.2-final-systems-validation-release-candidate`  
**Target Manuscript Orientation:** Systems-oriented multimedia resource management and edge-cloud orchestration.  
**Target Journal Candidates:** *Multimedia Tools and Applications* (Springer, SJR Q2, $0 mandatory APC via subscription track), *Multimedia Systems* (Springer), or *Computing* (Springer).

---

## 1. Verified Prior Art Analysis

An exhaustive review of the closest literature was conducted to resolve vague citations and accurately ground Tri-SACTA's contributions.

### 1.1 Evidential Multimodal Fusion: Wu et al. (2026)
* **Citation:** Yubin Wu, Xianxun Zhu, and Huilin Liu, *"Reliability-Aware Multi-Modal Sentiment Analysis Under Missing and Corrupted Modalities,"* *Electronics*, Vol. 15, Issue 16, Article 3624, August 2026.  
* **DOI:** `10.3390/electronics15163624`
* **Method:** Formulates an evidential deep neural network that models uncertainty via Dirichlet distributions over class probabilities, dynamically discounting corrupted modalities during fusion.
* **Action Space & Assumptions:** Monolithic, centralized model. Assumes all audio, visual, and textual modalities are pre-transmitted, co-located in memory, and available simultaneously.
* **Systems Scope:** Does NOT model edge-cloud networks, does not measure socket wire bytes, and does not enforce deadlines or budget admission.
* **Differentiation:** Tri-SACTA operates on the orthogonal **communication runtime layer**: deciding *whether* and *when* to transmit remote modalities over a network socket to minimize wire bytes under hard latency deadlines, rather than optimizing centralized fusion weights.

### 1.2 Quality-Aware Multimodal Stabilization: RegCal-Net (2026)
* **Citation:** Ao Ding, Bin Xu, Weiyang W, et al., *"Register-augmented attention and self-calibrated fusion for robust multimodal sentiment analysis,"* *Multimedia Systems*, 2026.  
* **DOI:** `10.1007/s00530-026-02653-8`
* **Method:** Introduces register tokens to stabilize cross-modal attention maps and self-calibrates feature fusion on CMU-MOSI and CMU-MOSEI.
* **Action Space & Scope:** Pure architectural deep learning contribution evaluated on static offline GPU clusters. Does not measure distributed communication, network RTT, or feature transmission costs.
* **Differentiation:** Tri-SACTA does not propose a new deep transformer architecture; rather, it designs a lightweight sequential edge-cloud acquisition runtime that achieves >60% network wire byte savings over real TCP sockets.

### 1.3 Confidence Routing & Early Exit Frameworks
* **Literature:** E.g., DeeBERT (Xin et al., ACL 2020), Shallow-Deep Networks (Kaya et al., ICML 2019), and dynamic early-exit multimodal routing.
* **Scope & Limitation in Literature:** Prior confidence routing works focus almost exclusively on reducing internal GPU floating-point operations (FLOPs) inside a centralized neural network. They do not implement physical network sockets, do not account for application-layer framing overhead, and do not model independent remote provider failure or monotonic deadline reserves.
* **Tri-SACTA Rectification:** Tri-SACTA implements a production-grade, fail-closed runtime over real TCP stream sockets, measuring physical wire byte reductions, socket RTT, monotonic deadline enforcement, and provider timeout handling.

---

## 2. Claim-Evidence-Counterevidence Matrix

| Claim ID | Proposition | Supporting Evidence | Potential Counterevidence & Scope Limitations | Verdict |
| :---: | :--- | :--- | :--- | :---: |
| **CLM-01** | **Threshold Equivalence:** V13's static expected utility is algebraically and empirically identical to sequential confidence routing. | Formal algebraic proof; 100% agreement (0/10,001 mismatches) between `DerivedThresholdRouter` and `UtilityRouter`. | Holds for static priors $\gamma, \nu, \Delta C$. State-conditioned gain models ($z$-dependent) break algebraic equivalence, but fail to achieve higher F1 empirically. | **VERIFIED** |
| **CLM-02** | **Algorithmic Simplicity:** Complex state-conditioned utility estimation (`ConditionalUtilityRouter`) does not provide a statistically significant advantage over simpler `SequentialConfidenceRouter`. | On 84,000 developmental queries, `SequentialConfidence_0.55_0.45` achieved Macro-F1 **0.5415 $\pm$ 0.0569** vs `ConditionalUtility_0.05` **0.5310 $\pm$ 0.0552**; $\Delta = +0.0104$ `[-0.0044, +0.0244]`. | Meta-gain estimation adds empirical estimation variance on small samples without improving generalization. | **VERIFIED** |
| **CLM-03** | **Resource Reduction:** `SequentialConfidence_0.55_0.45` achieves $\ge 15\%$ cost reduction with predictive non-inferiority (margin $-0.010$) vs Always-On and Static TA. | Paired cluster bootstrap ($B=2,000$): Cost saved vs Always-On = **+50.1%** `[+48.6%, +51.8%]`, $\Delta \text{F1} = +0.0127$ `[-0.0037, +0.0313]`. Cost saved vs Static TA = **+18.8%** `[+16.3%, +21.5%]`, $\Delta \text{F1} = +0.0052$ `[-0.0047, +0.0158]`. | Comparisons conditional on the 7 developmental workloads; confirmatory holdout remains frozen for Phase II. | **VERIFIED ON DEV** |
| **CLM-04** | **Physical Network Savings:** Over real TCP stream sockets, sequential acquisition reduces application-protocol bytes by $>60\%$ compared to full transmission. | Measured on localhost TCP sockets across 250 validation queries: **63.48% wire bytes reduction** (66,308 B vs 181,560 B) and **62.88% lower latency** (4.79 ms vs 12.89 ms). | Measurements reflect feature tensor transmission, NOT raw video/audio media ingestion. Link-layer IP/TCP headers add ~40-60 B/packet. | **VERIFIED ON DEV** |
| **CLM-05** | **Novelty Positioning:** Tri-SACTA provides an honest systems orchestration contribution for resource-efficient multimedia classification. | Verified differentiation against Wu et al. (evidential), RegCal-Net (centralized fusion), and DeeBERT (FLOP early exit). | Novelty lies in systems integration, deadline admission, and physical edge-cloud socket telemetry, NOT in a new neural layer. | **ESTABLISHED (SYSTEMS SCOPE)** |

---

## 3. Scope Boundaries & Explicit Non-Claims

1. **NO Claim of Distribution-Free Conformal Guarantees:** Conformal coverage strictly holds on clean, exchangeable validation data (~90.5%). Under severe distribution shift (e.g., compound faults), coverage degrades to ~75.9%.
2. **NO Claim of End-to-End Raw Media Ingestion:** Input features are pre-extracted representations (DeBERTa 768-D, COVAREP 74-D, FACET 35-D) from the standard benchmark distribution (`mosei.pkl`).
3. **NO Claim of Human Speaker Disjointness:** CMU-MOSEI does not provide verified human speaker biometrics. All statistical independence claims are strictly bounded to YouTube video cluster disjointness.
