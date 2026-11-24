# Tri-SACTA V13.1 — Forensic Audit & Scientific Rectification of V13

**Audit Date:** 2026-11-24  
**Audited Branch:** `research/v13-utility-driven-orchestration`  
**Audited Commit:** `a95c375bb04df2ac7fa67d54ecc8f58c419de136`  
**Parent V12.1 Commit:** `44bc28c31eb38cca3de841b61320a1ea057d5135`  
**Working Branch:** `research/v13.1-conditional-utility-independent-validation`  
**Status Superseded:** `PROCEED_SYSTEMS_PAPER` (from V13) is officially downgraded to `PROMISING_EXPLORATORY_EVIDENCE — INDEPENDENT_VALIDATION_REQUIRED`.

---

## 1. Context and Audit Scope

The Tri-SACTA V13 iteration introduced an expected-utility formulation for sequential modality acquisition, deployed task-matched subset MLPs (7 subsets $\times$ 3 seeds), and implemented video-clustered paired bootstrap testing. V13 concluded with a positive verdict (`PROCEED_SYSTEMS_PAPER`), claiming statistically significant resource savings and Pareto dominance.

An adversarial independent audit of the committed V13 artifacts revealed three foundational scientific vulnerabilities:
1. **Mathematical Equivalence Defect:** The expected utility acquisition rule is algebraically equivalent to sequential confidence thresholding.
2. **Transition Target Mismatch:** The hardcoded rescue rate $\gamma(T, \text{audio}) = 0.3768$ was derived from unimodal Audio correcting Text, rather than the deployed multimodal model $TA$ correcting Text ($\gamma \approx 0.1693$).
3. **Mislabeled Confirmatory Split:** The 500-sample test set had been exposed to development across V7–V9 and quarantined in V12.1. V13 reused this set and mislabeled it as an "independent confirmatory evaluation."

This document records the exact findings, derivations, and corrections required for Tri-SACTA V13.1.

---

## 2. Forensic Findings

### 2.1 The Mathematical Equivalence Defect
The V13 decision rule specifies:
$$\Delta \hat{R}(m \mid S) = R(S) \gamma(m \mid S) - (1 - R(S)) \nu(m \mid S)$$
$$U(m \mid S) = \frac{\Delta \hat{R}(m \mid S)}{\Delta C(m)}$$
$$\text{Acquire } m^* = \arg\max_{m \in \mathcal{A}_{\text{feasible}}} U(m \mid S) \quad \text{iff } U(m^* \mid S) \ge \lambda$$

Where $R(S) = 1 - p$ and $p = \max_c P(Y=c \mid S)$.

Because $\gamma$, $\nu$, and $\Delta C(m)$ are static constants for any given state transition $S \to S \cup \{m\}$, $U(m \mid S)$ is a strictly affine function of $p$:
$$U(m \mid S) = \frac{\gamma - (\gamma + \nu)p}{\Delta C(m)}$$
For $\gamma + \nu > 0$, the inequality $U(m \mid S) \ge \lambda$ is algebraically identical to:
$$p \le \frac{\gamma - \lambda \Delta C(m)}{\gamma + \nu} \equiv \tau(S, m, \lambda)$$

Across 10,001 confidence points evaluated in `scripts/v13_1_fit_transition_estimators.py`, there is **zero decision divergence** between the expected utility rule and the derived confidence threshold. Therefore, V13 did not introduce a novel algorithmic mechanism; it simply reparameterized sequential confidence thresholding.

### 2.2 The Crossover and Unreachability Defect
Under V13's hardcoded priors at state $T$:
- Audio: $\gamma = 0.3768, \nu = 0.1160, \Delta C = 1.2 \implies U(A \mid T) = 0.3140 - 0.4107 p$
- Visual: $\gamma = 0.3387, \nu = 0.2245, \Delta C = 1.3 \implies U(V \mid T) = 0.2605 - 0.4332 p$

For all $p \in [0, 1]$:
$$U(A \mid T) - U(V \mid T) \in [+0.0535, +0.0760] > 0$$
Visual utility is strictly inferior to Audio utility across the entire probability simplex. As a result, **the Visual modality could never be acquired at state $T$** unless Audio was artificially masked out.

### 2.3 The Transition Rate Mismatch
In V13, $\gamma(T, \text{audio}) = 0.3768$ was taken from the unimodal head-to-head comparison $P(E_A=0 \mid E_T=1)$. However, when the router acquires Audio, it executes the multimodal predictor $TA$. The empirical transition rates on the validation set across all three seeds are:
- $T \to TA$: $\gamma = 0.1693$ (53 rescues / 313 errors), $\nu = 0.1167$ (51 harms / 437 correct), Net Gain = $+2$
- $T \to TV$: $\gamma = 0.2045$ (64 rescues / 313 errors), $\nu = 0.2059$ (90 harms / 437 correct), Net Gain = $-26$
- $TA \to TAV$: $\gamma = 0.2315$ (72 rescues / 311 errors), $\nu = 0.1754$ (77 harms / 439 correct), Net Gain = $-5$

V13 overestimated the rescue capability of Audio by over $122\%$ ($0.3768$ vs $0.1693$).

---

## 3. Scientific Rectification Action Plan for V13.1

1. **Acknowledge Equivalence:** Implement `DerivedThresholdRouter` as an explicit baseline to demonstrate structural equivalence to V13.
2. **Develop Conditional Transition Estimator (Candidate M2):** Condition gain estimation on pre-acquisition features $z = (p_S, M(p_S), H(p_S), \text{budget}_{\text{rem}}, \text{deadline}_{\text{rem}})$. If state conditioning does not outperform simple confidence routing, the project must honestly simplify to the strong sequential confidence baseline.
3. **Enforce Matched Action Spaces:** Compare against a Sequential Confidence Router with the identical action space, predictor bank, and budget constraints.
4. **Quarantine Historical Test Split:** Label the 500-sample test set as an exposed historical baseline. Prospective confirmatory evaluation must use the unexposed 4,131-utterance CMU-MOSEI pool.
5. **Real Distributed Measurement (Stage G):** Measure actual socket serialization and network latency, eliminating all mock delays.
