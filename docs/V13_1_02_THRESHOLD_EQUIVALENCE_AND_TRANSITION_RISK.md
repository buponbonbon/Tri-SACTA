# Tri-SACTA V13.1 — Mathematical Equivalence & Empirical Transition Risk

**Document Version:** 1.0.0  
**Timestamp (UTC):** 2026-11-24T15:31:00Z  
**Branch:** `research/v13.1-conditional-utility-independent-validation`  
**Artifact References:**
- `results/v13_1/priors/empirical_transition_priors.json`
- `results/v13_1/priors/derived_thresholds.json`

---

## 1. Formal Proof of Threshold Equivalence

### 1.1 Expected Utility Definition in Tri-SACTA V13
Let $S$ denote the set of currently acquired modalities, and $m \notin S$ denote an unacquired candidate modality. Let $p = \max_{c \in \{0, 1, 2\}} P(Y = c \mid x_S)$ denote the maximum softmax confidence of the task-matched subset predictor for $S$. The plug-in risk estimate is:
$$R(S) = 1 - p$$

Let $\gamma(S, m)$ denote the rescue prior (probability that $S \cup \{m\}$ is correct given $S$ is incorrect) and $\nu(S, m)$ denote the negative transfer prior (probability that $S \cup \{m\}$ is incorrect given $S$ is correct). The expected risk reduction is:
$$\Delta \hat{R}(m \mid S) = R(S) \gamma(S, m) - (1 - R(S)) \nu(S, m)$$

Substituting $R(S) = 1 - p$:
$$\Delta \hat{R}(m \mid S) = (1 - p) \gamma(S, m) - p \nu(S, m) = \gamma(S, m) - (\gamma(S, m) + \nu(S, m)) p$$

The marginal expected utility per incremental cost $\Delta C(m)$ is:
$$U(m \mid S) = \frac{\Delta \hat{R}(m \mid S)}{\Delta C(m)} = \frac{\gamma(S, m) - (\gamma(S, m) + \nu(S, m)) p}{\Delta C(m)}$$

### 1.2 Derivation of the Equivalent Confidence Threshold
The acquisition policy triggers modality acquisition if and only if $U(m \mid S) \ge \lambda$, where $\lambda > 0$ is the utility threshold parameter:
$$\frac{\gamma(S, m) - (\gamma(S, m) + \nu(S, m)) p}{\Delta C(m)} \ge \lambda$$
$$\gamma(S, m) - (\gamma(S, m) + \nu(S, m)) p \ge \lambda \Delta C(m)$$
$$(\gamma(S, m) + \nu(S, m)) p \le \gamma(S, m) - \lambda \Delta C(m)$$

Assuming $\gamma(S, m) + \nu(S, m) > 0$ (which strictly holds for all real predictive models):
$$p \le \frac{\gamma(S, m) - \lambda \Delta C(m)}{\gamma(S, m) + \nu(S, m)} \equiv \tau(S, m, \lambda)$$

### 1.3 Inclusion of the Early-Stopping Criterion
The V13 policy also enforced an early-stopping rule: stop acquisition if $p \ge \tau_{\text{stop}}$ (where $\tau_{\text{stop}} = 0.75$).
Thus, acquisition occurs if and only if:
$$p \le \min\left( \frac{\gamma(S, m) - \lambda \Delta C(m)}{\gamma(S, m) + \nu(S, m)}, \; \tau_{\text{stop}} \right)$$

**Conclusion:** The expected utility rule with static priors is **algebraically and functionally identical** to a sequential confidence policy with threshold $\tau(S, m, \lambda)$. Any claim that V13 expected utility represents a distinct optimization mechanism from confidence routing is mathematically refuted.

---

## 2. Table of Derived Thresholds

Evaluating $\tau(S, m, \lambda)$ at $\lambda = 0.05$ and $\tau_{\text{stop}} = 0.75$:

| Transition ($S \to S \cup \{m\}$) | Cost $\Delta C$ | V13 $\gamma$ | V13 $\nu$ | V13 Derived $\tau$ | Empirical $\gamma$ (Val) | Empirical $\nu$ (Val) | Corrected $\tau$ |
|---|---|---|---|---|---|---|---|
| $T \to TA$ (+Audio) | 1.2 | 0.3768 | 0.1160 | **0.6429** | 0.1693 | 0.1167 | **0.3822** |
| $T \to TV$ (+Visual) | 1.3 | 0.3387 | 0.2245 | **0.4860** | 0.2045 | 0.2059 | **0.3398** |
| $TA \to TAV$ (+Visual) | 1.3 | 0.2437 | 0.1700 | **0.4320** | 0.2315 | 0.1754 | **0.4092** |
| $TV \to TAV$ (+Audio) | 1.2 | 0.2891 | 0.1850 | **0.4832** | 0.2183 | 0.1241 | **0.4623** |
| $A \to TA$ (+Text) | 1.0 | 0.5495 | 0.1200 | **0.7461** | 0.5207 | 0.3259 | **0.5560** |
| $A \to AV$ (+Visual) | 1.3 | 0.3781 | 0.2500 | **0.4985** | 0.2558 | 0.3892 | **0.2958** |
| $V \to TV$ (+Text) | 1.0 | 0.5473 | 0.1200 | **0.7452** | 0.4683 | 0.3276 | **0.5256** |
| $V \to AV$ (+Audio) | 1.2 | 0.3461 | 0.2600 | **0.4720** | 0.2516 | 0.3549 | **0.3160** |
| $AV \to TAV$ (+Text) | 1.0 | 0.5467 | 0.1100 | **0.7564** ($\to 0.75$) | 0.4753 | 0.2697 | **0.5709** |

---

## 3. Proof of Visual Modality Unreachability at State $T$ in V13

Let $p \in [0, 1]$. Using V13 parameters:
$$U(A \mid T) = \frac{0.3768 - 0.4928 p}{1.2} = 0.31400 - 0.41067 p$$
$$U(V \mid T) = \frac{0.3387 - 0.5632 p}{1.3} = 0.26054 - 0.43323 p$$

The difference function $D(p) = U(A \mid T) - U(V \mid T)$ is:
$$D(p) = (0.31400 - 0.26054) - (0.41067 - 0.43323) p = 0.05346 + 0.02256 p$$

Evaluating at the boundaries:
- At $p = 0.0$: $D(0.0) = +0.05346 > 0$
- At $p = 1.0$: $D(1.0) = +0.07602 > 0$
- Derivative: $D'(p) = +0.02256 > 0$ (monotonically increasing)

**Theorem:** For all $p \in [0, 1]$, $U(A \mid T) > U(V \mid T)$ strictly holds.  
**Corollary:** Under the V13 expected utility policy, when both Audio and Visual are feasible and unmasked, the Visual modality is **never** acquired from the initial state $T$. The policy's effective action tree is strictly constrained to $T \to TA \to TAV$.

---

## 4. Empirical Transition Frequency Breakdown (Validation Set, 3 Seeds)

Measured across all 250 validation utterances $\times$ 3 training seeds ($N = 750$ evaluation instances):

| Transition | Total Queries | Correct $\to$ Correct | Correct $\to$ Harm | Wrong $\to$ Rescue | Wrong $\to$ Wrong | Gamma (95% CI) | Nu (95% CI) | Net Gain |
|---|---|---|---|---|---|---|---|---|
| $T \to TA$ | 750 | 386 | 51 | 53 | 260 | 0.1693 [0.1314, 0.2154] | 0.1167 [0.0897, 0.1504] | **+2** |
| $T \to TV$ | 750 | 347 | 90 | 64 | 249 | 0.2045 [0.1630, 0.2534] | 0.2059 [0.1704, 0.2467] | **-26** |
| $TA \to TAV$ | 750 | 362 | 77 | 72 | 239 | 0.2315 [0.1880, 0.2818] | 0.1754 [0.1423, 0.2142] | **-5** |
| $TV \to TAV$ | 750 | 360 | 51 | 74 | 265 | 0.2183 [0.1774, 0.2657] | 0.1241 [0.0955, 0.1597] | **+23** |
| $A \to TA$ | 750 | 213 | 103 | 226 | 208 | 0.5207 [0.4735, 0.5676] | 0.3259 [0.2764, 0.3797] | **+123** |
| $V \to TV$ | 750 | 197 | 96 | 214 | 243 | 0.4683 [0.4227, 0.5146] | 0.3276 [0.2764, 0.3831] | **+118** |

### Key Observations:
1. **Adding Visual to Text Alone Harms Performance:** $T \to TV$ causes 90 correct predictions to flip to wrong, while rescuing only 64 errors (Net Gain = -26).
2. **Adding Visual to Text+Audio is Net Neutral/Negative:** $TA \to TAV$ rescues 72 but harms 77 (Net Gain = -5).
3. **Text is the Indispensable Anchor:** Transitioning from audio/visual to text ($A \to TA$, $V \to TV$) produces massive net gains (+123, +118).
4. These findings demonstrate that static global priors cannot reliably decide whether to acquire additional modalities. To provide genuine predictive value, acquisition decisions must be **conditioned on input-specific uncertainty and feature attributes** (Candidate M2).
