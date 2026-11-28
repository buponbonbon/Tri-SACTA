# Tri-SACTA V13.1 — Fair Baseline Protocol & Matched Budget Benchmark

**Document Version:** 1.0.0  
**Timestamp (UTC):** 2026-11-24T15:45:00Z  
**Branch:** `research/v13.1-conditional-utility-independent-validation`  
**Artifact References:**
- `results/v13_1/evaluation/policy_metrics.json`
- `results/v13_1/evaluation/matched_budget_frontiers.csv`
- `results/v13_1/evaluation/query_traces.json`

---

## 1. Protocol Motivation & Adversarial Corrections

Tri-SACTA V13 compared its `UtilityRouter` primarily against fixed One-Shot Confidence Routing (`CR0.55` and `CR0.65`) and static `AlwaysOnFusion`. An adversarial audit revealed that:
1. One-shot confidence routing does not re-evaluate model predictions after acquiring a second modality, whereas sequential utility does. This was an unaligned comparison.
2. V13 expected utility was algebraically identical to sequential confidence thresholding, yet was presented as a distinct mathematical mechanism.
3. To evaluate whether state-dependent conditioning provides genuine scientific value, the candidate must be compared against a **Sequential Confidence Router** with the **identical action space, identical predictor bank, identical feasibility controller, and matched budgets**.

---

## 2. Policy Families & Algorithmic Definitions

All policies inherit from `BasePredictorPolicy`, execute on identical PyTorch tensors, and use the exact same seven task-matched subset MLPs ($T, A, V, TA, TV, AV, TAV$) across seeds 42, 100, and 2026.

### 2.1 Candidate M2: Conditional Utility Router (`ConditionalUtilityRouter`)
- **State Representation:** $z = [p_{\max}, \text{margin}, \text{entropy}, p_{\text{neg}}, p_{\text{neu}}, p_{\text{pos}}, \text{cost}_{\text{rem\_ratio}}, \text{lat}_{\text{rem\_ratio}}]$
- **Gain Estimation:** $\hat{G}(z, a) = \hat{\mathbb{E}}[E_S - E_{S \cup \{a\}} \mid z]$ fitted via group-cross-validation on the 1,000-sample training split.
- **Utility Rule:** $U(a \mid z) = \frac{\hat{G}(z, a)}{\Delta C(a)}$. Acquire $a^* = \arg\max_{a \in \mathcal{A}_{\text{feasible}}} U(a \mid z)$ iff $U(a^* \mid z) \ge \lambda$.
- **Early Stopping:** If $p_{\max} \ge \tau_{\text{stop}}$ (0.75), stop immediately.

### 2.2 Fair Sequential Baseline: Sequential Confidence Router (`SequentialConfidenceRouter`)
- **Action Space:** Identical sequential state machine ($T \to TA \to TAV$ or $T \to TV \to TAV$).
- **Stage 1 Decision:** Evaluate $p_T$. If $p_T < \tau_1$, acquire secondary modality (Audio preferred, Visual fallback).
- **Stage 2 Decision:** Re-predict with $TA$ (or $TV$). If $p_{S'} < \tau_2$, acquire tertiary modality ($TAV$).
- **Constraints:** Enforces identical cumulative budget ceilings and monotonic deadlines.

### 2.3 Derived Equivalent Baselines (`DerivedThresholdRouter`)
- **M0 (Historical V13):** Executes derived threshold $\tau(S, m) = \frac{\gamma - \lambda \Delta C}{\gamma + \nu}$ using frozen V13 constants. Proves 100% equivalence to V13 `UtilityRouter`.
- **M1 (Corrected Priors):** Executes derived threshold using empirical validation transition priors.

### 2.4 One-Shot Confidence Baselines (`ConfidenceRoutingPolicy`)
- Evaluates initial modality. If $p < \tau$, acquires all remaining available modalities in one shot.

### 2.5 Static & Fallback Baselines
- `Static_T`: Always evaluates unimodal Text (cost 1.0).
- `Static_TA`: Always acquires Text + Audio (cost 2.2), fallback to Text under missingness.
- `Static_TV`: Always acquires Text + Visual (cost 2.3), fallback to Text under missingness.
- `Static_TAV` / `AlwaysOnFusion`: Always acquires all three modalities (cost 3.5), fallback under missingness.
- `FixedFallback`: Lowest-cost fallback policy.

### 2.6 Matched Budget Random Baseline (`MatchedBudgetRandomPolicy`)
- Acquires unacquired modalities with Bernoulli probability $p_{\text{acquire}}$ calibrated to match target nominal cost.

---

## 3. Benchmark Workload Suite (7 Stress Workloads)

1. `CLEAN_MULTIMODAL`: Standard unaltered multimodal feature vectors.
2. `TEXT_EMBEDDING_NOISE`: Emulates ASR transcription errors via Gaussian perturbation ($\sigma = 0.40$).
3. `AUDIO_FEATURE_MASKING`: Complete acoustic channel failure / mute (Audio masked to zero, mask indicated).
4. `VISUAL_ZERO_MASK`: Complete camera occlusion / packet loss (Visual masked to zero, mask indicated).
5. `COMPOUND_FAULT`: Simultaneous text noise ($\sigma = 0.50$) and acoustic channel loss.
6. `INTERMITTENT_MISSINGNESS`: Bernoulli channel dropout (Audio 30%, Visual 40%, Text 10%).
7. `HETEROGENEOUS_DEADLINE_STRESS`: Tight latency budgets (15 ms unimodal, 30 ms partial, 50 ms relaxed).

---

## 4. Evaluation Metrics & Preregistered Criteria

For each policy, metrics are aggregated across 7 workloads $\times$ 3 seeds ($N = 21$ runs per split, 5,250 evaluated utterances):
- **Macro-F1:** Exact macro-averaged F1 across 3 fixed classes (`[0, 1, 2]`).
- **Accuracy:** Overall classification accuracy.
- **Mean Spent Cost:** Average nominal resource cost charged per query.
- **Mean Latency:** Average end-to-end processing latency (ms).
- **On-Time Rate:** Percentage of requests completed strictly before the query deadline.
- **Primary Hypothesis Criterion (H1/H2):** At matched acquisition cost, Candidate M2 must achieve non-inferior predictive performance ($\Delta \text{Macro-F1} \ge -0.01$) with lower bound of 95% paired clustered CI strictly above zero for superiority or above $-0.01$ for non-inferiority.
