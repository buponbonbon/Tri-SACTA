# Tri-SACTA V3 — Mathematical Foundations & Online Conformal Guarantees

**Document Version:** 3.0 (Post-Re-Audit Scientific Hardening)  
**Authors:** Tri-SACTA Research Consortium  
**Target Venue:** Q1 Journal (Information Fusion / IEEE TMM / Neural Networks)  

---

## 1. Context & Retraction of Prior Inaccurate Theorems

> [!WARNING] Retraction Notice (V2 $\to$ V3 Transition)
> In Tri-SACTA V2, `Theorem 2` claimed that any bounded sequence of adaptive learning rates $\eta_t \in [\eta_{\text{min}}, \eta_{\text{max}}]$ guarantees asymptotic convergence of average miscoverage to the nominal target $\alpha$, i.e., $\lim_{T \to \infty} \frac{1}{T} \sum_{t=1}^T e_t = \alpha$.
>
> As proven by counterexample (see Section 3.2 and `tests/test_aci_counterexample.py`), this claim is **mathematically false** for arbitrary variable step sizes without bounded total variation. Furthermore, `Theorem 3` (feedback delay bound based purely on feature spectral velocity $L_{\text{shift}}$) is formally retracted, as feature drift alone cannot control independent concept/label shifts.
>
> In this V3 specification, we establish:
> 1. The exact **Clipping Residual & Summation-by-Parts Identity** governing general variable-step online ACI.
> 2. The **Fixed-Step Mask-Conditional ACI Reference Controller**, with an exact, closed-form pathwise finite-sample tracking bound.
> 3. The precise status of the **Mask-and-Disagreement Dynamic Step Size** as a validated empirical heuristic.

---

## 2. Problem Setup & Non-Exchangeable Streaming Drift

Let $\mathcal{X} = \mathcal{X}^{(1)} \times \dots \times \mathcal{X}^{(M)}$ denote the multimodal feature space across $M$ sensor streams, and $\mathcal{Y} = \{0, 1, \dots, C-1\}$ be the discrete label space. At streaming step $t \in \mathbb{N}$:
1. An incoming test sample $\mathbf{x}_t \in \mathcal{X}$ is observed alongside a physical sensor availability mask $\mathbf{m}_t \in \{0, 1\}^M \setminus \{\mathbf{0}\}$.
2. The multimodal backbone with adaptive parameters $\theta_t$ produces class conditional probability estimates $\hat{p}_t(y \mid \mathbf{x}_t, \mathbf{m}_t)$.
3. An online conformal controller constructs a prediction set $\mathcal{C}_t(\mathbf{x}_t, \mathbf{m}_t) \subseteq \mathcal{Y}$.
4. True label $y_t \in \mathcal{Y}$ is revealed with delay $\delta \ge 0$, defining the empirical miscoverage indicator:
   $$e_t = \mathbb{I}\big\{y_t \notin \mathcal{C}_t(\mathbf{x}_t, \mathbf{m}_t)\big\} \in \{0, 1\}$$

Under compound modality shifts (sensor degradation, acoustic SNR drops, ASR drift, missing modalities), the joint distribution shifts continuously: $P_t(X, Y) \ne P_s(X, Y)$. The sequence of non-conformity scores is non-exchangeable, rendering static split conformal prediction invalid.

---

## 3. Mathematical Foundations of Online Conformal Control

### 3.1. General Update Rule and Clipping Residuals
Let $\alpha \in (0, 1)$ be the nominal target miscoverage (e.g., $\alpha = 0.10$ for $90\%$ coverage).
Let $\alpha_t$ be the controller's nominal parameter at step $t$. When empirical error $e_t \in \{0, 1\}$ is observed, the unprojected update is:
$$\tilde{\alpha}_{t+1} = \alpha_t + \eta_t (\alpha - e_t)$$
If projected/clipped to $[\alpha_{\text{min}}, \alpha_{\text{max}}] \subseteq [0, 1]$:
$$\alpha_{t+1} = \Pi_{[\alpha_{\text{min}}, \alpha_{\text{max}}]}(\tilde{\alpha}_{t+1}) = \tilde{\alpha}_{t+1} + c_t$$
where $c_t = \alpha_{t+1} - \tilde{\alpha}_{t+1}$ is the **clipping residual**:
$$c_t = \begin{cases}
\alpha_{\text{min}} - \tilde{\alpha}_{t+1} & \text{if } \tilde{\alpha}_{t+1} < \alpha_{\text{min}} \quad (c_t > 0) \\
\alpha_{\text{max}} - \tilde{\alpha}_{t+1} & \text{if } \tilde{\alpha}_{t+1} > \alpha_{\text{max}} \quad (c_t < 0) \\
0 & \text{otherwise.}
\end{cases}$$

Rearranging gives the fundamental algebraic identity:
$$\alpha - e_t = \frac{\alpha_{t+1} - \alpha_t}{\eta_t} - \frac{c_t}{\eta_t}$$

Summing over $t = 1, \dots, T$:
$$\sum_{t=1}^T (\alpha - e_t) = \sum_{t=1}^T \frac{\alpha_{t+1} - \alpha_t}{\eta_t} - \sum_{t=1}^T \frac{c_t}{\eta_t}$$

Let $b_t = \frac{1}{\eta_t}$. Applying Abel summation by parts:
$$\sum_{t=1}^T b_t (\alpha_{t+1} - \alpha_t) = b_T \alpha_{T+1} - b_1 \alpha_1 + \sum_{t=2}^T (b_{t-1} - b_t) \alpha_t$$

Assuming $\alpha_t \in [0, A]$ is uniformly bounded:
$$\left|\sum_{t=1}^T (\alpha - e_t)\right| \le \frac{2A}{\eta_{\text{min}}} + A \sum_{t=2}^T \left|\frac{1}{\eta_t} - \frac{1}{\eta_{t-1}}\right| + \sum_{t=1}^T \frac{|c_t|}{\eta_t}$$

Dividing by $T$ yields the **General Diagnostic Pathwise Bound**:
$$\left|\bar{e}_T - \alpha\right| \le \frac{2A}{T \cdot \eta_{\text{min}}} + \frac{A}{T} \sum_{t=2}^T \left|\frac{1}{\eta_t} - \frac{1}{\eta_{t-1}}\right| + \frac{1}{T} \sum_{t=1}^T \frac{|c_t|}{\eta_t}$$

> [!IMPORTANT] Critical Theoretical Takeaway: Sufficient Conditions
> The empirical tracking error $|\bar{e}_T - \alpha|$ is guaranteed to vanish as $T \to \infty$ under the following **sufficient conditions**:
> 1. Total variation of the inverse step sizes is sublinear: $\sum_{t=2}^T |1/\eta_t - 1/\eta_{t-1}| = o(T)$.
> 2. Cumulative clipping residuals are sublinear: $\sum_{t=1}^T |c_t|/\eta_t = o(T)$.
>
> *Note on Mathematical Scope:* Because Eq. (64) is derived via the triangle inequality, these conditions are strictly **sufficient** rather than necessary; non-vanishing variation could theoretically admit path-dependent cancellations, but no universal convergence guarantee can hold without controlling variation and clipping. If step sizes oscillate without sublinear variation (as demonstrated in Section 3.2), unweighted miscoverage fails to converge to $\alpha$.

---

### 3.2. Arithmetic Counterexample to V2 Theorem 2
Consider the following discrete sequence:
- Target: $\alpha = 0.10$. Initial state: $\alpha_1 = 0.50$.
- Step $t$ odd: $e_t = 1$, $\eta_t = 0.01 \implies \Delta \alpha_t = 0.01(0.10 - 1.0) = -0.009$.
- Step $t$ even: $e_t = 0$, $\eta_t = 0.09 \implies \Delta \alpha_t = 0.09(0.10 - 0.0) = +0.009$.

Notice:
1. $\eta_t \in [0.01, 0.09]$ is strictly bounded ($0 < \eta_{\text{min}} \le \eta_t \le \eta_{\text{max}} < 1$).
2. $\alpha_t$ alternates strictly between $0.500$ and $0.491$, remaining strictly within $(0, 1)$ with **zero clipping** ($c_t \equiv 0$).
3. The empirical average miscoverage is:
   $$\bar{e}_T = \frac{1}{T} \sum_{t=1}^T e_t = 0.50 \ne \alpha = 0.10$$
   The tracking gap is $|0.50 - 0.10| = 0.40$ for all even $T$, permanently violating asymptotic validity.

This arithmetic counterexample (implemented and verified in `tests/test_aci_counterexample.py`) rigorously refutes Theorem 2 of V2.

---

## 4. The Fixed-Step Mask-Conditional ACI Reference Controller

To provide an ironclad, mathematically provable reference method, we specify the **Fixed-Step Mask-Conditional ACI Controller**.

### 4.1. Specification
For each observed modality mask group $g = \mathbf{m} \in \{0, 1\}^M \setminus \{\mathbf{0}\}$:
1. Maintain local update count $n_g$ and unprojected controller state $\alpha_{t, g}$.
2. Step size is constant per group: $\eta_g = \eta > 0$.
3. **Boundary Set Emission Policy:**
   $$\mathcal{C}_t(x_t, g) = \begin{cases}
   \mathcal{Y} = \{0, 1, \dots, C-1\} & \text{if } \alpha_{t, g} \le 0 \quad (\implies e_t = 0) \\
   \emptyset & \text{if } \alpha_{t, g} \ge 1 \quad (\implies e_t = 1) \\
   \{y : s(x_t, y) \le \hat{q}_{t, g}\} & \text{if } 0 < \alpha_{t, g} < 1
   \end{cases}$$
   where $\hat{q}_{t, g}$ is the empirical $\lceil (n_g + 1)(1 - \alpha_{t, g})\rceil$-th order statistic from the local non-conformity calibration window.
   Crucially, **no forced top-1 prediction** is applied when $\alpha_{t, g} \ge 1$.

### 4.2. Pathwise Telescoping Theorem under Immediate Feedback

**Theorem 1 (Exact Finite-Sample Pathwise Validity under Immediate Feedback):**
For any arbitrary, non-exchangeable, adversarial streaming sequence of inputs and labels, under the fixed-step mask controller with boundary emission policy and immediate label feedback ($\Delta_{\text{delay}} = 0$):
After $n_g \ge 1$ updates on mask group $g$:
$$\left| \frac{1}{n_g} \sum_{i=1}^{n_g} e_{i, g} - \alpha \right| \le \frac{1 + \eta}{\eta \cdot n_g}$$

*Proof:*
Since the controller state $\alpha_{i, g}$ is unprojected, the exact update rule is:
$$\alpha_{i+1, g} = \alpha_{i, g} + \eta (\alpha - e_{i, g})$$
Summing directly over all $i = 1, \dots, n_g$:
$$\sum_{i=1}^{n_g} (\alpha - e_{i, g}) = \frac{\alpha_{n_g+1, g} - \alpha_{1, g}}{\eta}$$

We now bound the range of the unprojected state $\alpha_{n_g+1, g}$:
- Suppose $\alpha_{i, g} \le 0$. By the boundary policy, $\mathcal{C}_i = \mathcal{Y}$, so $e_{i, g} = 0$. The update is $\alpha_{i+1, g} = \alpha_{i, g} + \eta \alpha > \alpha_{i, g}$. Since $e_{i, g} \in \{0, 1\}$ and $\alpha \in (0, 1)$, $\alpha_{i+1, g}$ can overshoot below $0$ by at most $\eta (1 - \alpha)$. Thus, $\alpha_{t, g} \ge -\eta(1 - \alpha) > -\eta$.
- Suppose $\alpha_{i, g} \ge 1$. By the boundary policy, $\mathcal{C}_i = \emptyset$, so $e_{i, g} = 1$. The update is $\alpha_{i+1, g} = \alpha_{i, g} + \eta (\alpha - 1) < \alpha_{i, g}$. The overshoot above $1$ is at most $\eta \alpha < \eta$. Thus, $\alpha_{t, g} \le 1 + \eta \alpha < 1 + \eta$.

Given $\alpha_{1, g} = \alpha \in [0, 1]$:
$$|\alpha_{n_g+1, g} - \alpha_{1, g}| \le \max(1 + \eta\alpha - 0, 1 - (-\eta(1-\alpha))) \le 1 + \eta$$

Dividing both sides by $n_g$:
$$\left| \alpha - \frac{1}{n_g} \sum_{i=1}^{n_g} e_{i, g} \right| \le \frac{|\alpha_{n_g+1, g} - \alpha_{1, g}|}{\eta \cdot n_g} \le \frac{1 + \eta}{\eta \cdot n_g}$$
$$\blacksquare$$

**Significance:** As $n_g \to \infty$, the miscoverage error on mask group $g$ converges to $\alpha$ at rate $\mathcal{O}(1 / (\eta \cdot n_g))$, holding for any arbitrary data stream without distribution assumptions.

### 4.3. Feedback Delay Analysis: Immediate Guarantee vs Delayed Implementation

The pathwise telescoping bound of Theorem 1 assumes **immediate label feedback** ($\Delta_{\text{delay}} = 0$), where each true label $y_t$ arrives and updates $\alpha_{t+1, g}$ before prediction set emission at step $t+1$.

In practical streaming deployments with **delayed feedback** ($\Delta_{\text{delay}} > 0$):
1. Let $T_g$ be the total number of events emitted for mask group $g$.
2. At time step $T$, let $n_g^{\text{obs}}$ be the number of observed/matured labels received and processed by the controller, and let $K_{\text{pending}, g} = T_g - n_g^{\text{obs}}$ be the number of pending labels currently in the FIFO delay buffer.
3. Because the maximum buffer capacity is bounded by the delay parameter $\Delta_{\text{delay}}$, we have:
   $$0 \le K_{\text{pending}, g} \le \min(\Delta_{\text{delay}}, T_g)$$
4. The cumulative miscoverage over all $T_g$ emitted events satisfies:
   $$\sum_{i=1}^{T_g} e_{i, g} = \sum_{j=1}^{n_g^{\text{obs}}} e_{j, g} + \sum_{k=1}^{K_{\text{pending}, g}} e_{k, g}$$
   Applying Theorem 1 to the $n_g^{\text{obs}}$ updated events and bounding pending errors by $K_{\text{pending}, g} \le \Delta_{\text{delay}}$:
   $$\left| \frac{1}{T_g} \sum_{i=1}^{T_g} e_{i, g} - \alpha \right| \le \frac{1 + \eta}{\eta \cdot T_g} + \frac{\Delta_{\text{delay}}}{T_g}$$

**Implication for Real-World Systems:**
For any finite, fixed feedback latency $\Delta_{\text{delay}} < \infty$, the asymptotic tracking error still vanishes as $\mathcal{O}((\eta^{-1} + \Delta_{\text{delay}})/T_g) \to 0$. However, during transient phases or short evaluation windows where $T_g \approx \Delta_{\text{delay}}$, pending feedback creates empirical coverage lag. The V4 benchmark protocol explicitly evaluates performance across $\Delta_{\text{delay}} \in \{0, 10, 50\}$ to quantify this transient lag.

---

## 5. Dynamic Step Size: Empirical Heuristic Formulation

While the fixed-step controller provides the theoretical anchor, rapid distribution shifts benefit empirically from adaptive step size responsiveness.

Tri-SACTA implements a state-dependent dynamic step size:
$$\eta_t = \text{clip}\left(\eta_0 \cdot \big(1 + \beta_1 |\Delta d_t| + \beta_2 \kappa_t\big), \; \eta_{\text{min}}, \; \eta_{\text{max}}\right)$$
where:
1. $|\Delta d_t| = \frac{1}{M} \sum_{m=1}^M |d_t^{(m)} - d_{t-1}^{(m)}|$ is the mean absolute spectral shift velocity across modalities.
2. $\kappa_t = \frac{1}{|\mathcal{P}_t|} \sum_{(m_1, m_2) \in \mathcal{P}_t} D_{\text{JS}}(p_t^{(m_1)} \,\|\, p_t^{(m_2)})$ is the pairwise cross-modal disagreement computed strictly across validly observed modality pairs $\mathcal{P}_t$.

> [!NOTE] Empirical Status in Paper
> In accordance with scientific integrity, we do NOT claim that this dynamic step size satisfies an unconditional asymptotic vanishing bound. It is presented and evaluated as an **adaptive tracking heuristic** designed to accelerate convergence during abrupt transitions, and is benchmarked directly against the provably convergent Fixed-Step Reference Controller.

---

## 6. Delayed Feedback Queue & Event Semantics

When labels arrive with delay $\delta > 0$:
1. At prediction step $t$, the system constructs set $\mathcal{C}_t(x_t)$ and emits a **deeply immutable** `PredictionRecord`:
   - Cloned FP32 logits and conformity scores.
   - Immutable tuple of class indices.
   - Timestamp $t$ and modality group key $g$.
2. The record is held in a FIFO feedback queue.
3. At time $t' = t + \delta$, the true label $y_t$ arrives. The error $e_t = \mathbb{I}\{y_t \notin \mathcal{C}_t(x_t)\}$ is evaluated strictly against the historical prediction set $\mathcal{C}_t$, never retroactively recomputed.
4. The controller state $\alpha_{t', g}$ is then updated using $e_t$.

---

## 7. Pooled Global Fallback for Rare Masks

For mask groups with limited empirical support ($n_g < N_{\text{min}}$):
- Scores are drawn from the pooled global window $\mathcal{S}_{\text{global}}$.
- We explicitly designate this mechanism as **Pooled Global Fallback** (rather than "hierarchical shrinkage").
- We disclose that pooled global fallback provides empirical stability for cold-start modalities, but does not inherit mask-conditional coverage guarantees when score distributions differ substantially across sensor subsets.
