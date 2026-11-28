# V14-03: Formal Decision-Theoretic Problem Formulation

- **Observable State:** `s = (S, z_S, p_S, q_S, t_remaining, b_remaining, n_obs, history)`, strictly excluding unacquired modality features `z_{M \ S}` and ground-truth labels.
- **Feasible Action Space `A(s)`:** `{STOP, ACQUIRE_A, ACQUIRE_V, ACQUIRE_AV_PARALLEL}` constrained by modality availability, `b_remaining`, and `t_remaining` (sequential `1` round for `A`/`V`, `1` parallel round for `AV_PARALLEL` vs `2` sequential rounds for `A` then `V`).
- **Rescue & Harm Decomposition:**
  - `R(s,a) = P(c_s = 0, c_{s,a} = 1 | s, a)` and `H(s,a) = P(c_s = 1, c_{s,a} = 0 | s, a)`.
  - `E[c_{s,a} - c_s | s, a] = R(s,a) - H(s,a)` (elementary identity, **not** claimed as a novel theorem).
  - Under **risk-averse / asymmetric harm penalty** `alpha > 1`: `U(s,a) = R(s,a) - alpha * H(s,a) - lambda_c * cost(a) - lambda_t * P(T > t_remaining | s, a)`.
- **Macro-F1 Alignment:** Macro-F1 is nonlinear; per-request class-weighted correctness surrogates are used for policy optimization and evaluated via exact Service-Quality Macro-F1.
- **Sequential-Parallel Option Value:** Acquiring `A` sequentially preserves the option to `STOP` before `V` (saving `B_V` bytes), at the cost of consuming `RTT_A` of `t_remaining`; `AV_PARALLEL` trades away that option value to save `1` round-trip on the critical path.
