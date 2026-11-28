# V14-08: Primary & Backup Method Design

- **Primary Candidate (Candidate B — Joint Sequential-Parallel Option-Value Policy, `JOV-Acquisition`):**
  - *State:* `s = (S, p_S, entropy(p_S), margin(p_S), z_S_norm, t_rem, b_rem, rtt_est, avail_mask)`.
  - *Action Space:* `A(s) = {STOP, ACQ_A, ACQ_V, ACQ_AV_PAR}` filtered by `(avail_mask, b_rem, t_rem)`.
  - *Value Recursion (2-step finite horizon):*
    - `Q(s, STOP) = 0`
    - `Q(s, ACQ_AV_PAR) = R_hat(s, AV) - alpha * H_hat(s, AV) - lambda_c * (c_A + c_V) - lambda_t * P_miss(T_par > t_rem)`
    - `Q(s, ACQ_m) = R_hat(s, m) - alpha * H_hat(s, m) - lambda_c * c_m - lambda_t * P_miss(T_m > t_rem) + E_{s'|s,m}[max(0, Q(s', ACQ_other))]`
  - *Training:* Grouped cross-fitted calibrated logistic/MLP estimators trained on `train`/`val` counterfactual transitions (`0` future features at inference time).
- **Backup Candidate (Candidate C):** Conservative LCB Stopping (`G_hat(s,a) - kappa * sigma(s,a)`) if `JOV` value estimation exhibits high variance on out-of-group validation.
