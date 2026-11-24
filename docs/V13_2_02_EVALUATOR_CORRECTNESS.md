# Tri-SACTA V13.2: Evaluator Correctness, Budget Propagation & Estimand Definition

**Audit Timestamp:** 2026-11-24 UTC  
**Active Branch:** `research/v13.2-final-systems-validation-release-candidate`  
**Execution Script:** [test_v13_2_evaluator_regression.py](file:///c:/Users/ASUS/Desktop/Tri-SACTA/tests/test_v13_2_evaluator_regression.py)  
**Policy Implementations:** [baseline_policies.py](file:///c:/Users/ASUS/Desktop/Tri-SACTA/src/service/baseline_policies.py) & [conditional_router.py](file:///c:/Users/ASUS/Desktop/Tri-SACTA/src/service/conditional_router.py)

---

## 1. P0 Defect: The Unconditional Budget Override

### Defect Identified
In Tri-SACTA V13.1, `scripts/v13_1_evaluate_policies.py` contained the following line inside its evaluation loop:
```python
res = policy.route_single_query(
    sample_feat, sample_mask, deadline_ms=d_ms, cost_budget=3.5
)
```
Although `scripts/v13_generate_workloads.py` generated heterogeneous dynamic cost budgets (e.g., cycling through `[1.0, 2.2, 3.5]` in Workload 7), the evaluator unconditionally passed `cost_budget=3.5` to every policy. Consequently:
1. Hard per-request resource limits were ignored during execution.
2. Policies were never prevented from escalating to full fusion ($C = 3.5$) on queries that were supposed to be constrained to $C \le 1.0$ or $C \le 2.2$.

### Rectification
In Tri-SACTA V13.2:
- The per-request cost budget `w_budgets[i]` is extracted directly from the workload manifest and propagated into every policy call:
  ```python
  res = policy.route_single_query(
      sample_feat,
      sample_mask,
      deadline_ms=float(w_deadlines[i]),
      cost_budget=float(w_budgets[i]),
  )
  ```
- Every routing policy checks that `spent_cost + Delta C(mod) <= cost_budget` and `spent_latency + Delta L(mod) <= deadline_ms` before each sequential transition.

---

## 2. Infeasible Initial Acquisition and Blackout Handling

### Infeasible Initial Modality
If a query specifies a `cost_budget < 1.0` (less than the minimum single-modality cost of Text: 1.0) or `deadline_ms < 1.0 ms`:
- The policy **fails closed**.
- It returns an explicit rejection rather than an unconstrained forward pass or fabricated prediction:
  ```python
  {
      "prediction": -1,
      "probabilities": np.zeros(3, dtype=np.float32),
      "final_subset": "NONE",
      "spent_cost": 0.0,
      "spent_latency_ms": 0.0,
      "acquisition_path": [],
      "routing_reason": "infeasible_budget_or_deadline",
      "status": "REJECTED_INFEASIBLE",
  }
  ```

### Total Blackout
If all modalities are unavailable (`modality_mask = {"text": False, "audio": False, "visual": False}`):
- The policy returns `status = "REJECTED_BLACKOUT"`, `prediction = -1`, `spent_cost = 0.0`.
- In V13.1, blackout queries fell back to Text with `prediction = 0` (negative sentiment) and charged 1.0 cost for an unavailable modality. This behavior has been eliminated.

---

## 3. Masked Feature Access Trap

To ensure that unacquired or masked features are never read or leaked:
- `BasePredictorPolicy._predict_subset` verifies that every required modality $m \in \text{SUBSET\_METADATA}[K][\text{"mods"}]$ is marked available in `modality_mask`.
- If an unavailable modality is queried, a `MaskedFeatureAccessViolation` is raised immediately.
- Verified in [test_v13_2_evaluator_regression.py](file:///c:/Users/ASUS/Desktop/Tri-SACTA/tests/test_v13_2_evaluator_regression.py).

---

## 4. Separation of Always-On and Budget-Admitted Max-Subset

To prevent baseline conflation:
1. **Always-On Unrestricted (`AlwaysOnFusionPolicy`):** Transmits all available modalities (e.g. `TAV`), spending nominal cost 3.5 and latency 14.0 ms. If `spent_cost > cost_budget` or `spent_latency > deadline_ms`, it sets `"budget_violated": True` / `"deadline_violated": True` while computing raw full-fusion accuracy.
2. **Budget-Admitted Max-Subset (`BudgetAdmittedMaxSubsetPolicy`):** Selects the single largest feasible subset within the declared budget without multi-stage sequential triage (e.g. `TAV` if $C \ge 3.5$, `TA` if $2.2 \le C < 3.5$, `T` if $1.0 \le C < 2.2$). If no subset is affordable, it rejects with `-1`.

---

## 5. Canonical Estimands & Mathematical Definition

### Primary Predictive Estimand: Unweighted Macro-F1
The primary endpoint across development and confirmatory benchmarks is the unweighted mean of per-(workload, seed) 3-class Macro-F1 with fixed label set $\mathcal{Y} = \{0, 1, 2\}$:
$$\text{Macro-F1}_{\text{primary}} = \frac{1}{|\mathcal{W}| \cdot |\mathcal{S}|} \sum_{w \in \mathcal{W}} \sum_{s \in \mathcal{S}} \text{Macro-F1}(y^{(w, s)}, \hat{y}^{(w, s)})$$
where:
$$\text{Macro-F1} = \frac{1}{3} \sum_{c=0}^2 \frac{2 \cdot TP_c}{2 \cdot TP_c + FP_c + FN_c}$$
For rejected queries ($\hat{y}_i = -1$):
- $\hat{y}_i = c$ is False for all $c \in \{0, 1, 2\}$.
- Therefore, $TP_c$ and $FP_c$ are unaffected, but $FN_c = \sum_i \mathbb{I}(y_i = c) - TP_c$ properly increments for the true class $y_i$.
- Parity with `sklearn.metrics.f1_score(..., labels=[0, 1, 2], average='macro', zero_division=0.0)` is **exact ($\Delta = 0.0000000$)**.

### Secondary Estimand: Pooled Confusion Matrix Macro-F1
Pooled across all queries and seeds:
$$\text{Macro-F1}_{\text{pooled}} = \frac{1}{3} \sum_{c=0}^2 \frac{2 \cdot \sum_{(w, s)} TP_c^{(w, s)}}{2 \cdot \sum_{(w, s)} TP_c^{(w, s)} + \sum_{(w, s)} FP_c^{(w, s)} + \sum_{(w, s)} FN_c^{(w, s)}}$$
Both metrics are reported independently, eliminating the ambiguity noted in V13.1.
