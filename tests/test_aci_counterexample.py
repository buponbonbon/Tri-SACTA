"""
Unit test for R3-01: Counterexample to Variable-Step Online ACI Theorem 2.

Mathematical Statement of the Counterexample:
Target nominal miscoverage: alpha = 0.10.
Initial state: alpha_1 = 0.50.
Let the error sequence and adaptive step sequence alternate strictly:
  Step t odd:  e_t = 1 (miscoverage), step size eta_t = 0.01
               Update: alpha_{t+1} = alpha_t + 0.01 * (0.10 - 1.0) = alpha_t - 0.009
  Step t even: e_t = 0 (coverage),    step size eta_t = 0.09
               Update: alpha_{t+1} = alpha_t + 0.09 * (0.10 - 0.0) = alpha_t + 0.009

Properties of this trajectory:
1. Step bounds: eta_t in [0.01, 0.09] for all t, satisfying 0 < eta_min <= eta_t <= eta_max < 1.0.
2. Interior state: alpha_t strictly alternates between 0.50 and 0.491.
   It NEVER touches boundaries {0.0, 1.0} and NEVER gets clipped.
3. Average empirical miscoverage:
   For any even T = 2K:
     (1/T) sum_{t=1}^T e_t = (1/(2K)) * K = 0.50.
   Target nominal miscoverage is alpha = 0.10.
   Tracking error | (1/T) sum e_t - alpha | = |0.50 - 0.10| = 0.40 != 0 as T -> infty.

Conclusion:
Theorem 2 and Proposition 1 from Tri-SACTA V2 (which claimed that any bounded sequence
eta_t in [eta_min, eta_max] guarantees asymptotic convergence to alpha) are mathematically INVALID.
Unweighted asymptotic convergence holds if and only if step sizes are constant, or if
total variation sum_{t=2}^T |1/eta_t - 1/eta_{t-1}| = o(T) and clipping residuals sum |c_t|/eta_t = o(T).
"""

import numpy as np


def simulate_alternating_aci_counterexample(T=10000, alpha_target=0.10, alpha_init=0.50):
    """Simulates the variable-step alternating sequence and tracks miscoverage."""
    assert T % 2 == 0, "T must be even for exact symmetry"
    
    alpha = alpha_init
    errors = []
    alphas = [alpha]
    etas = []
    
    for t in range(1, T + 1):
        if t % 2 == 1:
            # Odd step: miscoverage
            e_t = 1
            eta_t = 0.01
        else:
            # Even step: coverage
            e_t = 0
            eta_t = 0.09
            
        errors.append(e_t)
        etas.append(eta_t)
        
        # Exact ACI update
        delta = eta_t * (alpha_target - e_t)
        alpha = alpha + delta
        alphas.append(alpha)
        
    errors = np.array(errors)
    alphas = np.array(alphas)
    etas = np.array(etas)
    
    mean_error = np.mean(errors)
    min_alpha = np.min(alphas)
    max_alpha = np.max(alphas)
    
    return {
        "T": T,
        "mean_error": mean_error,
        "alpha_target": alpha_target,
        "tracking_gap": abs(mean_error - alpha_target),
        "min_alpha": min_alpha,
        "max_alpha": max_alpha,
        "eta_min": np.min(etas),
        "eta_max": np.max(etas),
    }


def test_variable_step_aci_counterexample():
    """T01: Proves mathematically that variable-step ACI without TV constraint fails convergence."""
    res = simulate_alternating_aci_counterexample(T=10000, alpha_target=0.10, alpha_init=0.50)
    
    # Assert step sizes remain bounded
    assert res["eta_min"] == 0.01
    assert res["eta_max"] == 0.09
    
    # Assert alpha never clipped and stays strictly in interior (0.491 to 0.50)
    assert 0.490 < res["min_alpha"] < 0.501
    assert 0.490 < res["max_alpha"] < 0.501
    
    # Assert empirical mean error is exactly 0.50
    assert abs(res["mean_error"] - 0.50) < 1e-6
    
    # The tracking gap is 0.40, disproving asymptotic convergence to alpha=0.10!
    assert abs(res["tracking_gap"] - 0.40) < 1e-6
    print(f"R3-01 Counterexample verified: Mean Error = {res['mean_error']:.4f}, Target = {res['alpha_target']:.4f}, Gap = {res['tracking_gap']:.4f}")


def test_fixed_step_telescoping_identity():
    """T02: Verifies that constant step size eta strictly satisfies the telescoping identity."""
    T = 2000
    alpha_target = 0.10
    eta = 0.05
    alpha = 0.50
    
    # Random error sequence
    np.random.seed(42)
    # Generate arbitrary non-exchangeable errors (e.g. regime shifts)
    errors = np.zeros(T, dtype=int)
    errors[:500] = (np.random.rand(500) < 0.30).astype(int)   # 30% error
    errors[500:1500] = (np.random.rand(1000) < 0.05).astype(int) # 5% error
    errors[1500:] = (np.random.rand(500) < 0.20).astype(int)  # 20% error
    
    alpha_t = alpha
    alphas = [alpha_t]
    for e_t in errors:
        # Unprojected update
        alpha_t = alpha_t + eta * (alpha_target - e_t)
        alphas.append(alpha_t)
        
    # Telescoping identity:
    # sum_{t=1}^T (alpha_target - e_t) = (alpha_{T+1} - alpha_1) / eta
    lhs = np.sum(alpha_target - errors)
    rhs = (alphas[-1] - alphas[0]) / eta
    
    assert abs(lhs - rhs) < 1e-6, f"Telescoping identity failed: LHS={lhs}, RHS={rhs}"
    
    # Average error identity:
    # (1/T) sum e_t - alpha_target = (alpha_1 - alpha_{T+1}) / (T * eta)
    mean_err = np.mean(errors)
    predicted_diff = (alphas[0] - alphas[-1]) / (T * eta)
    actual_diff = mean_err - alpha_target
    assert abs(actual_diff - predicted_diff) < 1e-6
    print(f"Telescoping identity verified: actual_diff={actual_diff:.6f}, predicted_diff={predicted_diff:.6f}")


if __name__ == "__main__":
    test_variable_step_aci_counterexample()
    test_fixed_step_telescoping_identity()
    print("ALL ACI COUNTEREXAMPLE AND TELESCOPING TESTS PASSED.")
