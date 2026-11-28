# V14-10: V15 Execution Specification (Design Only — Not Implemented)

1. **Inputs:** Frozen 21 predictor checkpoints, `train` (`N=16,327`) counterfactual table for fitting `R_hat/H_hat` estimators, `cal` (`N=250`) for calibration/hyperparameter selection, `val` (`N=250`) for pre-freeze dev validation.
2. **Baselines:** `Unimodal_Text`, `AlwaysOn_Parallel`, `AlwaysOn_Serial`, `Static_TA`, `BudgetAdmittedMaxSubset`, `OneShot_ConfidenceRouting_0.55`, `SequentialConfidence_0.55_0.45`, `MatchedBudgetRandom_1.45`, Direct Expected-Gain, and `JOV-Acquisition`.
3. **Ablations & Negative Controls:** (i) `alpha = 1` vs `alpha > 1` (testing rescue-harm asymmetry), (ii) sequential-only vs sequential+parallel action space, (iii) shuffled/random `z_S` features, (iv) `TEXT_EMBEDDING_NOISE` overconfident-early-exit stress test.
4. **Minimum Conditions for Continuation:** Must demonstrate statistically supported paired improvement over `OneShot_ConfidenceRouting_0.55` and `SequentialConfidence_0.55_0.45` in Service-Quality Macro-F1 or Pareto dominance under controlled RTT regimes on a fresh non-exposed validation split.
