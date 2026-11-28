# V14-02: Forensic Analysis of V12–V13.2 Methods

1. **BRMO (V12/V12.1) & Conditional Utility / Derived Threshold (V13/V13.1):**
   - *Equivalence Audit:* When conditional gain `G(s,a)` depends only on `max(p_S)` and globally fixed rescue/harm rates, `G(s,a) > lambda * cost(a)` reduces algebraically to a monotonic threshold on `max(p_S)` — i.e., **mathematically equivalent to Confidence Routing**!
   - *Why V13.1 Conditional Utility failed to beat Confidence Routing:* Low-dimensional confidence-only conditioning cannot distinguish states where adding `V` or `A` **harms** a correct prediction (`H(s,a)`) from states where it **rescues** an error (`R(s,a)`).
2. **SequentialConfidence_0.55_0.45 vs OneShot_ConfidenceRouting_0.55 (V13.2):**
   - Phase II showed `OneShot` (`0.4897` SQ-F1, `156.2 B`) slightly exceeds `Sequential` (`0.4895` SQ-F1, `134.6 B`), while `Unimodal_Text` achieves `0.4834` SQ-F1 at `5.0 B`.
   - Fixed `T -> TA -> TAV` order cannot skip `A` when `V` is cleaner or dispatch `{A,V}` in parallel when `t_remaining` is tight for 2 sequential RTTs.
3. **Diagnostic Evidence (`val`, `N=250`):** Across 7 workloads (`1,750` pairs), `T -> TV` produces `158` rescues vs **`246` harms**, and `T -> TAV` produces `163` rescues vs **`184` harms**, confirming that fixed unconditioned escalation frequently degrades correct text predictions.
