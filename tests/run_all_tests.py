"""
Test discovery and execution runner for Tri-SACTA unit tests (T01 - T23).
Does not require external pytest binary; provides compatibility shim.
"""

import sys
import os
import inspect
import traceback
from contextlib import contextmanager

# Ensure repository root is on sys.path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Check or install pytest compatibility shim in sys.modules
if "pytest" not in sys.modules:
    try:
        import pytest
    except ImportError:
        class _RaisesContext:
            def __init__(self, expected_exc):
                self.expected_exc = expected_exc

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type is None:
                    raise AssertionError(f"Expected exception {self.expected_exc.__name__} was not raised.")
                return issubclass(exc_type, self.expected_exc)

        class PytestShim:
            @staticmethod
            def fixture(func):
                func._is_fixture = True
                return func

            @staticmethod
            def raises(expected_exc):
                return _RaisesContext(expected_exc)

        sys.modules["pytest"] = PytestShim()

# Import the test modules
import tests.test_spectral as test_spectral
import tests.test_masked_fusion as test_masked_fusion
import tests.test_selective_adaptation as test_selective_adaptation
import tests.test_conformal as test_conformal
import tests.test_data_integrity as test_data_integrity
import tests.test_metrics_rigor as test_metrics_rigor
import tests.test_shift_generator as test_shift_generator
import tests.test_feedback_delay as test_feedback_delay
import tests.test_v3_verification as test_v3_verification
import tests.test_data_authenticity as test_data_authenticity
import tests.test_v5_provenance as test_v5_provenance
import tests.test_v6_multimodal as test_v6_multimodal
import tests.test_v7_regression as test_v7_regression
import tests.test_v8_regression as test_v8_regression
import tests.test_v9_regression as test_v9_regression
import tests.test_v10_regression as test_v10_regression
import tests.test_v11_integrity as test_v11_integrity
import tests.test_v12_systems as test_v12_systems
import tests.test_v12_1_systems as test_v12_1_systems
import tests.test_v13_utility as test_v13_utility
import tests.test_v13_1_utility as test_v13_1_utility
import tests.test_v13_1_provenance as test_v13_1_provenance
import tests.test_v13_1_accounting as test_v13_1_accounting
import tests.test_v13_1_statistics as test_v13_1_statistics
import tests.test_v13_2_evaluator_regression as test_v13_2_evaluator_regression
import tests.test_v13_2_runtime_safety as test_v13_2_runtime_safety
import tests.test_v13_2_verifier_adversarial as test_v13_2_verifier_adversarial


def run_test_module(module, module_name):
    print(f"\n==========================================")
    print(f"Running Test Suite: {module_name}")
    print(f"==========================================")

    # Collect fixtures
    fixtures = {}
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if callable(attr) and getattr(attr, "_is_fixture", False):
            fixtures[attr_name] = attr

    # Collect test functions
    tests = []
    for attr_name in dir(module):
        if attr_name.startswith("test_"):
            attr = getattr(module, attr_name)
            if callable(attr):
                tests.append((attr_name, attr))

    passed = 0
    failed = 0

    for name, func in tests:
        sig = inspect.signature(func)
        kwargs = {}
        for param_name in sig.parameters:
            if param_name in fixtures:
                kwargs[param_name] = fixtures[param_name]()
            else:
                raise ValueError(f"Unknown fixture / parameter '{param_name}' in test '{name}'")

        try:
            func(**kwargs)
            doc = func.__doc__.strip() if func.__doc__ else name
            print(f"  [PASS] {name}: {doc.splitlines()[0]}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}")
            traceback.print_exc()
            failed += 1

    # Also collect unittest.TestCase classes if present
    import unittest
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and issubclass(attr, unittest.TestCase) and attr is not unittest.TestCase:
            suite = unittest.TestLoader().loadTestsFromTestCase(attr)
            res = unittest.TestResult()
            suite.run(res)
            test_names = [m for m in dir(attr) if m.startswith("test_") and callable(getattr(attr, m))]
            for tname in test_names:
                doc = getattr(attr, tname).__doc__
                doc_str = doc.strip().splitlines()[0] if doc else tname
                is_err = any(getattr(t, "_testMethodName", None) == tname for t, _ in res.errors + res.failures)
                if is_err:
                    print(f"  [FAIL] {attr_name}.{tname}: {doc_str}")
                    failed += 1
                else:
                    print(f"  [PASS] {attr_name}.{tname}: {doc_str}")
                    passed += 1

    return passed, failed


def main():
    test_suites = [
        (test_spectral, "Spectral Shift Detector (AUD-01, AUD-04)"),
        (test_masked_fusion, "Masked Multimodal Fusion (AUD-02, AUD-03)"),
        (test_selective_adaptation, "Selective Adaptation (AUD-05, AUD-06)"),
        (test_conformal, "Online Conformal Prediction (AUD-07, AUD-08, AUD-09)"),
        (test_data_integrity, "Data Integrity & Collator (AUD-10, AUD-12, AUD-14)"),
        (test_metrics_rigor, "Metrics Rigor & Footprint (AUD-20, AUD-23)"),
        (test_shift_generator, "Shift Stream Generator V2 (AUD-16)"),
        (test_feedback_delay, "Feedback Delay & Rare Mask Isolation (AUD-08, AUD-18)"),
        (test_v3_verification, "V3 Scientific Hardening Suite (R3-01 to R3-11)"),
        (test_data_authenticity, "V4 Data Authenticity & Quarantine Suite (R4-01 to R4-09)"),
        (test_v5_provenance, "V5 Provenance Enforcement & Acceptance Suite (V5-01 to V5-11)"),
        (test_v6_multimodal, "V6 Multimodal Authenticity & Selective Adaptation Suite (V6-01 to V6-11)"),
        (test_v7_regression, "V7 Performance Optimization & Negative Transfer Suite (Task 1 to 14)"),
        (test_v8_regression, "V8 Valid Baselines & Event Trace Suite (P0 and P1)"),
        (test_v9_regression, "V9 Scientific Evidence & Negative Transfer Suite (P0 and P1)"),
        (test_v10_regression, "V10 Independent Scientific Audit & Causal Validation Suite (V10-01 to V10-06)"),
        (test_v11_integrity, "V11 Scientific Triage & Go/No-Go Suite (V11-01 to V11-06)"),
        (test_v12_systems, "V12 Systems Research Reframing Suite (V12-01 to V12-06)"),
        (test_v12_1_systems, "V12.1 Budgeted Modality Orchestration & Systems Rectification Suite (V12_1-01 to V12_1-06)"),
        (test_v13_utility, "V13 Utility-Driven Multimodal Orchestration Suite (V13-01 to V13-08)"),
        (test_v13_1_utility, "V13.1 Utility Equivalence & Conditional Router Suite"),
        (test_v13_1_provenance, "V13.1 Provenance, Lineage & Unexposed Pool Suite"),
        (test_v13_1_accounting, "V13.1 Distributed Socket & Wire Byte Accounting Suite"),
        (test_v13_1_statistics, "V13.1 Fast Macro-F1 & Video Cluster Bootstrap Suite"),
        (test_v13_2_evaluator_regression, "V13.2 Evaluator Correctness & Regression Suite"),
        (test_v13_2_runtime_safety, "V13.2 Runtime Safety & Fair Distributed Transport Suite"),
        (test_v13_2_verifier_adversarial, "V13.2 Automated Fail-Closed Verifier Adversarial Suite"),
    ]

    total_passed = 0
    total_failed = 0

    for mod, name in test_suites:
        p, f = run_test_module(mod, name)
        total_passed += p
        total_failed += f

    print("\n==========================================")
    print(f"SUMMARY: {total_passed} PASSED, {total_failed} FAILED")
    print("==========================================")

    if total_failed > 0:
        sys.exit(1)
    else:
        print("ALL TESTS PASSED SUCCESSFULLY.")
        sys.exit(0)


if __name__ == "__main__":
    main()
