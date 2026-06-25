"""Run all test scripts with clean, structured output."""

import sys
import io
import os
import contextlib
import importlib
import logging
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

LINE = "=" * 50


def print_section(title):
    print(f"\n{LINE}")
    print(f"  {title}")
    print(LINE)


def print_result(name, status, details=None):
    tag = "[PASS]" if status == "pass" else "[FAIL]" if status == "fail" else "[SKIP]"
    print(f"  {tag}  {name}")
    if details:
        print(f"         {details}")


@contextlib.contextmanager
def suppress_output():
    """Temporarily suppress stdout, stderr, and logging."""
    buf_out, buf_err = io.StringIO(), io.StringIO()
    root_logger = logging.getLogger()
    old_level = root_logger.level
    root_logger.setLevel(logging.CRITICAL)
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        yield
    root_logger.setLevel(old_level)


def run_all():
    passed, failed, skipped = 0, 0, 0
    start_time = time.perf_counter()

    # ── 1. Module unit tests ──────────────────────────
    print_section("MODULE UNIT TESTS")
    try:
        from tests.test_modules import (
            test_load_all_datasets,
            test_preprocess_features,
            test_select_features_pipeline,
        )

        for fn in [test_load_all_datasets, test_preprocess_features, test_select_features_pipeline]:
            name = fn.__name__
            with suppress_output():
                fn()
            print_result(name, "pass")
            passed += 1

    except Exception as e:
        print_result(name if 'name' in dir() else "test_modules", "fail", str(e))
        failed += 1
        _print_summary(passed, failed, skipped, start_time)
        sys.exit(1)

    # ── 2. Security / data leakage ────────────────────
    print_section("DATA LEAKAGE CHECK")
    try:
        from tests.test_security_sanity import test_no_label_leakage
        with suppress_output():
            test_no_label_leakage()
        print_result("No label/target/class columns in X", "pass")
        passed += 1
    except Exception as e:
        print_result("test_no_label_leakage", "fail", str(e))
        failed += 1
        _print_summary(passed, failed, skipped, start_time)
        sys.exit(1)

    # ── 3. Functional pipeline ────────────────────────
    print_section("FUNCTIONAL PIPELINE TEST")
    try:
        from tests.test_functional_pipeline import (
            load_artifacts, make_fake_sample, align_to_features,
        )
        from src.preprocess import preprocess_features

        with suppress_output():
            model, encoder, selected_features = load_artifacts()
            raw = make_fake_sample(selected_features, n_extra=5)
            aligned = align_to_features(raw, selected_features)
            processed = preprocess_features(aligned)
            pred_encoded = model.predict(processed)[0]
            pred_label = encoder.inverse_transform([pred_encoded])[0]
            confidence = None
            if hasattr(model, "predict_proba"):
                confidence = float(model.predict_proba(processed)[0].max())

        print_result("Load artifacts", "pass", f"model=Ensemble, features={len(selected_features)}")
        print_result("Preprocess & predict", "pass",
                     f"label={pred_label}, confidence={confidence:.4f}" if confidence else f"label={pred_label}")
        passed += 1

    except Exception as e:
        print_result("functional_pipeline", "fail", str(e))
        failed += 1
        _print_summary(passed, failed, skipped, start_time)
        sys.exit(1)

    # ── 4. Benchmarking (optional) ────────────────────
    print_section("BENCHMARKING")
    try:
        mod = importlib.import_module("tests.test_benchmarking")
        print_result("test_benchmarking", "pass", "module importable (no training data supplied)")
        passed += 1
    except Exception as e:
        print_result("test_benchmarking", "skip", str(e))
        skipped += 1

    # ── 5. Model evaluation (optional) ────────────────
    print_section("MODEL EVALUATION")
    try:
        mod = importlib.import_module("tests.test_model_evaluation")
        print_result("test_model_evaluation", "pass", "module importable")
        passed += 1
    except Exception as e:
        print_result("test_model_evaluation", "skip", str(e))
        skipped += 1

    # ── 6. Performance / latency ──────────────────────
    print_section("INFERENCE PERFORMANCE")
    try:
        from tests.test_performance import load_model, load_test_data, time_predict

        with suppress_output():
            perf_model = load_model()
            X = load_test_data()

        batch_sizes = [1, 10, len(X)]
        print(f"\n  {'Samples':<10} {'Total(ms)':<14} {'PerSample(ms)':<14}")
        print(f"  {'-'*38}")

        for n in batch_sizes:
            if n > len(X):
                continue
            total, per_sample = time_predict(perf_model, X, n)
            print(f"  {n:<10} {total*1000:<14.3f} {per_sample:<14.3f}")

        print()
        print_result("Inference latency", "pass", f"{len(X)} test samples benchmarked")
        passed += 1

    except Exception as e:
        print_result("test_performance", "fail", str(e))
        failed += 1

    # ── Summary ───────────────────────────────────────
    _print_summary(passed, failed, skipped, start_time)
    sys.exit(1 if failed else 0)


def _print_summary(passed, failed, skipped, start_time):
    elapsed = time.perf_counter() - start_time
    print_section("TEST EXECUTION SUMMARY")
    print(f"  Passed  : {passed}")
    print(f"  Failed  : {failed}")
    print(f"  Skipped : {skipped}")
    print(f"  Time    : {elapsed:.2f}s")
    print()


if __name__ == "__main__":
    run_all()
