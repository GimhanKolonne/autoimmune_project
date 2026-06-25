"""Basic unit-style tests for core pipeline modules."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.load_data import load_all_datasets
from src.preprocess import preprocess_features
from src.features import select_features_pipeline


def test_load_all_datasets():
    X, y = load_all_datasets()

    assert X.shape[0] > 0, "X has no rows"
    assert X.shape[1] > 0, "X has no columns"
    assert len(y) == len(X), f"y length ({len(y)}) != X rows ({len(X)})"
    assert "label" not in X.columns, "'label' column should not be in X"

    print(f"  load_all_datasets: {X.shape[0]} rows, {X.shape[1]} cols, {len(y)} labels")


def test_preprocess_features():
    df = pd.DataFrame({
        "a": [1.0, np.nan, 3.0],
        "b": [-2.0, 5.0, 0.0],
        "c": [0.0, 0.0, np.nan],
    })

    out = preprocess_features(df)

    assert not out.isna().any().any(), "Output still has NaN values"
    assert np.all(np.isfinite(out.values)), "Output has non-finite values"
    assert out.shape == df.shape, f"Shape changed: {df.shape} -> {out.shape}"

    print(f"  preprocess_features: {out.shape}, no NaN, all finite")


def test_select_features_pipeline():
    rng = np.random.default_rng(0)
    n_samples, n_features = 40, 8

    X_train = pd.DataFrame(
        rng.standard_normal((n_samples, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    X_test = pd.DataFrame(
        rng.standard_normal((10, n_features)),
        columns=X_train.columns,
    )
    y_train = pd.Series(rng.choice([0, 1, 2], size=n_samples))

    # Request more features than available
    k = n_features + 5

    X_tr_sel, X_te_sel, selected, scores = select_features_pipeline(
        X_train, y_train, X_test, k=k
    )

    assert X_tr_sel.shape[0] == n_samples, "Train row count changed"
    assert X_te_sel.shape[0] == 10, "Test row count changed"
    assert X_tr_sel.shape[1] == n_features, f"Expected {n_features} features, got {X_tr_sel.shape[1]}"
    assert X_te_sel.shape[1] == X_tr_sel.shape[1], "Train/test column count mismatch"
    assert len(selected) == n_features, f"Selected list length should be {n_features}"

    print(f"  select_features_pipeline: k={k} clamped to {n_features}, shapes OK")


# --------------- runner ---------------

ALL_TESTS = [
    test_load_all_datasets,
    test_preprocess_features,
    test_select_features_pipeline,
]


if __name__ == "__main__":
    passed, failed = 0, 0

    for test_fn in ALL_TESTS:
        name = test_fn.__name__
        try:
            test_fn()
            print(f"PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {name}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed out of {len(ALL_TESTS)}")
    sys.exit(1 if failed else 0)
