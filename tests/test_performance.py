"""Measure inference latency and throughput for the saved ensemble model."""

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src import config

MODEL_PATH = config.MODELS_DIR / "model_Ensemble.joblib"
TEST_DATA_PATH = config.RESULTS_DIR / "X_test_selected.csv"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def load_test_data():
    if TEST_DATA_PATH.exists():
        df = pd.read_csv(TEST_DATA_PATH)
        print(f"Loaded test data: {df.shape}")
        return df

    print("X_test_selected.csv not found — generating random data (19 cols)")
    rng = np.random.default_rng(42)
    return pd.DataFrame(rng.random((50, 19)), columns=[f"f{i}" for i in range(19)])


def time_predict(model, X, n_samples, n_repeats=5):
    """Average prediction time over several repeats."""
    subset = X.iloc[:n_samples]
    times = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        model.predict(subset)
        times.append(time.perf_counter() - start)
    total = np.median(times)
    per_sample = total / n_samples * 1000  # ms
    return total, per_sample


def run():
    model = load_model()
    X = load_test_data()

    batch_sizes = [1, 10, len(X)]

    print(f"\n{'Samples':<10} {'Total (ms)':<15} {'Per sample (ms)':<15}")
    print("-" * 40)

    for n in batch_sizes:
        if n > len(X):
            continue
        total, per_sample = time_predict(model, X, n)
        print(f"{n:<10} {total*1000:<15.3f} {per_sample:<15.3f}")


if __name__ == "__main__":
    run()
