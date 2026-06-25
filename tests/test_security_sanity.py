"""Sanity check: make sure no label-like columns leak into the feature matrix."""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.load_data import load_all_datasets

LEAK_KEYWORDS = ["label", "target", "class"]


def test_no_label_leakage():
    X, y = load_all_datasets()

    for col in X.columns:
        col_lower = col.lower()
        for keyword in LEAK_KEYWORDS:
            assert keyword not in col_lower, (
                f"Potential data leakage: column '{col}' contains '{keyword}'"
            )

    print(f"PASS  No label leakage detected ({X.shape[1]} columns checked)")


if __name__ == "__main__":
    test_no_label_leakage()
