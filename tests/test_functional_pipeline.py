"""
Functional end-to-end pipeline test.
Loads saved artifacts, builds a fake sample, preprocesses, predicts, and prints the result.
Run directly: python -m tests.test_functional_pipeline
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Allow imports from project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocess import preprocess_features
from src import config

ENSEMBLE_PATH = config.MODELS_DIR / "model_Ensemble.joblib"
ENCODER_PATH = config.MODELS_DIR / "label_encoder.joblib"
FEATURES_PATH = config.RESULTS_DIR / "selected_features.json"


def load_artifacts():
    """Load ensemble model, label encoder, and selected feature list."""
    for p in (ENSEMBLE_PATH, ENCODER_PATH, FEATURES_PATH):
        if not p.exists():
            raise FileNotFoundError(f"Missing artifact: {p}")

    model = joblib.load(ENSEMBLE_PATH)
    encoder = joblib.load(ENCODER_PATH)
    with open(FEATURES_PATH) as f:
        features = json.load(f)

    return model, encoder, features


def make_fake_sample(selected_features: list, n_extra: int = 5) -> pd.DataFrame:
    """Create a single-row DataFrame with more columns than needed."""
    rng = np.random.default_rng(42)

    # Start with selected features + some extra junk columns
    all_cols = list(selected_features) + [f"extra_col_{i}" for i in range(n_extra)]
    rng.shuffle(all_cols)

    values = rng.uniform(0, 100, size=(1, len(all_cols)))
    return pd.DataFrame(values, columns=all_cols)


def align_to_features(df: pd.DataFrame, selected_features: list) -> pd.DataFrame:
    """Keep only selected features; fill missing ones with 0."""
    aligned = pd.DataFrame(0.0, index=df.index, columns=selected_features)
    overlap = [c for c in selected_features if c in df.columns]
    aligned[overlap] = df[overlap].values
    return aligned


def run():
    print("Loading saved artifacts...")
    model, encoder, selected_features = load_artifacts()
    print(f"  Ensemble loaded | {len(selected_features)} features | classes: {list(encoder.classes_)}")

    print("Creating fake sample (25 cols)...")
    raw = make_fake_sample(selected_features, n_extra=5)
    print(f"  Raw shape: {raw.shape}")

    print("Aligning to selected features...")
    aligned = align_to_features(raw, selected_features)
    print(f"  Aligned shape: {aligned.shape}")

    print("Preprocessing...")
    processed = preprocess_features(aligned)

    print("Predicting...")
    pred_encoded = model.predict(processed)[0]
    pred_label = encoder.inverse_transform([pred_encoded])[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(processed)[0]
        confidence = float(proba.max())

    print(f"\nResult:")
    print(f"  Predicted label : {pred_label}")
    print(f"  Confidence      : {f'{confidence:.4f}' if confidence is not None else 'N/A'}")


if __name__ == "__main__":
    run()
