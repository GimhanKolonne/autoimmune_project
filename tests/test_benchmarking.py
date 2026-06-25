import pandas as pd
import numpy as np
from typing import Any, Dict
from sklearn.model_selection import cross_val_score


def benchmark_models(
    models: Dict[str, Any],
    X_train,
    y_train,
    cv: int = 5,
    scoring: str = "accuracy",
) -> pd.DataFrame:
    """Run cross-validation on each model and return a summary DataFrame."""

    results = []

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
        mean, std = scores.mean(), scores.std()
        print(f"{name}: {mean:.4f} ± {std:.4f}")
        results.append({"Model": name, "CV_Mean": round(mean, 4), "CV_Std": round(std, 4)})

    df = pd.DataFrame(results).sort_values("CV_Mean", ascending=False).reset_index(drop=True)
    return df
