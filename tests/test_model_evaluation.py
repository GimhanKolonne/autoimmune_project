import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: Union[np.ndarray, pd.Series],
    class_names: List[str],
    model_name: str,
) -> Dict[str, Optional[float]]:
    """Evaluate a fitted classifier and save confusion matrix + ROC curve plots."""

    y_pred = model.predict(X_test)
    y_arr = np.asarray(y_test)

    metrics: Dict[str, Optional[float]] = {
        "accuracy": float(accuracy_score(y_arr, y_pred)),
        "f1_macro": float(f1_score(y_arr, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_arr, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_arr, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_arr, y_pred, average="macro", zero_division=0)),
        "roc_auc_ovr_macro": None,
    }

    has_proba = hasattr(model, "predict_proba")

    if has_proba:
        y_proba = model.predict_proba(X_test)
        try:
            metrics["roc_auc_ovr_macro"] = float(
                roc_auc_score(y_arr, y_proba, multi_class="ovr", average="macro")
            )
        except ValueError:
            pass

    _save_confusion_matrix(y_arr, y_pred, class_names, model_name)

    if has_proba:
        _save_roc_curve(y_arr, y_proba, class_names, model_name)

    return metrics


def _save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    model_name: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title(f"Confusion Matrix — {model_name}")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"confusion_matrix_{model_name}.png", dpi=150)
    plt.close(fig)


def _save_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    model_name: str,
) -> None:
    unique_classes = sorted(np.unique(y_true))
    y_bin = label_binarize(y_true, classes=unique_classes)

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, cls in enumerate(unique_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        label = class_names[i] if i < len(class_names) else str(cls)
        ax.plot(fpr, tpr, label=label)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves (OVR) — {model_name}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"roc_curve_{model_name}.png", dpi=150)
    plt.close(fig)
