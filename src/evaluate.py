
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Dict, Any, List, Union
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)

from . import config


def evaluate_model(
    model: Any, 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    label_names: List[str] = None,
    model_name: str = "model"
) -> Dict[str, Any]:
    print(f"Evaluating model on {X_test.shape[0]} test samples...")
    
    if not hasattr(model, 'predict'):
        raise ValueError("Model must have a predict method")
    
    if not hasattr(model, 'predict_proba'):
        raise ValueError("Model must have a predict_proba method for ROC AUC calculation")
    
    if len(X_test) != len(y_test):
        raise ValueError(f"X_test and y_test length mismatch: {len(X_test)} vs {len(y_test)}")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    unique_labels = sorted(np.unique(y_test))
    
    if label_names is not None and len(label_names) == len(unique_labels):
        if all(isinstance(x, (int, np.integer)) for x in unique_labels) and unique_labels == list(range(len(unique_labels))):
            display_names = label_names
            print(f"Using provided label names for encoded labels: {display_names}")
        else:
            display_names = [str(label) for label in unique_labels]
            print(f"Using actual label values: {display_names}")
    else:
        display_names = [str(label) for label in unique_labels]
        if label_names is not None:
            print(f"Warning: label_names length ({len(label_names)}) doesn't match unique labels ({len(unique_labels)})")
        print(f"Using unique labels: {display_names}")
    
    print(f"Evaluating {len(unique_labels)} classes: {unique_labels}")
    
    metrics = {}
    
    accuracy = accuracy_score(y_test, y_pred)
    metrics['accuracy'] = float(accuracy)
    
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    metrics.update({
        'precision_macro': float(precision_macro),
        'precision_weighted': float(precision_weighted),
        'recall_macro': float(recall_macro),
        'recall_weighted': float(recall_weighted),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted)
    })
    
    class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    metrics['classification_report'] = class_report
    
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
    metrics['confusion_matrix'] = cm.tolist()
    
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
        roc_auc_weighted = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        metrics['roc_auc_ovr_macro'] = float(roc_auc)
        metrics['roc_auc_ovr_weighted'] = float(roc_auc_weighted)
    except Exception as e:
        print(f"Warning: Could not calculate ROC AUC: {str(e)}")
        metrics['roc_auc_ovr_macro'] = None
        metrics['roc_auc_ovr_weighted'] = None
    
    metrics['metadata'] = {
        'n_samples': int(len(y_test)),
        'n_classes': int(len(unique_labels)),
        'class_names': display_names,
        'class_distribution': {int(k): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))}
    }
    
    print(f"Evaluation completed:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 (macro): {f1_macro:.4f}")
    print(f"  F1 (weighted): {f1_weighted:.4f}")
    if metrics['roc_auc_ovr_macro'] is not None:
        print(f"  ROC AUC (macro): {roc_auc:.4f}")
    
    save_metrics(metrics, model_name)
    save_confusion_matrix_plot(cm, display_names, unique_labels, model_name)
    save_classification_report(class_report, model_name)
    
    if hasattr(model, 'predict_proba'):
        save_roc_curve_plot(y_test, y_pred_proba, display_names, model_name)
    
    return metrics


def save_metrics(metrics: Dict[str, Any], model_name: str) -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    filename = f"metrics_{model_name}.json"
    output_path = config.RESULTS_DIR / filename
    
    try:
        metrics_serializable = convert_for_json(metrics)
        
        with open(output_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        print(f"Metrics saved to: {output_path}")
        
    except Exception as e:
        print(f"Error saving metrics: {str(e)}")
        raise


def save_confusion_matrix_plot(
    cm: np.ndarray, 
    display_names: List[str], 
    unique_labels: List[str], 
    model_name: str
) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    filename = f"confusion_matrix_{model_name}.png"
    output_path = config.FIGURES_DIR / filename
    
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        
        plt.colorbar(im, ax=ax)
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        n_classes = len(display_names)
        ax.set_xticks(np.arange(n_classes))
        ax.set_yticks(np.arange(n_classes))
        ax.set_xticklabels(display_names, rotation=45, ha='right')
        ax.set_yticklabels(display_names)
        
        thresh = cm.max() / 2.
        for i in range(n_classes):
            for j in range(n_classes):
                text_color = "white" if cm[i, j] > thresh else "black"
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center", 
                       color=text_color, fontsize=10)
        
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix plot saved to: {output_path}")
        
    except Exception as e:
        print(f"Error saving confusion matrix plot: {str(e)}")
        raise


def save_roc_curve_plot(
    y_test: np.ndarray,
    y_pred_proba: np.ndarray, 
    class_names: List[str],
    model_name: str
) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    filename = f"roc_curve_{model_name}.png"
    output_path = config.FIGURES_DIR / filename
    
    try:
        from sklearn.preprocessing import label_binarize
        
        n_classes = len(class_names)
        
        y_test_binarized = label_binarize(y_test, classes=range(n_classes))
        
        # label_binarize returns a single column for binary; rebuild both columns
        if n_classes == 2:
            y_test_binarized = np.column_stack([1 - y_pred_proba[:, 1], y_pred_proba[:, 1]])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for i in range(n_classes):
            if n_classes == 2 and i == 0:
                continue
                
            fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
            roc_auc = roc_auc_score(y_test_binarized[:, i], y_pred_proba[:, i])
            
            ax.plot(fpr, tpr, linewidth=2, 
                   label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.8)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curves (One-vs-Rest) - {model_name}', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"ROC curve plot saved to: {output_path}")
        
    except Exception as e:
        print(f"Error saving ROC curve plot: {str(e)}")
        raise


def save_classification_report(class_report: Dict[str, Any], model_name: str) -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    filename = f"classification_report_{model_name}.json"
    output_path = config.RESULTS_DIR / filename
    
    try:
        report_serializable = convert_for_json(class_report)
        
        with open(output_path, 'w') as f:
            json.dump(report_serializable, f, indent=2)
        
        print(f"Classification report saved to: {output_path}")
        
    except Exception as e:
        print(f"Error saving classification report: {str(e)}")
        raise


def convert_for_json(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(item) for item in obj]
    else:
        return obj


def load_metrics(model_name: str) -> Dict[str, Any]:
    filename = f"metrics_{model_name}.json"
    metrics_path = config.RESULTS_DIR / filename
    
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        print(f"Metrics loaded from: {metrics_path}")
        return metrics
        
    except Exception as e:
        print(f"Error loading metrics: {str(e)}")
        raise


def compare_models(metrics_list: List[Dict[str, Any]], model_names: List[str] = None) -> pd.DataFrame:
    if model_names is None:
        model_names = [f"Model_{i+1}" for i in range(len(metrics_list))]
    
    if len(metrics_list) != len(model_names):
        raise ValueError("Length of metrics_list and model_names must match")
    
    comparison_data = []
    
    for name, metrics in zip(model_names, metrics_list):
        row = {
            'Model': name,
            'Accuracy': metrics.get('accuracy', None),
            'F1_Macro': metrics.get('f1_macro', None),
            'F1_Weighted': metrics.get('f1_weighted', None),
            'Precision_Macro': metrics.get('precision_macro', None),
            'Recall_Macro': metrics.get('recall_macro', None),
            'ROC_AUC_Macro': metrics.get('roc_auc_ovr_macro', None)
        }
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    if 'F1_Macro' in comparison_df.columns:
        comparison_df = comparison_df.sort_values('F1_Macro', ascending=False, na_last=True)
    
    return comparison_df


def print_detailed_metrics(metrics: Dict[str, Any]) -> None:
    print("\n" + "="*50)
    print("DETAILED EVALUATION RESULTS")
    print("="*50)
    
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"Recall (weighted): {metrics['recall_weighted']:.4f}")
    print(f"F1 Score (macro): {metrics['f1_macro']:.4f}")
    print(f"F1 Score (weighted): {metrics['f1_weighted']:.4f}")
    
    if metrics['roc_auc_ovr_macro'] is not None:
        print(f"ROC AUC (macro): {metrics['roc_auc_ovr_macro']:.4f}")
        print(f"ROC AUC (weighted): {metrics['roc_auc_ovr_weighted']:.4f}")
    
    print(f"\nTest set distribution:")
    for class_name, count in metrics['metadata']['class_distribution'].items():
        print(f"  {class_name}: {count}")
    
    print(f"\nPer-class metrics:")
    class_report = metrics['classification_report']
    for class_name in metrics['metadata']['class_names']:
        if class_name in class_report:
            cr = class_report[class_name]
            print(f"  {class_name}:")
            print(f"    Precision: {cr['precision']:.4f}")
            print(f"    Recall: {cr['recall']:.4f}")
            print(f"    F1-score: {cr['f1-score']:.4f}")
            print(f"    Support: {cr['support']}")


if __name__ == "__main__":
    print("Testing evaluation functions...")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    np.random.seed(42)
    n_samples, n_features = 300, 20
    
    X_data = np.random.randn(n_samples, n_features)
    y_data = np.random.choice(['Healthy', 'IBD', 'MS', 'RA'], size=n_samples)
    
    for i, class_label in enumerate(['Healthy', 'IBD', 'MS', 'RA']):
        mask = y_data == class_label
        X_data[mask, i*2:(i+1)*2] += 2
    
    feature_names = [f"Feature_{i}" for i in range(n_features)]
    X = pd.DataFrame(X_data, columns=feature_names)
    y = pd.Series(y_data, name='label')
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Created test data: train {X_train.shape}, test {X_test.shape}")
    print(f"Test set distribution: {dict(y_test.value_counts())}")
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"\nTrained RandomForest model")
    
    label_names = ['Healthy', 'IBD', 'MS', 'RA']
    metrics = evaluate_model(model, X_test, y_test, label_names, "RandomForest")
    
    print_detailed_metrics(metrics)
    
    print(f"\nTesting metrics loading...")
    loaded_metrics = load_metrics("RandomForest")
    print(f"Loaded metrics keys: {list(loaded_metrics.keys())}")
    
    print("Evaluation testing completed successfully!")