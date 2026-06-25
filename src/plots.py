import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Union


def plot_xgboost_feature_importance(
    model, 
    feature_names: List[str], 
    out_path: Union[str, Path], 
    top_n: int = 15
) -> None:
    """Horizontal bar chart of top-N XGBoost feature importances."""
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model must have feature_importances_ attribute")
    
    if len(feature_names) != len(model.feature_importances_):
        raise ValueError(f"Feature names length ({len(feature_names)}) must match "
                        f"importances length ({len(model.feature_importances_)})")
    
    importances = model.feature_importances_
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(10, 8))
    
    y_pos = np.arange(len(top_features))
    
    bars = plt.barh(y_pos, top_features['importance'], alpha=0.7, color='steelblue')
    
    plt.xlabel('Feature Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('Top Feature Importances (XGBoost)', fontsize=14, fontweight='bold')
    plt.yticks(y_pos, top_features['feature'].tolist())
    
    for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
        plt.text(importance + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{importance:.3f}', 
                ha='left', va='center', fontsize=9)
    
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Feature importance plot saved to: {out_path}")


def plot_model_comparison(
    results_dict: dict, 
    metric: str, 
    out_path: Union[str, Path],
    title: str = None
) -> None:
    """Bar chart comparing models on a given metric."""
    model_names = list(results_dict.keys())
    metric_values = [results_dict[name].get(metric, 0) for name in model_names]
    
    plt.figure(figsize=(10, 6))
    
    bars = plt.bar(model_names, metric_values, alpha=0.7, color='lightcoral')
    
    plt.xlabel('Models', fontsize=12)
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
    
    if title is None:
        title = f'Model Comparison - {metric.replace("_", " ").title()}'
    plt.title(title, fontsize=14, fontweight='bold')
    
    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.3f}', 
                ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Narrow y range to emphasise differences between models
    y_min = min(metric_values) * 0.95
    y_max = max(metric_values) * 1.05
    plt.ylim(y_min, y_max)
    
    plt.tight_layout()
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Model comparison plot saved to: {out_path}")


if __name__ == "__main__":
    print("Testing plotting functions...")
    
    # Mock XGBoost model for testing
    class MockXGBModel:
        def __init__(self):
            self.feature_importances_ = np.random.rand(20)
    
    # Test feature importance plot
    model = MockXGBModel()
    feature_names = [f"feature_{i}" for i in range(20)]
    
    plot_xgboost_feature_importance(
        model, 
        feature_names, 
        "test_feature_importance.png", 
        top_n=10
    )
    
    # Test model comparison plot
    mock_results = {
        'RandomForest': {'accuracy': 0.89, 'f1_macro': 0.87},
        'XGBoost': {'accuracy': 0.92, 'f1_macro': 0.91},
        'SVM': {'accuracy': 0.85, 'f1_macro': 0.84},
        'Ensemble': {'accuracy': 0.93, 'f1_macro': 0.92}
    }
    
    plot_model_comparison(
        mock_results, 
        'accuracy', 
        "test_model_comparison.png"
    )
    
    print("Plotting functions test completed successfully!")