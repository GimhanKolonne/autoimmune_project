import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from typing import Any, List
from pathlib import Path

from . import config


def run_shap_xgboost(
    model: Any,
    X_train: pd.DataFrame,
    feature_names: List[str],
    out_dir: Path,
    top_n: int = 15
) -> None:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Cap at 500 samples for SHAP computation speed
        if len(X_train) > 500:
            sample_indices = np.random.choice(len(X_train), 500, replace=False)
            X_sample = X_train.iloc[sample_indices]
        else:
            X_sample = X_train
        
        print(f"Computing SHAP values for {len(X_sample)} samples...")
        
        # Create TreeExplainer
        explainer = shap.TreeExplainer(model)
        
        # Compute SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        # shap_values is a list for multiclass; average across classes
        if isinstance(shap_values, list):
            if len(shap_values) > 1:
                shap_values_avg = np.mean(np.abs(shap_values), axis=0)
            else:
                shap_values_avg = shap_values[0]
        else:
            shap_values_avg = shap_values
        
        plt.figure(figsize=(10, 8))
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap.summary_plot(
                shap_values, X_sample, 
                feature_names=feature_names,
                show=False,
                max_display=top_n
            )
        else:
            shap.summary_plot(
                shap_values_avg, X_sample,
                feature_names=feature_names, 
                show=False,
                max_display=top_n
            )
        
        summary_path = out_dir / "shap_summary.png"
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"SHAP summary plot saved to: {summary_path}")
        
        plt.figure(figsize=(10, 8))
        
        # Mean |SHAP| per feature, averaged across classes for multiclass
        if isinstance(shap_values, list) and len(shap_values) > 1:
            mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            mean_shap = np.abs(shap_values_avg).mean(axis=0)
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_shap
        }).sort_values('importance', ascending=True).tail(top_n)
        
        plt.barh(range(len(feature_importance)), feature_importance['importance'])
        plt.yticks(range(len(feature_importance)), feature_importance['feature'])
        plt.xlabel('Mean |SHAP Value|', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title(f'Top {top_n} Features by SHAP Importance', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        
        for i, v in enumerate(feature_importance['importance']):
            plt.text(v + 0.01 * max(feature_importance['importance']), i, 
                    f'{v:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        bar_path = out_dir / "shap_bar.png"
        plt.savefig(bar_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"SHAP bar plot saved to: {bar_path}")
        
        print(f"SHAP analysis completed. Plots saved to: {out_dir}")
        
    except Exception as e:
        print(f"Error running SHAP analysis: {str(e)}")
        raise


def explain_model_predictions(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: List[str],
    model_name: str
) -> None:
    """Dispatch SHAP analysis based on model type. Currently only supports XGBoost."""
    try:
        model_type = str(type(model)).lower()
        
        if 'xgb' in model_type or 'xgboost' in model_type:
            print(f"Running SHAP explanation for XGBoost model: {model_name}")
            
            explain_dir = config.FIGURES_DIR / f"shap_{model_name}"
            
            run_shap_xgboost(
                model=model,
                X_train=X_train, 
                feature_names=feature_names,
                out_dir=explain_dir
            )
        else:
            print(f"SHAP explanation not implemented for model type: {type(model)}")
            
    except Exception as e:
        print(f"Error in explain_model_predictions: {str(e)}")
