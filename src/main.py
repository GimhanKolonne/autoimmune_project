
import json
import logging
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from . import config
from .load_data import load_all_datasets
from .preprocess import preprocess_features
from .features import select_features_pipeline
from .train_models import get_base_models, train_models_with_validation
from .ensemble import create_ensemble_pipeline
from .evaluate import evaluate_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting autoimmune disease classification pipeline")
    
    np.random.seed(config.RANDOM_SEED)
    
    logger.info("Loading data...")
    X, y = load_all_datasets()
    logger.info(f"Loaded {X.shape[0]} samples, {X.shape[1]} features")
    
    logger.info("Train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_SEED, 
        stratify=y
    )
    logger.info(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    logger.info("Preprocessing features...")
    X_train_processed = preprocess_features(X_train)
    X_test_processed = preprocess_features(X_test)
    
    logger.info("Feature selection...")
    X_train_selected, X_test_selected, selected_features, feature_scores = select_features_pipeline(
        X_train_processed, y_train, X_test_processed, k=config.N_FEATURES
    )
    logger.info(f"Selected {len(selected_features)} features")
    
    logger.info("Label encoding...")
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    class_names = label_encoder.classes_
    
    logger.info("Training base models...")
    base_models = get_base_models(random_state=config.RANDOM_SEED)
    fitted_models = train_models_with_validation(base_models, X_train_selected, y_train_encoded)
    logger.info(f"Trained {len(fitted_models)} models")
    
    for model_name, model in fitted_models.items():
        model_path = config.MODELS_DIR / f"model_{model_name}.joblib"
        joblib.dump(model, model_path)
    
    logger.info("Creating stacking ensemble...")
    ensemble, _ = create_ensemble_pipeline(
        fitted_models, X_train_selected, y_train_encoded, random_state=config.RANDOM_SEED
    )
    
    ensemble_path = config.MODELS_DIR / "model_Ensemble.joblib"
    joblib.dump(ensemble, ensemble_path)
    
    logger.info("Evaluating models...")
    results = {}
    
    for name, model in fitted_models.items():
        logger.info(f"Evaluating {name}")
        metrics = evaluate_model(model, X_test_selected, y_test_encoded, class_names.tolist(), model_name=name)
        results[name] = metrics
    
    ensemble_metrics = evaluate_model(ensemble, X_test_selected, y_test_encoded, class_names.tolist(), model_name="Ensemble")
    results['Ensemble'] = ensemble_metrics
    
    logger.info("Saving results...")
    results_summary = {
        'dataset_info': {
            'n_samples': X.shape[0],
            'n_features_original': X.shape[1],
            'n_features_selected': len(selected_features),
            'classes': class_names.tolist(),
            'train_samples': len(X_train_selected),
            'test_samples': len(X_test_selected)
        },
        'model_performance': {}
    }
    
    for model_name, metrics in results.items():
        results_summary['model_performance'][model_name] = {
            'accuracy': metrics['accuracy'],
            'f1_macro': metrics['f1_macro'],
            'f1_weighted': metrics['f1_weighted'],
            'precision_macro': metrics['precision_macro'],
            'recall_macro': metrics['recall_macro']
        }
    
    results_path = config.RESULTS_DIR / "results_summary.json"
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    
    csv_data = []
    for model_name, metrics in results.items():
        csv_data.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'F1_Macro': metrics['f1_macro'],
            'F1_Weighted': metrics['f1_weighted'],
            'Precision_Macro': metrics['precision_macro'],
            'Recall_Macro': metrics['recall_macro'],
            'ROC_AUC_Macro': metrics['roc_auc_ovr_macro']
        })
    
    csv_df = pd.DataFrame(csv_data)
    csv_path = config.RESULTS_DIR / "results_summary.csv"
    csv_df.to_csv(csv_path, index=False)
    logger.info(f"CSV summary saved to {csv_path}")
    
    # SHAP analysis (only supported for XGBoost)
    try:
        best_model_name = max(results.keys(), key=lambda k: results[k]['f1_macro'])
        logger.info(f"Best model: {best_model_name} (F1-Macro: {results[best_model_name]['f1_macro']:.4f})")
        
        if 'XGBoost' in best_model_name or 'xgb' in best_model_name.lower():
            from .explain import run_shap_xgboost
            
            run_shap_xgboost(
                model=fitted_models[best_model_name],
                X_train=X_train_selected,
                feature_names=selected_features.tolist(),
                out_dir=config.FIGURES_DIR
            )
        else:
            logger.info(f"Skipping SHAP - {best_model_name} is not XGBoost")
            
    except Exception as e:
        logger.warning(f"SHAP analysis failed: {str(e)}")
    
    # Save artifacts needed by inference / frontend
    logger.info("Saving inference artifacts...")
    
    joblib.dump(ensemble, config.MODELS_DIR / "model_Ensemble.joblib")
    
    if "XGBoost" in fitted_models:
        joblib.dump(fitted_models["XGBoost"], config.MODELS_DIR / "model_XGBoost.joblib")
    
    joblib.dump(label_encoder, config.MODELS_DIR / "label_encoder.joblib")
    
    with open(config.RESULTS_DIR / "selected_features.json", 'w') as f:
        json.dump(selected_features, f, indent=2)
    
    X_test_selected.to_csv(config.RESULTS_DIR / "X_test_selected.csv", index=False)
    pd.DataFrame({'label': y_test_encoded}).to_csv(config.RESULTS_DIR / "y_test_encoded.csv", index=False)
    
    logger.info("Pipeline completed successfully!")
    
    print("\nFINAL RESULTS:")
    print(f"{'Model':<15} {'Accuracy':<10} {'F1-Macro':<10}")
    print("-" * 35)
    for model_name, metrics in results.items():
        print(f"{model_name:<15} {metrics['accuracy']:<10.4f} {metrics['f1_macro']:<10.4f}")


if __name__ == "__main__":
    main()