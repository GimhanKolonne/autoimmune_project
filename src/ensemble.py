
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Union
import time
import warnings

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.base import clone

try:
    from . import config
except ImportError:
    config = None


def build_stacking_ensemble(base_models: Dict[str, Any], random_state: int = 42) -> StackingClassifier:
    if not base_models:
        raise ValueError("base_models cannot be empty")
    
    estimators = [(name, clone(model)) for name, model in base_models.items()]
    
    final_estimator = LogisticRegression(
        solver='lbfgs',
        max_iter=5000,
        class_weight='balanced',    
        random_state=random_state
    )
    
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        stack_method='predict_proba',
        passthrough=False,
        cv=5,
        n_jobs=-1,
        verbose=0
    )
    
    print(f"Built stacking ensemble with {len(estimators)} base models:")
    for name, _ in estimators:
        print(f"  - {name}")
    print(f"Meta-learner: LogisticRegression with 5-fold CV")
    
    return stacking_clf


def fit_ensemble(ensemble: StackingClassifier, X_train: Union[pd.DataFrame, np.ndarray], y_train: Union[pd.Series, np.ndarray]) -> StackingClassifier:
    print(f"Fitting stacking ensemble on data with shape {X_train.shape}")
    print(f"Target classes: {sorted(np.unique(y_train))}")
    
    if X_train.shape[0] == 0 or len(y_train) == 0:
        raise ValueError("Training data cannot be empty")
    
    if len(X_train) != len(y_train):
        raise ValueError(f"Feature and label length mismatch: {len(X_train)} vs {len(y_train)}")
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        
        start_time = time.time()
        
        try:
            ensemble.fit(X_train, y_train)
            
            fit_time = time.time() - start_time
            print(f"Ensemble fitted successfully in {fit_time:.2f}s")
            
        except Exception as e:
            print(f"Ensemble fitting failed: {str(e)}")
            raise
    
    return ensemble


def evaluate_base_vs_ensemble(
    base_models: Dict[str, Any], 
    ensemble: StackingClassifier,
    X_train: Union[pd.DataFrame, np.ndarray], 
    y_train: Union[pd.Series, np.ndarray],
    cv: int = 5
) -> pd.DataFrame:
    print(f"Evaluating models using {cv}-fold cross-validation...")
    
    results = []
    
    for name, model in base_models.items():
        print(f"Evaluating {name}...", end=" ")
        
        try:
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            mean_score = scores.mean()
            std_score = scores.std()
            results.append({
                'Model': name,
                'Type': 'Base',
                'CV_Mean': float(mean_score),
                'CV_Std': float(std_score),
                'CV_Scores': scores.tolist()
            })
            print(f"accuracy: {mean_score:.4f} ± {std_score:.4f}")
            
        except Exception as e:
            print(f"failed - {str(e)}")
            continue
    
    print(f"Evaluating Ensemble...", end=" ")
    try:
        scores = cross_val_score(ensemble, X_train, y_train, cv=cv, scoring='accuracy')
        mean_acc = scores.mean()
        std_acc = scores.std()
        results.append({
            'Model': 'Stacking_Ensemble',
            'Type': 'Ensemble',
            'CV_Mean': float(mean_acc),
            'CV_Std': float(std_acc),
            'CV_Scores': scores.tolist()
        })
        print(f"accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        
    except Exception as e:
        print(f"failed - {str(e)}")
    
    comparison_df = pd.DataFrame(results)
    if len(comparison_df) > 0:
        comparison_df = comparison_df.sort_values('CV_Mean', ascending=False).reset_index(drop=True)
    
    return comparison_df


def get_ensemble_predictions(
    ensemble: StackingClassifier, 
    X_test: Union[pd.DataFrame, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    if not hasattr(ensemble, 'classes_'):
        raise ValueError("Ensemble must be fitted before making predictions")
    
    print(f"Generating ensemble predictions for {X_test.shape[0]} samples...")
    
    try:
        predictions = ensemble.predict(X_test)
        probabilities = ensemble.predict_proba(X_test)
        
        print(f"Generated predictions: {predictions.shape}")
        print(f"Generated probabilities: {probabilities.shape}")
        print(f"Predicted classes: {np.unique(predictions)}")
        
        return predictions, probabilities
        
    except Exception as e:
        print(f"Prediction generation failed: {str(e)}")
        raise


def analyze_ensemble_structure(ensemble: StackingClassifier) -> Dict[str, Any]:
    if not hasattr(ensemble, 'classes_'):
        raise ValueError("Ensemble must be fitted before analysis")
    
    analysis = {
        'n_base_models': len(ensemble.estimators_),
        'base_model_names': [name for name, _ in ensemble.estimators],
        'base_model_types': [type(model).__name__ for model in ensemble.estimators_],
        'final_estimator_type': type(ensemble.final_estimator_).__name__,
        'n_classes': len(ensemble.classes_),
        'classes': ensemble.classes_.tolist(),
        'stack_method': ensemble.stack_method,
        'cv_folds': ensemble.cv,
        'passthrough': ensemble.passthrough
    }
    
    return analysis


def create_ensemble_pipeline(
    base_models: Dict[str, Any], 
    X_train: Union[pd.DataFrame, np.ndarray], 
    y_train: Union[pd.Series, np.ndarray],
    random_state: int = 42,
    evaluate: bool = True
) -> Tuple[StackingClassifier, pd.DataFrame]:
    print("Starting ensemble pipeline...")
    
    ensemble = build_stacking_ensemble(base_models, random_state=random_state)
    
    fitted_ensemble = fit_ensemble(ensemble, X_train, y_train)
    
    comparison_results = pd.DataFrame()
    if evaluate:
        comparison_results = evaluate_base_vs_ensemble(
            base_models, fitted_ensemble, X_train, y_train
        )
        
        # Save comparison results to CSV if config is available
        if config is not None and not comparison_results.empty:
            try:
                config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                csv_path = config.RESULTS_DIR / "cv_comparison.csv"
                comparison_results.to_csv(csv_path, index=False)
                print(f"Cross-validation comparison saved to: {csv_path}")
            except Exception as e:
                print(f"Warning: Could not save CV comparison: {str(e)}")
    
    structure = analyze_ensemble_structure(fitted_ensemble)
    print(f"\nEnsemble structure:")
    print(f"  Base models: {structure['n_base_models']}")
    print(f"  Classes: {structure['n_classes']} {structure['classes']}")
    print(f"  Meta-learner: {structure['final_estimator_type']}")
    
    print("Ensemble pipeline completed successfully!")
    
    return fitted_ensemble, comparison_results


if __name__ == "__main__":
    print("Testing ensemble functions...")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression as LR
    from xgboost import XGBClassifier
    
    np.random.seed(42)
    n_samples, n_features = 400, 30
    
    X_data = np.random.randn(n_samples, n_features)
    
    y_data = np.random.choice(['Healthy', 'IBD', 'MS', 'RA'], size=n_samples)
    
    for i, class_label in enumerate(['Healthy', 'IBD', 'MS', 'RA']):
        mask = y_data == class_label
        X_data[mask, i*3:(i+1)*3] += 2
    
    feature_names = [f"Feature_{i}" for i in range(n_features)]
    X_train = pd.DataFrame(X_data, columns=feature_names)
    y_train = pd.Series(y_data, name='label')
    
    print(f"Created test data: {X_train.shape} features, {len(y_train)} labels")
    print(f"Class distribution: {dict(y_train.value_counts())}")
    
    base_models = {
        'RF': RandomForestClassifier(n_estimators=50, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'LR': LR(max_iter=1000, random_state=42),
        'XGB': XGBClassifier(n_estimators=50, random_state=42, verbosity=0)
    }
    
    print(f"\nCreated {len(base_models)} base models")
    
    print("Pre-training base models...")
    for name, model in base_models.items():
        model.fit(X_train, y_train)
        print(f"  {name} trained")
    
    print(f"\nTesting ensemble pipeline...")
    ensemble, comparison = create_ensemble_pipeline(
        base_models, X_train, y_train, random_state=42, evaluate=True
    )
    
    if len(comparison) > 0:
        print(f"\nModel comparison results:")
        print(comparison[['Model', 'Type', 'CV_Mean', 'CV_Std']].to_string(index=False))
    
    print(f"\nTesting ensemble predictions...")
    test_predictions, test_probabilities = get_ensemble_predictions(ensemble, X_train[:10])
    print(f"Sample predictions: {test_predictions[:5]}")
    print(f"Sample probability shape: {test_probabilities[:2].shape}")
    
    print("Ensemble testing completed successfully!")