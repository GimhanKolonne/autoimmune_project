from typing import Dict, Any, Union
import time
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def get_base_models(random_state: int = 42) -> Dict[str, Any]:
    
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        ),
        
        'SVM': SVC(
            kernel='rbf',
            probability=True,
            class_weight='balanced',
            random_state=random_state
        ),
        
        'LogisticRegression': LogisticRegression(
            solver='lbfgs',
            max_iter=5000,
            class_weight='balanced',
            random_state=random_state
        )
    }
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBClassifier(
            objective='multi:softprob',
            eval_metric='mlogloss',
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            verbosity=0
        )
    
    print(f"Configured {len(models)} base models with random_state={random_state}")
    return models


def train_models_with_validation(
    models: Dict[str, Any], 
    X_train: Union[pd.DataFrame, np.ndarray], 
    y_train: Union[pd.Series, np.ndarray],
    validate: bool = True
) -> Dict[str, Any]:
    
    if validate:
        if X_train.shape[0] == 0:
            raise ValueError("Training data cannot be empty")
        
        if len(X_train) != len(y_train):
            raise ValueError(f"X_train and y_train length mismatch: {len(X_train)} vs {len(y_train)}")
    
    y_train_array = np.array(y_train)
    
    print(f"Training {len(models)} models on data with shape {X_train.shape}")
    
    unique_classes, counts = np.unique(y_train_array, return_counts=True)
    class_dist = dict(zip(unique_classes, counts))
    print(f"Target classes: {sorted(unique_classes)}")
    print(f"Class distribution: {class_dist}")
    
    fitted_models = {}
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        
        for name, model in models.items():
            print(f"Training {name}...", end=" ")
            start_time = time.time()
            
            try:
                fitted_model = model.fit(X_train, y_train_array)
                fitted_models[name] = fitted_model
                
                train_time = time.time() - start_time
                print(f"completed in {train_time:.2f}s")
                
            except Exception as e:
                print(f"FAILED - {str(e)}")
                continue
    
    print(f"Successfully trained {len(fitted_models)}/{len(models)} models")
    return fitted_models


if __name__ == "__main__":
    print("Testing model training functions...")
    
    np.random.seed(42)
    n_samples, n_features = 500, 50
    
    X_data = np.random.randn(n_samples, n_features)
    y_data = np.random.choice(['Healthy', 'IBD', 'MS', 'RA'], size=n_samples)
    
    feature_names = [f"Feature_{i}" for i in range(n_features)]
    X_train = pd.DataFrame(X_data, columns=feature_names)
    y_train = pd.Series(y_data, name='label')
    
    print(f"Created test data: {X_train.shape} features, {len(y_train)} labels")
    
    models = get_base_models(random_state=42)
    
    fitted_models = train_models_with_validation(models, X_train, y_train)
    
    print(f"\nTesting predictions...")
    for name, model in fitted_models.items():
        try:
            pred = model.predict(X_train[:5])
            pred_proba = model.predict_proba(X_train[:5])
            print(f"{name}: predictions shape {pred.shape}, probabilities shape {pred_proba.shape}")
        except Exception as e:
            print(f"{name}: prediction test failed - {str(e)}")
    
    print(f"\nModel training test completed successfully!")
    print(f"Trained models: {list(fitted_models.keys())}")