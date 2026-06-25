
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from typing import Tuple, List, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_feature_selector(k: int, random_state: int = None) -> SelectKBest:
    logger.info(f"Building feature selector for k={k} features using f_classif")
    
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    
    selector = SelectKBest(score_func=f_classif, k=k)
    
    logger.info(f"Created SelectKBest selector with f_classif scoring")
    
    return selector


def fit_transform_selector(
    selector: SelectKBest, 
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    logger.info(f"Fitting feature selector on training data with shape {X_train.shape}")
    
    if X_train.empty:
        raise ValueError("Training data is empty")
    
    if X_test.empty:
        raise ValueError("Test data is empty")
    
    if len(X_train.columns) != len(X_test.columns):
        raise ValueError(f"Train and test have different number of features: {len(X_train.columns)} vs {len(X_test.columns)}")
    
    if not all(X_train.columns == X_test.columns):
        raise ValueError("Train and test have different feature names")
    
    n_features = X_train.shape[1]
    requested_k = selector.k
    
    if requested_k > n_features:
        logger.warning(f"Requested k={requested_k} is greater than available features ({n_features}). Using all features.")
        selector.k = n_features
        actual_k = n_features
    else:
        actual_k = requested_k
    
    logger.info(f"Selecting {actual_k} out of {n_features} features")
    
    try:
        selector.fit(X_train, y_train)
        logger.info("Feature selector fitted successfully")
    except Exception as e:
        raise RuntimeError(f"Error fitting feature selector: {str(e)}")
    
    try:
        X_train_selected = selector.transform(X_train)
        X_test_selected = selector.transform(X_test)
        logger.info(f"Data transformed: train {X_train.shape} -> {X_train_selected.shape}, test {X_test.shape} -> {X_test_selected.shape}")
    except Exception as e:
        raise RuntimeError(f"Error transforming data: {str(e)}")
    
    selected_mask = selector.get_support()
    selected_feature_names = X_train.columns[selected_mask].tolist()
    
    logger.info(f"Selected {len(selected_feature_names)} features")
    logger.debug(f"Selected features: {selected_feature_names[:10]}{'...' if len(selected_feature_names) > 10 else ''}")
    
    X_train_df = pd.DataFrame(
        X_train_selected, 
        columns=selected_feature_names,
        index=X_train.index
    )
    
    X_test_df = pd.DataFrame(
        X_test_selected, 
        columns=selected_feature_names,
        index=X_test.index
    )
    
    assert X_train_df.shape[1] == len(selected_feature_names), "Training data column count mismatch"
    assert X_test_df.shape[1] == len(selected_feature_names), "Test data column count mismatch"
    assert X_train_df.shape[1] == actual_k, f"Expected {actual_k} features, got {X_train_df.shape[1]}"
    
    return X_train_df, X_test_df, selected_feature_names


def get_feature_scores(selector: SelectKBest, feature_names: List[str]) -> pd.DataFrame:
    if not hasattr(selector, 'scores_'):
        raise ValueError("Selector must be fitted before getting scores")
    
    if len(feature_names) != len(selector.scores_):
        raise ValueError(f"Number of feature names ({len(feature_names)}) doesn't match scores ({len(selector.scores_)})")
    
    scores_df = pd.DataFrame({
        'feature': feature_names,
        'score': selector.scores_,
        'selected': selector.get_support()
    })
    
    scores_df = scores_df.sort_values('score', ascending=False).reset_index(drop=True)
    
    return scores_df


def select_features_pipeline(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_test: pd.DataFrame,
    k: int,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], pd.DataFrame]:
    logger.info(f"Starting feature selection pipeline with k={k}")
    
    selector = build_feature_selector(k=k, random_state=random_state)
    
    X_train_sel, X_test_sel, selected_names = fit_transform_selector(
        selector, X_train, y_train, X_test
    )
    
    feature_scores = get_feature_scores(selector, X_train.columns.tolist())
    
    logger.info(f"Feature selection pipeline completed successfully")
    logger.info(f"Selected {len(selected_names)} features out of {X_train.shape[1]}")
    
    return X_train_sel, X_test_sel, selected_names, feature_scores


def analyze_feature_importance(feature_scores: pd.DataFrame, top_n: int = 20) -> None:
    print(f"\nFeature Selection Analysis:")
    print(f"Total features: {len(feature_scores)}")
    print(f"Selected features: {feature_scores['selected'].sum()}")
    print(f"Selection rate: {feature_scores['selected'].mean():.2%}")
    
    print(f"\nTop {top_n} features by score:")
    top_features = feature_scores.head(top_n)
    for idx, row in top_features.iterrows():
        status = "✓" if row['selected'] else "✗"
        print(f"  {status} {row['feature']}: {row['score']:.4f}")
    
    if feature_scores['selected'].sum() > 0:
        selected_scores = feature_scores[feature_scores['selected']]['score']
        print(f"\nSelected feature scores:")
        print(f"  Mean: {selected_scores.mean():.4f}")
        print(f"  Std: {selected_scores.std():.4f}")
        print(f"  Min: {selected_scores.min():.4f}")
        print(f"  Max: {selected_scores.max():.4f}")


if __name__ == "__main__":
    logger.info("Testing feature selection functions...")
    
    np.random.seed(42)
    n_samples, n_features = 200, 100
    
    X_data = np.random.randn(n_samples, n_features)
    
    y_data = np.random.choice(['A', 'B', 'C', 'D'], size=n_samples)
    
    for i in range(10):
        if i % 4 == 0:
            X_data[y_data == 'A', i] += 2
        elif i % 4 == 1:
            X_data[y_data == 'B', i] += 2
        elif i % 4 == 2:
            X_data[y_data == 'C', i] += 2
        else:
            X_data[y_data == 'D', i] += 2
    
    feature_names = [f"Feature_{i:03d}" for i in range(n_features)]
    X = pd.DataFrame(X_data, columns=feature_names)
    y = pd.Series(y_data)
    
    split_idx = int(0.8 * n_samples)
    X_train = X.iloc[:split_idx].copy()
    y_train = y.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    
    print(f"Created test data: train {X_train.shape}, test {X_test.shape}")
    
    k_features = 20
    X_train_sel, X_test_sel, selected_names, scores_df = select_features_pipeline(
        X_train, y_train, X_test, k=k_features, random_state=42
    )
    
    print(f"Feature selection results:")
    print(f"  Selected features shape: train {X_train_sel.shape}, test {X_test_sel.shape}")
    print(f"  Number of selected features: {len(selected_names)}")
    
    analyze_feature_importance(scores_df, top_n=15)
    
    print(f"\nTesting k > number of features...")
    X_small = X_train.iloc[:, :5]
    X_small_test = X_test.iloc[:, :5]
    
    X_sel, X_test_sel, names = fit_transform_selector(
        build_feature_selector(k=10),
        X_small, y_train, X_small_test
    )
    
    print(f"Requested 10 features from 5 available: got {X_sel.shape[1]} features")
    
    print("Feature selection testing completed successfully!")