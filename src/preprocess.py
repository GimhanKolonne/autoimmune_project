import pandas as pd
import numpy as np


def preprocess_features(X: pd.DataFrame) -> pd.DataFrame:
    
    X_processed = X.copy()
    print(f"Preprocessing: input shape {X.shape}")
    
    for col in X_processed.columns:
        X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce')
    
    nan_count = X_processed.isna().sum().sum()
    if nan_count > 0:
        print(f"  Replacing {nan_count} NaN values with 0")
        X_processed = X_processed.fillna(0)
    
    negative_count = (X_processed < 0).sum().sum()
    if negative_count > 0:
        print(f"  Clipping {negative_count} negative values to 0")
        X_processed = X_processed.clip(lower=0)
    
    #pseudocount to avoid log(0)
    X_processed = X_processed + 1e-9
    
    #relative abundance normalisation
    row_sums = X_processed.sum(axis=1)
    X_processed = X_processed.div(row_sums, axis=0)
    
    X_processed = np.log1p(X_processed)
    
    print(f"Preprocessing: output shape {X_processed.shape}")
    
    return X_processed


if __name__ == "__main__":
    np.random.seed(42)
    test_data = pd.DataFrame(
        np.random.randn(100, 20),
        columns=[f"feature_{i}" for i in range(20)]
    )
    
    test_data.iloc[0, 0] = -5
    test_data.iloc[1, 1] = np.nan
    
    print("Testing preprocessing function...")
    processed = preprocess_features(test_data)
    
    print(f"Original shape: {test_data.shape}")
    print(f"Processed shape: {processed.shape}")
    print(f"Row sums close to log(2): {np.allclose(processed.sum(axis=1), np.log(2), atol=1e-6)}")
    print("Preprocessing test completed successfully!")