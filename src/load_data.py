import pandas as pd
from pathlib import Path
from typing import Tuple

from .config import (
    HEALTHY_DATA_PATH, IBD_DATA_PATH, MS_DATA_PATH, RA_DATA_PATH
)


def load_single_dataset(file_path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    
    unnamed_cols = [col for col in df.columns if col.startswith('Unnamed')]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    
    df['label'] = label
    print(f"{label}: {df.shape[0]} samples, {df.shape[1] - 1} features")
    
    return df


def load_all_datasets() -> Tuple[pd.DataFrame, pd.Series]:
    datasets = [
        (HEALTHY_DATA_PATH, 'Healthy'),
        (IBD_DATA_PATH, 'IBD'),
        (MS_DATA_PATH, 'MS'),
        (RA_DATA_PATH, 'RA')
    ]
    
    dataframes = []
    
    for file_path, label in datasets:
        df = load_single_dataset(file_path, label)
        dataframes.append(df)
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    X = combined_df.drop('label', axis=1)
    y = combined_df['label']
    
    return X, y


if __name__ == "__main__":
    X, y = load_all_datasets()
    print(f"\nCombined dataset:")
    print(f"Total samples: {len(X)}")
    print(f"Total features: {X.shape[1]}")
    print(f"Label distribution:\n{y.value_counts()}")