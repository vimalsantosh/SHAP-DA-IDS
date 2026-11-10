"""
01_data_loading.py — Data Loading and Exploration
==================================================
Loads the DS2OS dataset, performs initial exploration,
and saves summary statistics.

Part of the SHAP-DA-IDS pipeline for DIFT extension.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import pandas as pd
import numpy as np
import os
import sys

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR = r"c:\Users\vimal\OneDrive\Desktop\7_sem\CNS\paper"
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "DS2OS_traffic_traces.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_dataset(path: str) -> pd.DataFrame:
    """Load the DS2OS dataset from CSV."""
    print(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    print(f"Loaded shape: {df.shape}")
    return df


def explore_dataset(df: pd.DataFrame) -> dict:
    """Explore dataset and return summary statistics."""
    print("\n" + "="*60)
    print("DATASET EXPLORATION")
    print("="*60)
    
    # Basic info
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Target column detection
    target_col = None
    for col in ["normality", "label", "class", "attack_type", "type"]:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        # Try last column
        target_col = df.columns[-1]
        print(f"Warning: target column guessed as '{target_col}'")
    
    print(f"\nTarget column: {target_col}")
    
    # Class distribution
    class_dist = df[target_col].value_counts().reset_index()
    class_dist.columns = ["class", "count"]
    class_dist["percentage"] = (class_dist["count"] / len(df) * 100).round(4)
    class_dist["imbalance_ratio"] = (class_dist["count"].max() / class_dist["count"]).round(1)
    
    print("\nClass Distribution:")
    print(class_dist.to_string(index=False))
    
    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    missing_df = missing_df[missing_df["missing_count"] > 0]
    
    print(f"\nMissing Values (columns with any NaN):")
    if len(missing_df) > 0:
        print(missing_df.to_string())
    else:
        print("  None")
    
    # Data types
    print(f"\nData Types:")
    print(df.dtypes.to_string())
    
    # Feature types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    print(f"\nNumeric features ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")
    
    # Save class distribution
    class_dist_path = os.path.join(OUTPUT_DIR, "class_distribution.csv")
    class_dist.to_csv(class_dist_path, index=False)
    print(f"\nSaved class distribution to: {class_dist_path}")
    
    return {
        "target_col": target_col,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "class_dist": class_dist,
        "missing": missing_df,
        "n_classes": class_dist["class"].nunique(),
        "n_samples": len(df),
        "n_features": df.shape[1] - 1,
    }


if __name__ == "__main__":
    df = load_dataset(DATA_PATH)
    info = explore_dataset(df)
    print("\n✓ Data loading complete.")
    print(f"  Samples: {info['n_samples']:,}")
    print(f"  Features: {info['n_features']}")
    print(f"  Classes: {info['n_classes']}")
    print(f"  Target column: {info['target_col']}")
