"""
02_preprocessing.py — Data Preprocessing Pipeline
===================================================
Performs:
  1. Missing value imputation (training data only)
  2. Dropping non-informative identifier columns
  3. Label encoding of categorical features
  4. Target variable encoding
  5. Standard scaling of numeric features
  6. Stratified train/test split (80/20)
  7. Saves preprocessed data

Part of the SHAP-DA-IDS pipeline.
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
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR = r"c:\Users\vimal\OneDrive\Desktop\7_sem\CNS\paper"
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "DS2OS_traffic_traces.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Columns to drop (identifiers, raw addresses — do not generalise)
DROP_COLS = [
    "sourceid", "sourceAddress", "destinationServiceAddress",
    "accessedNodeAddress", "timestamp"
]

# Target column name in DS2OS
TARGET_COL = "normality"


def load_and_preprocess():
    """Full preprocessing pipeline."""
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"Original shape: {df.shape}")

    # ── Step 1: Drop identifier columns ──────────────────────────────────────
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"After dropping identifiers: {df.shape}")

    # ── Step 2: Encode target variable ────────────────────────────────────────
    # Map class names to integers deterministically
    class_names = sorted(df[TARGET_COL].unique())
    class_to_int = {name: i for i, name in enumerate(class_names)}
    int_to_class = {i: name for name, i in class_to_int.items()}
    
    df["label"] = df[TARGET_COL].map(class_to_int)
    df = df.drop(columns=[TARGET_COL])
    
    print(f"\nClass mapping:")
    for name, idx in class_to_int.items():
        count = (df["label"] == idx).sum()
        print(f"  {idx}: {name} ({count:,} samples)")

    # ── Step 3: Identify feature types ────────────────────────────────────────
    X = df.drop(columns=["label"])
    y = df["label"].values
    
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    print(f"\nNumeric features: {numeric_cols}")
    print(f"Categorical features: {categorical_cols}")

    # ── Step 4: Train/test split (80/20, stratified) ──────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"\nTrain set: {X_train.shape}")
    print(f"Test set:  {X_test.shape}")

    # ── Step 5: Impute missing values (fit on train only) ─────────────────────
    # Numeric: impute with median
    num_imputer = SimpleImputer(strategy="median")
    if numeric_cols:
        X_train[numeric_cols] = num_imputer.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = num_imputer.transform(X_test[numeric_cols])

    # Categorical: impute with most frequent
    cat_imputer = SimpleImputer(strategy="most_frequent")
    if categorical_cols:
        X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

    # ── Step 6: Encode categorical features (fit on train only) ───────────────
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        # Handle unseen labels in test set
        X_test[col] = X_test[col].astype(str).map(
            lambda x, le=le: le.transform([x])[0]
            if x in le.classes_ else -1
        )
        label_encoders[col] = le
    
    # Convert to numpy
    X_train_arr = X_train.values.astype(np.float32)
    X_test_arr = X_test.values.astype(np.float32)
    feature_names = X_train.columns.tolist()

    # ── Step 7: Standard scaling (fit on train only) ──────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_arr)
    X_test_scaled = scaler.transform(X_test_arr)

    # ── Step 8: Save preprocessed data ────────────────────────────────────────
    np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train_scaled)
    np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test_scaled)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)
    
    # Save metadata
    metadata = {
        "feature_names": feature_names,
        "class_to_int": class_to_int,
        "int_to_class": int_to_class,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "n_classes": len(class_to_int),
        "n_features": len(feature_names),
        "train_size": len(y_train),
        "test_size": len(y_test),
    }
    
    with open(os.path.join(PROCESSED_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)
    with open(os.path.join(PROCESSED_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(PROCESSED_DIR, "encoders.pkl"), "wb") as f:
        pickle.dump(label_encoders, f)
    
    print(f"\nPreprocessed data saved to: {PROCESSED_DIR}")
    print(f"Feature names: {feature_names}")
    print(f"\nClass counts in y_train:")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {int_to_class[u]}: {c:,}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, metadata


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, meta = load_and_preprocess()
    print(f"\n✓ Preprocessing complete.")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  Features: {meta['feature_names']}")
