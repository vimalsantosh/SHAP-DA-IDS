"""
03_shap_feature_selection.py — Phase 1: SHAP-based Feature Selection
======================================================================
Trains a preliminary Random Forest classifier on the imbalanced training
data, computes SHAP values, ranks features by mean absolute SHAP value,
and selects the top-k features (k determined by 95% cumulative importance
threshold).

Part of the SHAP-DA-IDS novelty:
  Phase 1 — SHAP Feature Selection reduces input noise, potentially
  improving minority-class signal-to-noise ratio.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import pandas as pd
import pickle
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, f1_score

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR = r"c:\Users\vimal\OneDrive\Desktop\7_sem\CNS\paper"
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_data():
    """Load preprocessed data and metadata."""
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    with open(os.path.join(PROCESSED_DIR, "metadata.pkl"), "rb") as f:
        meta = pickle.load(f)
    return X_train, X_test, y_train, y_test, meta


def train_preliminary_classifier(X_train, y_train, meta):
    """Train a preliminary Random Forest for SHAP computation."""
    print("Training preliminary Random Forest classifier...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=5,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)
    print("  Training complete.")
    
    # Cross-validated predictions for difficulty estimation
    print("  Computing cross-validated predictions...")
    # Use subset for speed (sample up to 50k training examples)
    n_cv = min(50000, len(y_train))
    idx = np.random.choice(len(y_train), n_cv, replace=False)
    y_cv_pred = cross_val_predict(
        RandomForestClassifier(n_estimators=50, max_depth=10, random_state=RANDOM_SEED,
                               n_jobs=-1, class_weight="balanced"),
        X_train[idx],
        y_train[idx],
        cv=3,
    )
    
    report = classification_report(
        y_train[idx], y_cv_pred,
        target_names=[meta["int_to_class"][i] for i in sorted(meta["int_to_class"])],
        output_dict=True,
        zero_division=0,
    )
    
    print("\nPreliminary classifier per-class F1 (CV, subset):")
    per_class_f1 = {}
    for class_name in meta["class_to_int"]:
        f1 = report.get(class_name, {}).get("f1-score", 0.0)
        per_class_f1[class_name] = f1
        print(f"  {class_name}: F1={f1:.4f}")
    
    return rf, per_class_f1


def compute_shap_importance(rf, X_train, feature_names, n_sample=1000):
    """Compute SHAP feature importance using a sample of training data."""
    print(f"\nComputing SHAP values (sample={n_sample})...")
    
    # Sample for SHAP computation (full SHAP on 280k rows is too slow)
    idx = np.random.choice(len(X_train), min(n_sample, len(X_train)), replace=False)
    X_sample = X_train[idx]
    
    explainer = shap.TreeExplainer(rf)
    shap_values_all = explainer.shap_values(X_sample)
    
    # shap_values_all: list of arrays [class_0, class_1, ...] each shape (n_sample, n_features)
    # Compute mean |SHAP| across all classes and samples
    if isinstance(shap_values_all, list):
        shap_arr = np.array(shap_values_all)
        mean_abs_shap = np.mean(np.abs(shap_arr), axis=(0, 1))
    elif isinstance(shap_values_all, np.ndarray) and shap_values_all.ndim == 3:
        mean_abs_shap = np.mean(np.abs(shap_values_all), axis=(0, 2))
    else:
        mean_abs_shap = np.mean(np.abs(shap_values_all), axis=0)
    
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    
    importance_df["cumulative_importance"] = (
        importance_df["mean_abs_shap"].cumsum() / importance_df["mean_abs_shap"].sum()
    )
    
    print("\nFeature Importance (SHAP):")
    print(importance_df.to_string(index=False))
    
    return importance_df, shap_values_all, X_sample


def select_features(importance_df, threshold=0.95):
    """Select features until cumulative importance >= threshold."""
    selected = importance_df[importance_df["cumulative_importance"] <= threshold]
    
    # Always include at least the top feature
    if len(selected) == 0:
        selected = importance_df.head(1)
    
    # Include one more feature to exceed threshold
    n_selected = min(len(selected) + 1, len(importance_df))
    selected = importance_df.head(n_selected)
    
    selected_features = selected["feature"].tolist()
    print(f"\nSelected {len(selected_features)} features (cumulative importance >= {threshold*100:.0f}%)")
    print(f"  Selected: {selected_features}")
    return selected_features


def save_results(importance_df, selected_features, per_class_f1, shap_values, X_sample, meta):
    """Save SHAP results and figures."""
    # Save importance table
    importance_df.to_csv(
        os.path.join(BASE_DIR, "results", "shap_feature_importance.csv"), index=False
    )
    
    # Save selected features and per-class F1
    with open(os.path.join(PROCESSED_DIR, "selected_features.pkl"), "wb") as f:
        pickle.dump(selected_features, f)
    with open(os.path.join(PROCESSED_DIR, "per_class_f1_preliminary.pkl"), "wb") as f:
        pickle.dump(per_class_f1, f)
    
    print(f"\nSaved SHAP importance and selected features.")


def main():
    X_train, X_test, y_train, y_test, meta = load_data()
    feature_names = meta["feature_names"]
    
    # Train preliminary classifier
    rf, per_class_f1 = train_preliminary_classifier(X_train, y_train, meta)
    
    # Compute SHAP importance
    importance_df, shap_values, X_sample = compute_shap_importance(
        rf, X_train, feature_names, n_sample=1000
    )
    
    # Select features
    selected_features = select_features(importance_df, threshold=0.95)
    
    # Save results
    save_results(importance_df, selected_features, per_class_f1, shap_values, X_sample, meta)
    
    print(f"\n✓ SHAP feature selection complete.")
    print(f"  Features selected: {len(selected_features)} of {len(feature_names)}")
    return importance_df, selected_features, per_class_f1


if __name__ == "__main__":
    main()
