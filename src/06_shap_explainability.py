"""
06_shap_explainability.py — Phase 5: SHAP Explainability Analysis
==================================================================
Applies SHAP TreeExplainer to the best trained model (SHAP-DA-IDS / XGBoost)
to generate:
  - Global feature importance (mean |SHAP|)
  - Per-class SHAP rankings
  - Saves SHAP values for plotting

Part of the SHAP-DA-IDS explainability contribution.
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
import shap

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR = r"c:\Users\vimal\OneDrive\Desktop\7_sem\CNS\paper"
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    # Load best model and test data
    with open(os.path.join(PROCESSED_DIR, "best_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(PROCESSED_DIR, "metadata.pkl"), "rb") as f:
        meta = pickle.load(f)
    with open(os.path.join(PROCESSED_DIR, "selected_features.pkl"), "rb") as f:
        selected_features = pickle.load(f)
    
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test_sel.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    
    int_to_class = meta["int_to_class"]
    class_names = [int_to_class[i] for i in sorted(int_to_class)]
    feature_names = selected_features
    
    print(f"Computing SHAP values on test set (n={len(y_test):,})...")
    print(f"Features: {feature_names}")
    print(f"Classes: {class_names}")
    
    # Sample test set for SHAP (full test set computation)
    n_shap = min(1000, len(X_test))
    idx = np.random.choice(len(X_test), n_shap, replace=False)
    X_shap = X_test[idx]
    y_shap = y_test[idx]
    
    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)
    
    print(f"SHAP values shape: {np.array(shap_values).shape}")
    
    # ── Global feature importance ─────────────────────────────────────────────
    if isinstance(shap_values, list):
        shap_arr = np.array(shap_values)
        global_importance = np.mean(np.abs(shap_arr), axis=(0, 1))
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        global_importance = np.mean(np.abs(shap_values), axis=(0, 2))
    else:
        global_importance = np.mean(np.abs(shap_values), axis=0)
    
    global_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": global_importance,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    
    print("\nGlobal SHAP Feature Importance:")
    print(global_df.to_string(index=False))
    global_df.to_csv(os.path.join(RESULTS_DIR, "shap_global_importance.csv"), index=False)
    
    # ── Per-class SHAP importance ─────────────────────────────────────────────
    per_class_dfs = []
    if isinstance(shap_values, list):
        for c_idx, shap_c in enumerate(shap_values):
            class_name = int_to_class.get(c_idx, f"Class_{c_idx}")
            class_importance = np.mean(np.abs(shap_c), axis=0)
            class_df = pd.DataFrame({
                "class": class_name,
                "feature": feature_names,
                "mean_abs_shap": class_importance,
            }).sort_values("mean_abs_shap", ascending=False)
            per_class_dfs.append(class_df)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        for c_idx in range(shap_values.shape[2]):
            class_name = int_to_class.get(c_idx, f"Class_{c_idx}")
            class_importance = np.mean(np.abs(shap_values[:, :, c_idx]), axis=0)
            class_df = pd.DataFrame({
                "class": class_name,
                "feature": feature_names,
                "mean_abs_shap": class_importance,
            }).sort_values("mean_abs_shap", ascending=False)
            per_class_dfs.append(class_df)
    
    if per_class_dfs:
        per_class_shap = pd.concat(per_class_dfs, ignore_index=True)
        per_class_shap.to_csv(os.path.join(RESULTS_DIR, "shap_per_class_importance.csv"), index=False)
        
        print("\nTop features per class (by mean |SHAP|):")
        for class_name in class_names:
            top = per_class_shap[per_class_shap["class"] == class_name].head(3)
            print(f"  {class_name}: {list(top['feature'])}")
    
    # Save SHAP values for plotting
    np.save(os.path.join(PROCESSED_DIR, "shap_values.npy"), np.array(shap_values))
    np.save(os.path.join(PROCESSED_DIR, "X_shap.npy"), X_shap)
    np.save(os.path.join(PROCESSED_DIR, "y_shap.npy"), y_shap)
    
    print(f"\n✓ SHAP explainability analysis complete.")
    print(f"  SHAP values saved for plotting.")
    
    return shap_values, X_shap, global_df


if __name__ == "__main__":
    main()
