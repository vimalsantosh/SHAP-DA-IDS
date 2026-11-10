"""
05_model_training.py — Train and Evaluate All Baseline Models + SHAP-DA-IDS
=============================================================================
Trains and evaluates:
  1. Logistic Regression (no balancing)
  2. Random Forest (no balancing)
  3. XGBoost (no balancing)
  4. XGBoost + Class Weights (cost-sensitive)
  5. XGBoost + SMOTE Fixed (DIFT-style fixed augmentation)
  6. XGBoost + SHAP-DA-IDS (our proposed method)

All models use the same test set (no augmentation on test).
Evaluation: accuracy, macro/weighted F1, precision, recall, AUC-ROC.
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
import time
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR = r"c:\Users\vimal\OneDrive\Desktop\7_sem\CNS\paper"
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_all_data():
    """Load all preprocessed and augmented datasets."""
    data = {}
    files = [
        "X_train_sel", "X_test_sel", "X_train", "X_test",
        "y_train", "y_test",
        "X_dift_aug", "y_dift_aug",
        "X_adaptive_aug", "y_adaptive_aug",
    ]
    for f in files:
        path = os.path.join(PROCESSED_DIR, f"{f}.npy")
        if os.path.exists(path):
            data[f] = np.load(path)
    
    with open(os.path.join(PROCESSED_DIR, "metadata.pkl"), "rb") as f:
        data["meta"] = pickle.load(f)
    
    return data


def evaluate_model(model, X_test, y_test, class_names, model_name, train_time, inf_time):
    """Comprehensive model evaluation."""
    y_pred = model.predict(X_test)
    
    # Probabilities for AUC
    try:
        y_prob = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
    except:
        auc = float("nan")
    
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    macro_prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    
    report = classification_report(
        y_test, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    
    cm = confusion_matrix(y_test, y_pred)
    
    # Summary row
    summary = {
        "model": model_name,
        "accuracy": round(acc * 100, 2),
        "macro_precision": round(macro_prec * 100, 2),
        "macro_recall": round(macro_recall * 100, 2),
        "macro_f1": round(macro_f1 * 100, 2),
        "weighted_f1": round(weighted_f1 * 100, 2),
        "roc_auc": round(auc, 4) if not np.isnan(auc) else "N/A",
        "train_time_s": round(train_time, 2),
        "inf_time_ms": round(inf_time * 1000, 2),
    }
    
    # Per-class metrics
    per_class = {}
    for class_name in class_names:
        if class_name in report:
            per_class[class_name] = {
                "precision": round(report[class_name]["precision"] * 100, 2),
                "recall": round(report[class_name]["recall"] * 100, 2),
                "f1": round(report[class_name]["f1-score"] * 100, 2),
                "support": report[class_name]["support"],
            }
    
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    print(f"Accuracy:        {summary['accuracy']:.2f}%")
    print(f"Macro F1:        {summary['macro_f1']:.2f}%")
    print(f"Weighted F1:     {summary['weighted_f1']:.2f}%")
    print(f"ROC-AUC (macro): {summary['roc_auc']}")
    print(f"Train time:      {summary['train_time_s']:.2f}s")
    print(f"Inference time:  {summary['inf_time_ms']:.2f}ms")
    print("\nPer-class F1:")
    for c, m in per_class.items():
        print(f"  {c:25s}: F1={m['f1']:5.2f}% | Prec={m['precision']:5.2f}% | Rec={m['recall']:5.2f}%")
    
    return summary, per_class, cm


def train_logistic_regression(X_train, y_train, X_test, y_test, class_names):
    """Baseline: Logistic Regression (no balancing)."""
    print("\nTraining: Logistic Regression...")
    t0 = time.time()
    model = LogisticRegression(
        max_iter=200, random_state=RANDOM_SEED,
        C=1.0, solver="lbfgs", n_jobs=-1
    )
    model.fit(X_train, y_train)
    t_train = time.time() - t0
    
    t0 = time.time()
    _ = model.predict(X_test)
    t_inf = time.time() - t0
    
    return evaluate_model(model, X_test, y_test, class_names, "Logistic Regression", t_train, t_inf)


def train_random_forest(X_train, y_train, X_test, y_test, class_names, weighted=False):
    """Baseline: Random Forest."""
    name = "Random Forest" + (" (Weighted)" if weighted else "")
    print(f"\nTraining: {name}...")
    t0 = time.time()
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=3,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced" if weighted else None,
    )
    model.fit(X_train, y_train)
    t_train = time.time() - t0
    
    t0 = time.time()
    _ = model.predict(X_test)
    t_inf = time.time() - t0
    
    return evaluate_model(model, X_test, y_test, class_names, name, t_train, t_inf)


def train_xgboost(X_train, y_train, X_test, y_test, class_names, 
                  weighted=False, model_name="XGBoost"):
    """XGBoost classifier."""
    print(f"\nTraining: {model_name}...")
    
    # Compute class weights for XGBoost
    if weighted:
        class_counts = Counter(y_train)
        n_total = len(y_train)
        # sample_weight array
        weight_map = {c: n_total / (len(class_counts) * cnt)
                      for c, cnt in class_counts.items()}
        sample_weight = np.array([weight_map[y] for y in y_train])
    else:
        sample_weight = None
    
    t0 = time.time()
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
    )
    
    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)
    
    t_train = time.time() - t0
    
    t0 = time.time()
    _ = model.predict(X_test)
    t_inf = time.time() - t0
    
    return model, evaluate_model(model, X_test, y_test, class_names, model_name, t_train, t_inf)


def main():
    print("Loading data...")
    data = load_all_data()
    meta = data["meta"]
    int_to_class = meta["int_to_class"]
    class_names = [int_to_class[i] for i in sorted(int_to_class)]
    
    # Data splits
    X_train_orig = data.get("X_train_sel", data.get("X_train"))
    X_test = data.get("X_test_sel", data.get("X_test"))
    y_train_orig = data["y_train"]
    y_test = data["y_test"]
    X_dift = data.get("X_dift_aug", X_train_orig)
    y_dift = data.get("y_dift_aug", y_train_orig)
    X_adaptive = data.get("X_adaptive_aug", X_train_orig)
    y_adaptive = data.get("y_adaptive_aug", y_train_orig)
    
    print(f"Train original: {X_train_orig.shape}")
    print(f"Test: {X_test.shape}")
    print(f"DIFT aug: {X_dift.shape}")
    print(f"Adaptive aug: {X_adaptive.shape}")
    print(f"Classes: {class_names}")
    
    all_results = []
    all_per_class = {}
    all_cms = {}
    best_model = None
    best_macro_f1 = -1
    
    # ── 1. Logistic Regression (no balancing) ─────────────────────────────────
    summary, per_class, cm = train_logistic_regression(
        X_train_orig, y_train_orig, X_test, y_test, class_names
    )
    all_results.append(summary)
    all_per_class["Logistic Regression"] = per_class
    all_cms["Logistic Regression"] = cm
    
    # ── 2. Random Forest (no balancing) ───────────────────────────────────────
    summary, per_class, cm = train_random_forest(
        X_train_orig, y_train_orig, X_test, y_test, class_names
    )
    all_results.append(summary)
    all_per_class["Random Forest"] = per_class
    all_cms["Random Forest"] = cm
    
    # ── 3. XGBoost (no balancing) ─────────────────────────────────────────────
    model_xgb, (summary, per_class, cm) = train_xgboost(
        X_train_orig, y_train_orig, X_test, y_test, class_names,
        weighted=False, model_name="XGBoost"
    )
    all_results.append(summary)
    all_per_class["XGBoost"] = per_class
    all_cms["XGBoost"] = cm
    
    # ── 4. XGBoost + Class Weights ────────────────────────────────────────────
    _, (summary, per_class, cm) = train_xgboost(
        X_train_orig, y_train_orig, X_test, y_test, class_names,
        weighted=True, model_name="XGBoost + Class Weights"
    )
    all_results.append(summary)
    all_per_class["XGBoost + Class Weights"] = per_class
    all_cms["XGBoost + Class Weights"] = cm
    
    # ── 5. XGBoost + SMOTE Fixed (DIFT-style) ─────────────────────────────────
    _, (summary, per_class, cm) = train_xgboost(
        X_dift, y_dift, X_test, y_test, class_names,
        weighted=False, model_name="XGBoost + SMOTE Fixed (DIFT-style)"
    )
    all_results.append(summary)
    all_per_class["XGBoost + SMOTE Fixed (DIFT-style)"] = per_class
    all_cms["XGBoost + SMOTE Fixed (DIFT-style)"] = cm
    
    # ── 6. SHAP-DA-IDS (our proposed method) ──────────────────────────────────
    best_model, (summary, per_class, cm) = train_xgboost(
        X_adaptive, y_adaptive, X_test, y_test, class_names,
        weighted=False, model_name="SHAP-DA-IDS (Ours)"
    )
    all_results.append(summary)
    all_per_class["SHAP-DA-IDS (Ours)"] = per_class
    all_cms["SHAP-DA-IDS (Ours)"] = cm
    
    # ── Save results ───────────────────────────────────────────────────────────
    # Model comparison CSV
    comparison_df = pd.DataFrame(all_results)
    comparison_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)
    print(f"\n\nModel Comparison Summary:")
    print(comparison_df[["model", "accuracy", "macro_f1", "weighted_f1", "roc_auc"]].to_string(index=False))
    
    # Per-class metrics CSV
    per_class_rows = []
    for model_name, per_class in all_per_class.items():
        for class_name, metrics in per_class.items():
            per_class_rows.append({
                "model": model_name,
                "class": class_name,
                **metrics,
            })
    per_class_df = pd.DataFrame(per_class_rows)
    per_class_df.to_csv(os.path.join(RESULTS_DIR, "per_class_metrics.csv"), index=False)
    
    # Confusion matrix for best model (SHAP-DA-IDS)
    cm_df = pd.DataFrame(
        all_cms["SHAP-DA-IDS (Ours)"],
        index=class_names,
        columns=class_names,
    )
    cm_df.to_csv(os.path.join(RESULTS_DIR, "confusion_matrix.csv"))
    
    # Save best model
    with open(os.path.join(PROCESSED_DIR, "best_model.pkl"), "wb") as f:
        pickle.dump(best_model, f)
    
    # Save all CMs
    with open(os.path.join(PROCESSED_DIR, "all_cms.pkl"), "wb") as f:
        pickle.dump(all_cms, f)
    
    print(f"\n✓ All models trained and evaluated.")
    print(f"  Results saved to: {RESULTS_DIR}")
    
    return comparison_df, per_class_df, all_cms


if __name__ == "__main__":
    main()
