"""
04_difficulty_adaptive_augmentation.py — Phase 2 & 3: Difficulty-Adaptive Augmentation
=========================================================================================
Implements the SHAP-DA-IDS core novelty:
  - Computes per-class difficulty scores from preliminary classifier
  - Adaptively sets SMOTE augmentation targets proportional to difficulty
  - Applies SMOTE with adaptive targets to generate balanced training data
  - Avoids data leakage (all augmentation on training data only)

This is contrasted with the original DIFT approach which uses a fixed
average-count augmentation target for all minority classes.
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
from collections import Counter
from imblearn.over_sampling import SMOTE

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR = r"c:\Users\vimal\OneDrive\Desktop\7_sem\CNS\paper"
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def compute_dift_fixed_strategy(y_train, int_to_class):
    """
    Original DIFT strategy: set augmentation target as the average count
    of all minority classes.
    """
    class_counts = Counter(y_train)
    # Identify majority class (class 0 = Normal in DS2OS after encoding)
    majority_class = max(class_counts, key=class_counts.get)
    minority_classes = [c for c in class_counts if c != majority_class]
    
    # Average minority class count
    avg_minority = int(np.mean([class_counts[c] for c in minority_classes]))
    
    sampling_strategy = {
        c: max(class_counts[c], avg_minority)
        for c in minority_classes
    }
    
    print("\nDIFT Fixed Strategy (average-count target):")
    print(f"  Average minority count: {avg_minority:,}")
    for c, target in sampling_strategy.items():
        print(f"  {int_to_class[c]}: {class_counts[c]:,} → {target:,}")
    
    return sampling_strategy


def compute_adaptive_strategy(y_train, per_class_f1, class_to_int, int_to_class, alpha=0.3):
    """
    SHAP-DA-IDS Strategy: set augmentation target proportional to per-class
    detection difficulty (1 - F1 from preliminary classifier).
    
    Args:
        alpha: Controls maximum augmentation as fraction of majority class count
               (default=0.3 means minority classes can reach up to 30% of Normal)
    """
    class_counts = Counter(y_train)
    majority_class = max(class_counts, key=class_counts.get)
    n_majority = class_counts[majority_class]
    minority_classes = [c for c in class_counts if c != majority_class]
    
    # Compute difficulty scores
    difficulty_scores = {}
    for class_name, f1 in per_class_f1.items():
        if class_name in class_to_int:
            c_idx = class_to_int[class_name]
            if c_idx in minority_classes:
                difficulty_scores[c_idx] = 1.0 - f1
    
    # Normalize difficulty scores (sum to 1 among minority classes)
    total_diff = sum(difficulty_scores.values())
    if total_diff == 0:
        # Fallback: uniform difficulty
        for c in minority_classes:
            difficulty_scores[c] = 1.0 / len(minority_classes)
        total_diff = 1.0
    
    norm_difficulty = {c: d / total_diff for c, d in difficulty_scores.items()}
    
    # Compute adaptive targets
    # Base target = current count * 2 (minimum doubling)
    # + difficulty-proportional augmentation up to alpha * n_majority
    max_additional = int(alpha * n_majority)
    
    sampling_strategy = {}
    for c in minority_classes:
        base = class_counts[c] * 2  # minimum: double existing samples
        difficulty_bonus = int(norm_difficulty.get(c, 0) * max_additional)
        target = max(base, class_counts[c] + difficulty_bonus)
        target = min(target, n_majority)  # cap at majority count
        # SMOTE requires at least k+1 samples; k_neighbors=5 by default
        target = max(target, class_counts[c])  # never reduce
        sampling_strategy[c] = target
    
    print("\nSHAP-DA-IDS Adaptive Strategy:")
    print(f"  Majority class ({int_to_class[majority_class]}): {n_majority:,}")
    print(f"  alpha (max augmentation fraction): {alpha}")
    for c, target in sampling_strategy.items():
        f1 = per_class_f1.get(int_to_class[c], 0)
        diff = difficulty_scores.get(c, 0)
        print(f"  {int_to_class[c]}: {class_counts[c]:,} → {target:,} "
              f"(F1={f1:.3f}, difficulty={diff:.3f})")
    
    return sampling_strategy


def apply_augmentation(X_train, y_train, sampling_strategy, strategy_name):
    """Apply SMOTE with the given sampling strategy."""
    print(f"\nApplying SMOTE ({strategy_name})...")
    print(f"  Input class distribution: {Counter(y_train)}")
    
    # SMOTE requires k_neighbors < min_class_size - 1
    class_counts = Counter(y_train)
    min_class_size = min(class_counts.values())
    k_neighbors = min(5, min_class_size - 1)
    k_neighbors = max(1, k_neighbors)
    
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=RANDOM_SEED,
    )
    
    X_aug, y_aug = smote.fit_resample(X_train, y_train)
    print(f"  Output class distribution: {Counter(y_aug)}")
    print(f"  Total samples: {len(y_aug):,}")
    
    return X_aug, y_aug


def main():
    # Load preprocessed data
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    
    with open(os.path.join(PROCESSED_DIR, "metadata.pkl"), "rb") as f:
        meta = pickle.load(f)
    with open(os.path.join(PROCESSED_DIR, "per_class_f1_preliminary.pkl"), "rb") as f:
        per_class_f1 = pickle.load(f)
    with open(os.path.join(PROCESSED_DIR, "selected_features.pkl"), "rb") as f:
        selected_features = pickle.load(f)
    
    class_to_int = meta["class_to_int"]
    int_to_class = meta["int_to_class"]
    feature_names = meta["feature_names"]
    
    # Get indices of selected features
    selected_indices = [feature_names.index(f) for f in selected_features if f in feature_names]
    if not selected_indices:
        selected_indices = list(range(len(feature_names)))
    
    # Apply feature selection to training and test data
    X_train_sel = X_train[:, selected_indices]
    X_test_sel = X_test[:, selected_indices]
    
    print(f"Feature selection: {X_train.shape[1]} → {X_train_sel.shape[1]} features")
    
    # ── Strategy 1: DIFT-style fixed augmentation ─────────────────────────────
    dift_strategy = compute_dift_fixed_strategy(y_train, int_to_class)
    X_dift, y_dift = apply_augmentation(X_train_sel, y_train, dift_strategy, "DIFT-fixed")
    
    # ── Strategy 2: SHAP-DA-IDS adaptive augmentation ─────────────────────────
    adaptive_strategy = compute_adaptive_strategy(
        y_train, per_class_f1, class_to_int, int_to_class, alpha=0.3
    )
    X_adaptive, y_adaptive = apply_augmentation(
        X_train_sel, y_train, adaptive_strategy, "SHAP-DA-IDS-adaptive"
    )
    
    # Save augmented datasets
    np.save(os.path.join(PROCESSED_DIR, "X_train_sel.npy"), X_train_sel)
    np.save(os.path.join(PROCESSED_DIR, "X_test_sel.npy"), X_test_sel)
    np.save(os.path.join(PROCESSED_DIR, "X_dift_aug.npy"), X_dift)
    np.save(os.path.join(PROCESSED_DIR, "y_dift_aug.npy"), y_dift)
    np.save(os.path.join(PROCESSED_DIR, "X_adaptive_aug.npy"), X_adaptive)
    np.save(os.path.join(PROCESSED_DIR, "y_adaptive_aug.npy"), y_adaptive)
    
    # Save strategy summaries for reporting
    strategy_data = []
    class_counts_original = Counter(y_train)
    for c_idx in sorted(class_counts_original):
        strategy_data.append({
            "class": int_to_class[c_idx],
            "original_count": class_counts_original[c_idx],
            "dift_target": dift_strategy.get(c_idx, class_counts_original[c_idx]),
            "adaptive_target": adaptive_strategy.get(c_idx, class_counts_original[c_idx]),
            "preliminary_f1": per_class_f1.get(int_to_class[c_idx], 1.0),
        })
    
    strategy_df = pd.DataFrame(strategy_data)
    strategy_df.to_csv(os.path.join(RESULTS_DIR, "augmentation_strategy_comparison.csv"), index=False)
    
    print(f"\n✓ Augmentation complete.")
    print(f"  DIFT-fixed augmented: {len(y_dift):,} samples")
    print(f"  SHAP-DA-IDS augmented: {len(y_adaptive):,} samples")
    print(f"\nAugmentation strategy comparison:")
    print(strategy_df.to_string(index=False))
    
    return X_train_sel, X_test_sel, y_train, y_test, X_dift, y_dift, X_adaptive, y_adaptive


if __name__ == "__main__":
    main()
