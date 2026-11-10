# Proposed Methodology: SHAP-DA-IDS

## Full Title

**SHAP-DA-IDS: Difficulty-Adaptive Augmentation with Explainable Feature Selection for Imbalanced IoT Intrusion Detection**

---

## 1. Overview

We propose **SHAP-DA-IDS**, an extension of the DIFT framework that introduces three improvements:

1. **SHAP-based Feature Selection (SFS):** Use SHapley Additive exPlanations on a preliminary classifier to rank feature importance. Remove low-importance features before augmentation and training. This reduces input noise that dilutes minority-class signals.

2. **Difficulty-Adaptive Augmentation (DAA):** Instead of DIFT's fixed average-count augmentation target, compute per-class difficulty scores from the preliminary classifier's per-class F1. Set augmentation targets proportional to difficulty — harder classes receive more synthetic samples.

3. **Explainability Layer (EL):** After final classification, provide SHAP-based explanations (global feature importance + per-prediction local attribution) to make predictions interpretable for security analysts.

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     SHAP-DA-IDS Pipeline                        │
│                                                                 │
│  Imbalanced IoT Dataset                                         │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────┐                                    │
│  │  Phase 1: Feature       │                                    │
│  │  Selection              │                                    │
│  │  - Train preliminary RF │                                    │
│  │  - Compute SHAP values  │                                    │
│  │  - Rank features by     │                                    │
│  │    mean |SHAP|          │                                    │
│  │  - Select top-k features│                                    │
│  └────────────┬────────────┘                                    │
│               │ Reduced feature set                             │
│               ▼                                                 │
│  ┌─────────────────────────┐                                    │
│  │  Phase 2: Difficulty    │                                    │
│  │  Estimation             │                                    │
│  │  - Compute per-class F1 │                                    │
│  │    from preliminary RF  │                                    │
│  │  - Difficulty score =   │                                    │
│  │    1 - per_class_F1     │                                    │
│  │  - Normalize difficulty │                                    │
│  └────────────┬────────────┘                                    │
│               │ Per-class difficulty scores                     │
│               ▼                                                 │
│  ┌─────────────────────────┐                                    │
│  │  Phase 3: Difficulty-   │                                    │
│  │  Adaptive Augmentation  │                                    │
│  │  - Target_i = majority  │                                    │
│  │    * difficulty_i       │                                    │
│  │    * α (scale factor)   │                                    │
│  │  - Apply SMOTE per class│                                    │
│  │    to reach Target_i    │                                    │
│  └────────────┬────────────┘                                    │
│               │ Balanced, feature-selected dataset              │
│               ▼                                                 │
│  ┌─────────────────────────┐                                    │
│  │  Phase 4: Final         │                                    │
│  │  Classification         │                                    │
│  │  - Train XGBoost        │                                    │
│  │    classifier           │                                    │
│  │  - Evaluate on holdout  │                                    │
│  │    test set             │                                    │
│  └────────────┬────────────┘                                    │
│               │ Predictions                                     │
│               ▼                                                 │
│  ┌─────────────────────────┐                                    │
│  │  Phase 5: Explainability│                                    │
│  │  - SHAP TreeExplainer   │                                    │
│  │  - Global summary plot  │                                    │
│  │  - Per-class SHAP values│                                    │
│  │  - Feature attribution  │                                    │
│  └─────────────────────────┘                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Algorithm

### Phase 1: SHAP-based Feature Selection

```
Input: X_train (imbalanced), y_train, feature_names
Output: X_train_reduced, selected_features

1. Train preliminary RandomForest classifier on (X_train, y_train)
   - n_estimators = 100, max_depth = None, random_state = 42
   
2. Compute SHAP values using shap.TreeExplainer(rf_model)
   - shap_values = explainer.shap_values(X_train_sample)
   - For multi-class: take mean of |shap_values| across classes and samples
   
3. Rank features by mean |SHAP| (descending)

4. Select top-k features (k = number of features with cumulative importance ≥ 95%)
   - Alternatively: select features above mean importance threshold

5. Return reduced dataset with selected features
```

**Justification:** Removing redundant and low-importance features reduces input dimensionality, potentially improves minority-class signal-to-noise ratio, and reduces augmentation complexity.

### Phase 2: Difficulty Estimation

```
Input: Preliminary RF predictions on X_val, y_val
Output: difficulty_scores (dict: class → difficulty)

1. Compute per-class F1 scores from preliminary model's CV predictions
   - f1_scores = classification_report(y_val, y_pred, output_dict=True)
   
2. For each class c:
   difficulty_c = 1.0 - f1_scores[c]['f1-score']
   
3. Normalise: difficulty_c_norm = difficulty_c / sum(difficulty_c)
   (Excludes majority class from normalisation)

4. Return difficulty_scores
```

**Justification:** A class with low F1 on a preliminary model has high detection difficulty. It needs more augmentation to become learnable. This is more principled than DIFT's fixed average-count target.

### Phase 3: Difficulty-Adaptive Augmentation

```
Input: X_train_reduced, y_train, difficulty_scores, α (scale factor)
Output: X_augmented, y_augmented

1. Compute majority class count: n_majority = count(normal class)

2. For each minority class c:
   target_c = min(n_majority, 
                  max(2 * count(class_c), 
                      int(n_majority * difficulty_c_norm * α)))
   
   where α = 0.3 (tunable; keeps total augmented data manageable)
   
3. Apply SMOTE with custom sampling_strategy:
   sampling_strategy = {c: target_c for c in minority_classes}
   
4. Fit SMOTE only on training data (prevent data leakage)

5. Return augmented training data
```

**Note on SMOTE vs. Diffusion:** SMOTE is used as the augmentation mechanism. While DIFT uses a diffusion model, the novelty here is in the **adaptive target-setting strategy**, not the augmentation mechanism itself. SMOTE produces linearly interpolated samples between existing minority-class instances. This is a tractable implementation that enables fair comparison. If the full DIFT diffusion model were available, the difficulty-adaptive strategy would apply directly to it (the same target-setting logic, different augmentation backend).

**Comparison with DIFT's strategy:**
- DIFT: target_c = average_class_count (fixed for all minority classes)
- SHAP-DA-IDS: target_c ∝ difficulty_c (adaptive per class)

### Phase 4: Final Classification

```
Input: X_augmented_train, y_augmented_train, X_test_reduced, y_test
Output: predictions, probabilities

1. Train XGBoost classifier:
   - n_estimators = 200
   - max_depth = 6
   - learning_rate = 0.1
   - use_label_encoder = False
   - eval_metric = 'mlogloss'
   - random_state = 42

2. Evaluate on original test set (not augmented):
   - accuracy, precision, recall, f1 (macro & weighted)
   - per-class precision, recall, F1
   - confusion matrix
   - ROC-AUC (macro OvR)

Important: Test set is never augmented or modified.
```

**Why XGBoost:** It is fast, interpretable via SHAP (TreeSHAP is exact for tree models), and achieves competitive results on tabular data. It is a strong, well-established baseline that enables exact SHAP computation (unlike neural networks requiring KernelSHAP approximation).

### Phase 5: Explainability

```
Input: Trained XGBoost, X_test_reduced, y_test, class_names
Output: SHAP summary plots, per-class attribution

1. Compute SHAP values: explainer = shap.TreeExplainer(xgb_model)
                         shap_values = explainer.shap_values(X_test_reduced)

2. Global summary plot: shap.summary_plot(shap_values, X_test_reduced, 
                                           feature_names, class_names)

3. Per-class bar plots: for each class c, top-10 features by mean |SHAP_c|

4. Local explanation example: waterfall plot for one MITM prediction
   (most informative for the hardest class)
```

---

## 4. Comparison with Original DIFT

| Component | Original DIFT | SHAP-DA-IDS (Our Approach) |
|---|---|---|
| **Augmentation method** | Diffusion model (U-Net backbone) | SMOTE (tractable proxy) |
| **Augmentation target** | Fixed: average class count | Adaptive: proportional to per-class detection difficulty |
| **Feature selection** | None — all features used | SHAP-guided: selects top-k features by importance |
| **Augmentation guidance** | No guidance; all minority classes augmented equally | Harder classes (lower F1) receive more synthetic samples |
| **Classifier** | Time Series Transformer (patching + multi-head attention) | XGBoost (gradient-boosted trees) |
| **Explainability** | None — black-box output | SHAP TreeExplainer — global & local feature attribution |
| **Transparency** | No feature rankings | Full feature importance ranking per class |
| **Compute required** | NVIDIA A100 GPU (200 diffusion epochs + 100 classifier epochs) | CPU-feasible (minutes to hours) |
| **Code availability** | No public code | Full reproducible code provided |

**Important clarification:** The augmentation proxy change (diffusion → SMOTE) means our implementation cannot directly reproduce DIFT's accuracy numbers. The novelty is the difficulty-adaptive strategy and SHAP integration. We compare our proposed SHAP-DA-IDS against classic baselines (LR, RF, SMOTE+XGB, XGB with class weights) to demonstrate the value of our contributions.

---

## 5. Baselines for Comparison

| Baseline | Description | Addresses Imbalance? | Addresses Explainability? |
|---|---|---|---|
| Logistic Regression | Linear classifier | ❌ No | Coefficient-level only |
| Random Forest (no balancing) | Ensemble trees | ❌ No | SHAP possible |
| XGBoost (no balancing) | Gradient boosting | ❌ No | SHAP possible |
| XGBoost + Class Weights | Cost-sensitive | ✅ Partial | SHAP possible |
| SMOTE + XGBoost (fixed) | Fixed SMOTE (like DIFT's strategy) | ✅ Fixed target | SHAP possible |
| **SHAP-DA-IDS (ours)** | Adaptive SMOTE + feature selection + SHAP | ✅ Adaptive | ✅ Full |

---

## 6. Evaluation Plan

### Primary Metrics
- **Macro F1-score** (most important for imbalanced datasets — treats all classes equally)
- **Weighted F1-score**
- **Per-class Precision, Recall, F1** (especially for MITM, Malicious Control, Wrong Setup)
- **Accuracy**
- **ROC-AUC (macro OvR)**
- **Confusion matrix**
- **Training time**
- **Inference time**

### Figures to Generate
1. Dataset class distribution (bar chart showing imbalance)
2. SHAP-DA-IDS pipeline architecture diagram
3. Model comparison (macro F1 bar chart)
4. Confusion matrix (best model)
5. Per-class F1 comparison (all models)
6. SHAP feature importance (global summary plot)
7. Difficulty-adaptive augmentation target visualisation (original vs. our strategy)

---

## 7. Expected Outcomes

**Hypothesis:** Difficulty-adaptive augmentation will improve per-class F1 for the rarest, hardest-detected classes (MITM, Malicious Control, Wrong Setup) compared to fixed-target augmentation (SMOTE + XGBoost). Feature selection via SHAP will further reduce noise. The explainability layer will provide actionable feature attributions unavailable in DIFT.

**Validation:** 
- If per-class F1 for the rarest classes improves under SHAP-DA-IDS vs. fixed SMOTE+XGBoost → hypothesis supported
- If macro F1 improves vs. no-balancing baselines → imbalance handling is effective
- If SHAP explanations identify domain-relevant features (consistent with network security knowledge) → explanations are meaningful

---

## 8. Reproducibility Requirements

- Fixed random seed: **RANDOM_SEED = 42** throughout
- Train/test split: **80/20 stratified**, matching DIFT's approximate split ratio
- All preprocessing fitted only on training data (no data leakage)
- Full experiment log to `results/experiment_log.txt`
- Results to `results/metrics.csv`, `results/model_comparison.csv`
- All plots saved to `results/figures/`
- Code modular and documented in `src/`
