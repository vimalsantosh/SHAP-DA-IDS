# Research Gap and Novelty Analysis

## 1. Literature Review Summary

Based on a systematic review of 2023–2026 research on IoT intrusion detection, imbalanced learning, diffusion models, and explainable AI:

### 1.1 State of the Art

| Theme | Key Works | Methods | Limitation |
|---|---|---|---|
| Imbalanced IoT IDS | DIFT (Wang et al., 2025) | Diffusion + Transformer | No explainability, fixed augmentation target |
| GAN-based augmentation | S2CGAN-IDS (Wang et al., 2023) | GAN + CNN | Mode collapse |
| Transformer IDS | IDS-INT (Ullah et al., 2023) | Transformer + transfer learning | No imbalance handling |
| XAI + IDS | XCM (Fauvel et al., 2021) | Explainable CNN | No imbalance handling, coarse explanation |
| SHAP + ML IDS | Multiple 2024 works | SHAP + Random Forest/ML | Not applied with diffusion-augmented data |
| Edge IoT IDS | Multiple 2024–2025 works | Lightweight attention | No diffusion-based generation |
| Tabular diffusion | TabDDPM + variants | Diffusion for tabular | General domain, not IoT-specific IDS |

### 1.2 What DIFT Does NOT Address

After carefully reading the full DIFT paper and surveying recent literature:

1. **No explainability:** DIFT provides no mechanism to identify which features contributed to a detection decision. Security Operations Center (SOC) analysts cannot trust or audit predictions.

2. **Fixed augmentation target — blind to class difficulty:** DIFT sets the generation target as the average class count. For MITM (66.22% F1 on TON_IoT) and Malicious Control (79.48% F1 on DS2OS), this is insufficient. The model still fails at the rarest, hardest classes.

3. **No feature importance or selection:** DIFT uses all available features (44 for TON_IoT, 13 for DS2OS) without evaluating which features are discriminative. Redundant features dilute minority-class signals.

4. **Black-box prediction output:** Predictions are raw class labels with probabilities. No attribution information.

5. **No joint optimisation of augmentation quality and classification:** The diffusion phase and classification phase are entirely separate ("offline generation → training").

---

## 2. Proposed Novelty Options

### Option A: SHAP-based Post-hoc Explainability Layer on DIFT Pipeline

**What changes:** After training the classifier on diffusion-augmented data, apply SHAP to produce global feature rankings and per-prediction local explanations.

**Why useful:** Addresses the black-box problem. Security analysts need to understand why a prediction was made.

**Problem addressed:** Interpretability gap in deep-learning IoT IDS.

**How implemented:** Use `shap.KernelExplainer` or `shap.TreeExplainer` (for the classifier component). Plot SHAP summary plots, bar plots, and beeswarm plots per class.

**Experiments:** Compare prediction quality before/after SHAP-guided feature pruning; show which features drive minority class detection.

**Compute required:** Moderate. SHAP on tabular classifiers is feasible on CPU in minutes.

**Related work:** XCM claims explainability via conv filters; no prior work applies SHAP specifically to diffusion-augmented IoT IDS pipelines.

**Novelty assessment:** Moderate. SHAP is well-known. Applying it to a diffusion-augmented transformer IDS and studying whether SHAP explanations are stable and meaningful under synthetic augmentation is genuinely novel.

---

### Option B: Difficulty-Adaptive Diffusion Augmentation (D-DIFT)

**What changes:** Instead of a fixed "average class count" target, compute a per-class difficulty score from a lightweight preliminary classifier (e.g., logistic regression or random forest). Classes with lower F1 scores on the preliminary model receive proportionally more synthetic samples.

**Why useful:** Addresses the core weakness — MITM (0.23% of data, 66.22% F1) and Malicious Control (0.25%, 79.48% F1) need far more augmentation than DoS (5.78% of data, already well-detected).

**Problem addressed:** Fixed augmentation target is blind to per-class difficulty.

**How implemented:**
1. Train a preliminary Random Forest on the original imbalanced data.
2. Compute per-class F1 scores from the preliminary model.
3. Compute difficulty score = 1 - per_class_F1 (normalised).
4. Set generation target for each class proportional to difficulty score.
5. Run diffusion generation with class-conditioned targets.
6. Train the full DIFT-style classifier.

**Experiments:** Compare per-class F1 improvement vs. original DIFT's fixed strategy; show improvement specifically on hardest classes.

**Compute required:** Low-to-moderate. Preliminary RF + existing diffusion-equivalent (using SMOTE as proxy since full diffusion needs PyTorch).

**Novelty assessment:** Genuine novel contribution. The concept of difficulty-adaptive augmentation targets for diffusion models in IoT IDS is not covered in existing literature. The paper's limitation ("limitations in detecting multi-layer attacks" and high failure on MITM/Malicious Control) makes this directly motivated.

---

### Option C: Combined SHAP Feature Selection + Difficulty-Adaptive Augmentation (SHAP-DA-IDS)

**What changes:** Combines A and B.
1. SHAP-based feature selection on preliminary model → remove low-importance features
2. Train diffusion-equivalent model on reduced feature set
3. Difficulty-adaptive augmentation target setting
4. Classifier training on balanced, feature-selected data
5. Final SHAP explainability layer for predictions

**Why useful:** Addresses both the efficiency problem (feature redundancy dilutes minority signal) and the difficulty problem (fixed augmentation target).

**Problem addressed:** Feature noise + fixed augmentation + no explainability.

**How implemented:** Feasible with scikit-learn, SHAP, and SMOTE (as diffusion proxy). Full pipeline implementable on CPU.

**Experiments:** Ablation on feature selection threshold, augmentation strategy comparison (fixed vs. adaptive), final explainability analysis.

**Compute required:** Moderate. All components run on CPU with standard libraries.

**Novelty assessment:** Strong. The combination of SHAP-guided feature selection + difficulty-adaptive augmentation + explainability output for IoT IDS is not found in existing literature (2023–2026).

---

### Option D: Lightweight Edge-Deployable IDS

**What changes:** Knowledge distillation from DIFT to a small student model; model quantisation.

**Why useful:** IoT devices have constrained compute.

**Novelty assessment:** Moderate, but requires a full DIFT implementation first (source code unavailable). Not feasible within constraints.

---

### Option E: Adversarial Robustness Evaluation

**What changes:** Evaluate DIFT against adversarial examples crafted to evade detection.

**Novelty assessment:** Interesting but tangential to the original paper's focus.

---

## 3. Selected Novelty: **Option C — SHAP-guided Difficulty-Adaptive Augmentation IDS (SHAP-DA-IDS)**

### Rationale for Selection

| Criterion | Assessment |
|---|---|
| Directly motivated by DIFT's stated limitations? | ✅ Yes — MITM, Malicious Control failures; no explainability |
| Genuinely novel vs. existing literature? | ✅ Yes — no prior work combines SHAP feature selection + difficulty-adaptive augmentation for IoT IDS |
| Feasible without DIFT source code? | ✅ Yes — can use SMOTE as augmentation proxy; ML classifiers are transparent |
| Feasible with available compute (no A100 GPU)? | ✅ Yes — all components CPU-compatible |
| Defensible academic contribution? | ✅ Yes — addresses two independently identified gaps |
| Can produce real experimental results? | ✅ Yes — using DS2OS (publicly downloadable) |

### Honest Assessment of Limitations
- We cannot re-implement the full DIFT model (requires NVIDIA A100-class GPU and undisclosed U-Net architecture details). Our approach uses a **tabular classifier + SMOTE** as the augmentation proxy, with the novelty residing in the difficulty-adaptive strategy and SHAP integration.
- We will clearly label our implementation as an **approximation of the DIFT pipeline** with our proposed improvements, NOT as an exact DIFT reproduction.
- The comparison with DIFT's reported numbers is for context; our direct baselines are the classifiers we actually train and compare.

---

## 4. Research Gap Statement (for paper)

> Although DIFT advances the state-of-the-art in IoT intrusion detection by using diffusion-based minority-class augmentation, it has three notable limitations: (1) its augmentation target is set as a fixed average class count, failing to account for per-class detection difficulty, leaving the rarest and hardest-to-detect attack classes (such as MITM with 66.22% F1 and Malicious Control with 79.48% F1) under-augmented; (2) it performs no feature selection, exposing the classifier to redundant features that dilute the minority-class signal; and (3) it produces no explanations, making its predictions opaque to security analysts. No prior work has simultaneously addressed all three gaps for IoT IDS with imbalanced learning.

---

## 5. References Supporting This Gap Analysis

1. Wang, P. et al., "DIFT: A Diffusion-Transformer for Intrusion Detection of IoT with Imbalanced Learning," JNSM 2025.
2. Lundberg, S., Lee, S., "A Unified Approach to Interpreting Model Predictions," NeurIPS 2017.
3. Ho, J. et al., "Denoising Diffusion Probabilistic Models," NeurIPS 2020.
4. Ullah, F. et al., "IDS-INT: Intrusion Detection System using Transformer-based Transfer Learning," DCN 2023.
5. Fauvel, K. et al., "XCM: An Explainable CNN for Multivariate Time Series Classification," Mathematics 2021.
6. Wang, C. et al., "Effective Intrusion Detection in Highly Imbalanced IoT Networks with S2CGAN-IDS," IEEE IoT Journal 2023.
7. Thakkar, A., Lohiya, R., "Attack Classification of Imbalanced Intrusion Data for IoT Network," IEEE IoT Journal 2023.
8. Chawla, N.V. et al., "SMOTE: Synthetic Minority Over-sampling Technique," JAIR 2002.
9. Chen, T., Guestrin, C., "XGBoost: A Scalable Tree Boosting System," KDD 2016.
