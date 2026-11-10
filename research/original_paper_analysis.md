# Original Paper Analysis: DIFT

## Bibliographic Information

- **Title:** DIFT: A Diffusion-Transformer for Intrusion Detection of IoT with Imbalanced Learning
- **Authors:** Peng Wang, Yafei Song, Xiaodan Wang, Xiangke Guo
- **Journal:** Journal of Network and Systems Management, Springer, 2025
- **Volume:** 33, Article 48
- **DOI:** https://doi.org/10.1007/s10922-025-09926-z
- **Received:** 5 October 2024 | **Accepted:** 15 April 2025 | **Published:** 24 April 2025
- **Affiliation:** Air Force Engineering University, Xi'an, China

---

## 1. Problem Statement

IoT intrusion detection faces a **severe class imbalance problem**: normal network traffic samples vastly outnumber attack samples. Standard deep learning models trained on such skewed data are biased toward the majority class (normal traffic) and fail to detect rare but dangerous attack types (minority classes such as MITM, Spying, Wrong Setup, etc.).

Specifically:
- In DS2OS: "Wrong setup" has only 122 samples vs. 347,935 "Normal" samples (imbalance ratio ~2851:1)
- In TON_IoT: "MITM" has only 1,034 samples vs. 300,000 "Normal" samples (imbalance ratio ~290:1)

This leads to **high false-negative rates** for rare attacks — meaning dangerous intrusions go undetected.

---

## 2. Existing Approaches Surveyed (Related Work)

The paper categorises existing methods as:

### A. Algorithm Adjustment Methods
- **Ensemble learning with class weights** (Thakkar et al., 2023): Uses bagging with DNNs; manually sets class weights. Problem: compromises majority-class detection, poor generalization.
- **Multi-classifier ensemble** (Abbas et al., 2022): Combines multiple classifiers. Problem: added complexity without generative understanding.
- **Focal loss** (Dina et al., 2023): Dynamically down-weights easy samples. Problem: still relies on manual hyperparameter tuning.

### B. Sample Generation Methods
- **SMOTE/Undersampling**: Classic resampling. Problem: ignores feature patterns, creates duplicate/invalid samples in heterogeneous IoT data.
- **GAN-based methods** (S2CGAN-IDS, Wang et al., 2023; Nie et al., 2021): GAN learns traffic distribution and generates minority samples. Problem: **mode collapse** — generator fails to capture diverse minority patterns.
- **VAE-based methods**: Similar to GAN but uses variational encoder. Problem: inferior sample quality compared to diffusion models.

### C. Limitations Identified by the Authors
- Existing methods "do not consider the imbalanced data itself and ignore the intrinsic structure and information of the class data"
- GAN "unable to effectively extract the characteristic patterns of traffic data to generate diverse minority class samples"
- Traditional sampling "causes the model to be over-sensitive to synthetic samples, especially in IoT heterogeneous scenarios"

---

## 3. DIFT Architecture

The DIFT framework has **three main phases**:

```
Raw IoT Traffic Data (Imbalanced)
         |
         v
[Phase 1: Dataset Generation Phase]
  - Diffusion Model (U-Net backbone)
  - Forward Diffusion: adds Gaussian noise over T steps
  - Backward Denoising: reconstructs minority class samples
  - Output: Balanced dataset (minority classes augmented)
         |
         v
[Phase 2: Patch Embedding]
  - Sliding window over traffic sequence X = [x1, x2, ..., xL]
  - Patch size P, stride S
  - Converts 1D sequence → 2D patch structure
  - Reduces sequence length from L to N = floor((L - P + S) / S)
  - Maps patches to D-dimensional embedding: X_E = X_p * W (W ∈ R^{P×D})
  - Adds positional encoding: x_F_i = x_E_i + p(i)
         |
         v
[Phase 3: Feature Extraction & Detection Phase]
  - Time Series Transformer (stacked encoder layers)
  - Multi-Head Self-Attention: Q=XW_q, K=XW_k, V=XW_v
  - Attention score: S = QK^T / sqrt(d_k)
  - Attention weights: A = softmax(S), Output: O = AV
  - Feed-Forward Network (3 layers: expand, nonlinear, reduce)
  - Layer normalization + residual connections
  - SoftMax classifier → class prediction
```

### 3.1 Diffusion Model Details

**Forward Process (adds noise):**
```
q(x_t | x_{t-1}) = N(x_t; sqrt(1-β_t) * x_{t-1}, β_t * I)
```
where β_t is the noise schedule parameter at step t.

**Backward Process (removes noise — the generative step):**
```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```
where μ_θ and Σ_θ are parameterised by a neural network.

**Loss Function:** Variational Lower Bound (VLB):
```
L_D = E_{q(x_0)} [Σ_{t=1}^{T} D_KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t)) - log p(x_0|x_1)]
```

**Backbone:** U-Net architecture with encoder (convolution + downsampling) and decoder (transposed convolution + skip-connections).

**Beta Schedule:** Cosine schedule.

**Training:** 200 epochs, Adam optimizer, learning rate 1e-3, EMA decay 0.995, batch size 16.

### 3.2 Patching Details

- IoT sequence data X = [x_1, x_2, ..., x_L]
- Patch window size P = 4, stride S = 2
- Number of patches: N = (L - P + 2S) / S (with tail padding of length S)
- Embedding dimension D = 64
- Complexity reduced from O(L²) to O(N²)

### 3.3 Time Series Transformer Details

| Hyperparameter | Value |
|---|---|
| Embedding dim | 64 |
| Multi-head attention heads | 8 |
| FFN dim | 128 |
| Number of encoder layers | 2 |
| Batch size | 1024 |
| Epochs | 100 |
| Learning rate | 1e-3 |
| Optimizer | Adam |

**Key equations:**
- Q = X·W_q, K = X·W_k, V = X·W_v (W_q, W_k, W_v ∈ R^{d×d_k})
- Attention: S = QK^T / sqrt(d_k); A = softmax(S); O = A·V
- Multi-head: H_i = Attention(X·W^Q_i, X·W^K_i, X·W^V_i); MH = Concat(H_1,...,H_h)·W_m
- FFN: E_l = (ReLU(LN(MH+Y)·W_1))·W_2; O_l = LN(E_l + LN(MH+Y))
- Final classifier: Y = Softmax(O) = exp(o^l_i) / Σ_k exp(o^l_k)

---

## 4. Dataset Details

### 4.1 DS2OS Dataset

| Attack Type | Train | Test | Total | Diffused Total |
|---|---|---|---|---|
| Normal | 260,951 | 86,984 | 347,935 | — |
| DoS | 4,335 | 1,445 | 5,780 | — |
| Scan | 1,160 | 387 | 1,547 | — |
| Malicious Control | 667 | 222 | 889 | 1,778 |
| Malicious Operation | 604 | 201 | 805 | 1,610 |
| Spying | 399 | 133 | 532 | 1,596 |
| Data Probing | 257 | 86 | 342 | 1,710 |
| Wrong Setup | 92 | 31 | 122 | 1,464 |

- **Total samples:** 357,952
- **8 classes:** 1 normal + 7 attack types
- **Imbalance ratio (worst case):** Normal:Wrong Setup ≈ 2,851:1
- **Features:** 13 (mostly nominal/categorical + continuous Value field + Timestamp)
- **Domain:** Smart home IoT; covers MQTT, HTTP protocols
- **Missing values:** Accessed Node Type and Value fields have some missing values

### 4.2 TON_IoT Dataset

| Attack Type | Train | Test | Total | Diffused Total |
|---|---|---|---|---|
| Normal | 210,000 | 90,000 | 300,000 | — |
| DDoS | 14,000 | 6,000 | 20,000 | — |
| DoS | 14,000 | 6,000 | 20,000 | — |
| Injection | 14,000 | 6,000 | 20,000 | — |
| MITM | 730 | 313 | 1,043 | 10,340 |
| Password | 14,000 | 6,000 | 20,000 | — |
| Ransomware | 14,000 | 6,000 | 20,000 | — |
| Scanning | 14,000 | 6,000 | 20,000 | — |
| XSS | 14,000 | 6,000 | 20,000 | — |
| Backdoor | 14,000 | 6,000 | 20,000 | — |

- **Total samples:** ~461,043
- **10 classes:** 1 normal + 9 attack types
- **Imbalance ratio (worst case):** Normal:MITM ≈ 290:1
- **Features:** ~44 network flow features (from IoT/IIoT network traffic)
- **Domain:** Industrial IoT + home IoT; covers MQTT, CoAP, HTTP
- **Source:** UNSW Canberra Cyber Range Lab, ADFA

### 4.3 Preprocessing

- **Standard normalisation** applied to both training and test sets
- Diffusion model generates minority samples until they reach approximately the **average class count** (rounded to a multiple of the original count)
- The generation phase is performed **offline** (before training the classifier)

---

## 5. Baselines Compared

| Baseline | Type | Description |
|---|---|---|
| LSTM | Deep learning | Long Short-Term Memory for temporal sequence |
| E-GraphSAGE | Graph neural network | Graph-based IDS for IoT |
| GMS-IDS | CNN | Multi-scale convolution (CNN+RNN) |
| XCM | CNN | Explainable CNN for multivariate time series |
| DRL | RL + GAN | Distributional reinforcement learning |
| ResNet-50 | CNN | Residual network |
| FT-Transformer | Transformer | Feature tokenizer transformer |
| IFT with VAE | Transformer + VAE | DIFT's transformer with VAE-based augmentation |
| IFT with GAN | Transformer + GAN | DIFT's transformer with GAN-based augmentation |
| DIFT | Transformer + Diffusion | The proposed full model |

**Hardware:** Intel Xeon Silver 4214R CPU, NVIDIA A100 80GB GPU, 128 GB RAM  
**Framework:** PyTorch 1.12.1, CUDA 12.1

---

## 6. Experimental Results

### 6.1 DS2OS Results (Full DIFT)

| Label | Precision (%) | Recall (%) | F1 (%) | ROC-AUC |
|---|---|---|---|---|
| Normal | 99.43 | 100 | 99.72 | 0.9993 |
| DoS | 100 | 100 | 100 | 1.0000 |
| Scan | 99.08 | 98.63 | 98.86 | 0.9990 |
| Malicious Control | 100 | 65.95 | 79.48 | 0.9940 |
| Malicious Operation | 99.50 | 99.01 | 99.26 | 0.9993 |
| Spying | 99.44 | 99.44 | 99.44 | 0.9993 |
| Data Probing | 96.73 | 99.04 | 97.87 | 0.9988 |
| Wrong Setup | 97.91 | 96.39 | 97.14 | 0.9983 |
| **Overall Accuracy** | | | **99.43%** | |

### 6.2 DS2OS Comparison with Baselines

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) |
|---|---|---|---|---|
| LSTM | 97.19 | 97.07 | 97.05 | 97.14 |
| E-GraphSAGE | 92.26 | 92.21 | 91.92 | 92.11 |
| GMS-IDS | 98.42 | 97.37 | 98.32 | 98.36 |
| XCM | 98.33 | 98.23 | 98.27 | 98.29 |
| DRL | 98.85 | 97.99 | 99.00 | 98.90 |
| ResNet-50 | 96.67 | 96.52 | 96.57 | 96.61 |
| FT-Transformer | 97.74 | 97.64 | 97.65 | 97.62 |
| IFT with VAE | 99.01 | 98.80 | 98.84 | 98.85 |
| IFT with GAN | 99.26 | 99.02 | 99.11 | 99.05 |
| **DIFT** | **99.43** | **99.41** | **99.39** | **99.41** |

### 6.3 TON_IoT Results (Full DIFT)

| Label | Precision (%) | Recall (%) | F1 (%) | ROC-AUC |
|---|---|---|---|---|
| Normal | 99.56 | 99.88 | 99.72 | 0.9999 |
| DDoS | 98.25 | 96.08 | 97.15 | 0.9997 |
| DoS | 99.17 | 97.22 | 98.19 | 0.9998 |
| Injection | 98.17 | 94.18 | 96.14 | 0.9995 |
| MITM | 58.80 | 75.77 | 66.22 | 0.9982 |
| Password | 99.37 | 98.73 | 98.08 | 0.9998 |
| Ransomware | 98.98 | 97.20 | 98.08 | 0.9998 |
| Scanning | 98.45 | 98.67 | 98.56 | 0.9999 |
| XSS | 94.70 | 96.39 | 97.14 | 0.9997 |
| Backdoor | 99.87 | 99.90 | 99.89 | 0.9999 |
| **Overall Accuracy** | | | **99.02%** | |

### 6.4 TON_IoT Comparison with Baselines

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) |
|---|---|---|---|---|
| LSTM | 96.54 | 96.54 | 95.21 | 95.83 |
| E-GraphSAGE | — | 82.79 | 81.82 | 82.30 |
| GMS-IDS | 98.16 | 96.86 | 94.83 | 95.70 |
| XCM | 97.78 | 97.77 | 97.24 | 97.47 |
| DRL | 98.28 | 97.92 | 98.14 | 98.02 |
| ResNet-50 | 95.84 | 95.78 | 94.76 | 96.86 |
| FT-Transformer | 97.06 | 96.94 | 96.67 | 97.23 |
| IFT with VAE | 98.58 | 97.74 | 98.25 | 98.28 |
| IFT with GAN | 98.86 | 98.42 | 98.57 | 98.54 |
| **DIFT** | **99.02** | **98.94** | **98.98** | **98.99** |

### 6.5 Ablation Study (DS2OS)

| Patch Embedding | Generation Phase | Feature Extraction | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) |
|---|---|---|---|---|---|---|
| ✗ | ✗ | ✓ | 98.64 | 98.51 | 98.58 | 98.62 |
| ✗ | ✓ | ✓ | 99.28 | 99.01 | 98.79 | 99.08 |
| ✓ | ✗ | ✓ | 98.86 | 99.02 | 98.85 | 98.83 |
| ✓ | ✓ | ✓ | **99.43** | **99.41** | **99.39** | **99.41** |

Generation phase contributes +0.65% accuracy; Patching contributes +0.22%; both together: +0.80%.

### 6.6 Model Complexity (DS2OS)

- **DIFT:** 4.71M parameters, 337 ms batch inference time, 99.43% accuracy
- **LSTM:** 2.62M parameters (fewest), 97.19% accuracy
- Patching reduces tokens and thus memory + inference time significantly

---

## 7. Key Limitations Explicitly Stated in the Paper

1. **Time-consuming generation phase:** The diffusion model's sample generation is slow (sequential denoising). The paper explicitly states this as a limitation.
2. **Limited multi-layer attack detection:** DIFT is limited in detecting "multi-layer attacks" (composite, coordinated attacks). This is explicitly stated.
3. **Weak minority-class recall (MITM on TON_IoT):** MITM class recall is only 75.77% and F1 only 66.22% — the worst-performing class despite being the rarest (0.23% of data). The diffusion model struggles with extremely rare classes.
4. **Weak minority-class recall (Malicious Control on DS2OS):** Recall of only 65.95% and F1 of 79.48% for Malicious Control — only 0.25% of data.
5. **No explainability:** DIFT is a black-box model. There is no mechanism to explain which features drove a particular prediction. This is unmentioned but is a clear gap.
6. **Fixed diffusion target:** The augmentation target is set as the average class count, not adaptively tuned to each class's difficulty.
7. **No feature selection:** All 44 (TON_IoT) or 13 (DS2OS) features are used without evaluating feature importance or removing redundant features.
8. **Offline augmentation only:** Diffusion generation is done offline before training, not adaptively during training.

---

## 8. Potential Research Extensions (Identified)

Based on the limitations and research gaps:

### Extension A: SHAP-based Explainability for DIFT
- Add SHAP (SHapley Additive exPlanations) post-hoc analysis to the DIFT classifier
- Provide global feature importance ranking and per-class local explanations
- Addresses the black-box problem; relevant for SOC analysts and trust in IoT IDS
- **Gap status:** DIFT has zero explainability; XCM (a baseline) claims explainability via conv filters but no post-hoc attribution on the DIFT pipeline exists

### Extension B: Adaptive Difficulty-Weighted Diffusion Augmentation
- Instead of a fixed "average class count" target, use per-class difficulty scoring (e.g., based on initial classifier confidence or minority-class overlap) to adaptively decide how many samples to generate for each class
- Addresses the MITM (66.22% F1) and Malicious Control (79.48% F1) problem
- **Gap status:** DIFT uses a fixed average-count heuristic; adaptive difficulty-aware augmentation is under-explored for tabular diffusion

### Extension C: SHAP-guided Feature Selection + DIFT
- Use SHAP on a preliminary model to rank and prune features before the full DIFT pipeline
- Reduces input dimensionality, removes noise features, may improve rare-class detection
- **Gap status:** DIFT uses all features without selection; combining SHAP feature selection with diffusion augmentation is not covered in the paper

### Extension D: Lightweight/Edge Deployment Optimisation
- Suggested explicitly by the paper: "edge-cloud cooperative intrusion detection"
- Requires knowledge distillation or quantisation of the DIFT model for IoT edge devices
- **Gap status:** High computational complexity gap, but requires significant engineering

### Extension E: Class-Conditional Diffusion (Explicit Conditioning)
- DIFT's diffusion generates minority samples per class but does not appear to use explicit class-label conditioning in the denoising network
- Adding explicit class-label conditioning (conditioning the U-Net on class embeddings) may improve sample quality
- **Gap status:** The paper does not clearly state whether the diffusion model is conditioned on class labels; this is an architectural gap

---

## 9. Information That Could NOT Be Independently Verified

1. **Source code:** No public GitHub repository for DIFT was found. The implementation details are described in text/equations but not verifiable code.
2. **Exact preprocessing pipeline:** The paper states "standard normalisation" without specifying MinMax vs. Z-score, feature encoding strategy, or which TON_IoT subset (Network, Windows OS, etc.) was used.
3. **Exact U-Net architecture:** The paper references U-Net but does not provide layer-by-layer details (depth, channel widths, kernel sizes).
4. **Number of diffusion steps T:** The paper mentions the cosine beta schedule but does not explicitly state T (the total number of diffusion steps).
5. **Which TON_IoT subset:** TON_IoT has multiple subsets (Network traffic, Windows OS logs, IoT Telemetry). The paper does not explicitly specify which subset was used, though context implies the IoT network traffic subset.
6. **Full feature list of DS2OS used:** Whether all 13 features were used or whether Timestamp/Source ID were dropped.
7. **Training wall-clock time for diffusion model:** "Time-consuming" is stated qualitatively but specific training time is not given.
