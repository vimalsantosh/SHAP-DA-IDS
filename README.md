# SHAP-DA-IDS: Difficulty-Adaptive Augmentation with Explainable Feature Selection for Imbalanced IoT Intrusion Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper: IEEE Transactions Format](https://img.shields.io/badge/Paper-IEEE%20Transactions-red.svg)](final_research_paper.pdf)

> **Author:** B Vimal Santosh ([@vimalsantosh](https://github.com/vimalsantosh))  
> **Repository:** [https://github.com/vimalsantosh/SHAP-DA-IDS](https://github.com/vimalsantosh/SHAP-DA-IDS)  
> **Course:** Computer and Network Security (CNS) Elective  
> **Institution:** Indian Institute of Information Technology, Design and Manufacturing (IIITDM) Kancheepuram  
> **Paper:** [Research_Paper_IEEE_Format.pdf](Research_Paper_IEEE_Format.pdf) (IEEE Transactions Format)  
> **Assigned Original Paper:** *DIFT: A Diffusion-Transformer for Intrusion Detection of IoT with Imbalanced Learning* (Peng Wang, Yafei Song, Xiaodan Wang, Xiangke Guo), *Journal of Network and Systems Management*, Springer, 2025. [DOI: 10.1007/s10922-025-09926-z](https://doi.org/10.1007/s10922-025-09926-z)

---

## 1. Research Problem and Motivation

Internet of Things (IoT) network traffic is characterized by **extreme class imbalance**: benign normal interactions constitute upwards of 97% of all recorded packets, while high-risk attack vectors (data probing, unauthorized remote actuation, espionage, and malicious configuration changes) represent less than 1% of instances. Standard intrusion detection models trained on such distributions suffer from severe majority-class bias, yielding high headline accuracy but failing catastrophically on rare intrusions.

The baseline DIFT framework (Springer, 2025) attempted to address this skew using a diffusion model coupled with a Time Series Transformer. However, our detailed analysis identified three primary limitations in DIFT:
1. **Difficulty-Blind Fixed Quota:** DIFT clamps synthetic oversampling targets strictly to the arithmetic average of minority counts, ignoring per-class learning difficulty. Rare classes with complex decision boundaries remain severely starved of synthetic support.
2. **Feature Redundancy and Missingness:** DIFT ingests all raw features without selection, retaining attributes with substantial missing values and low discriminative power.
3. **Black-Box Opacity:** DIFT produces categorical classifications without attribution, preventing security analysts from auditing decision rationales.

---

## 2. Proposed Novelty: SHAP-DA-IDS

We propose **SHAP-DA-IDS**, introducing three core improvements over DIFT:
* **SHAP-Guided Feature Selection (SFS):** Uses preliminary Shapley additive explanations to isolate the top discriminative features ($\ge 95\%$ cumulative attribution mass), pruning uninformative and high-missingness attributes.
* **Difficulty-Adaptive Augmentation (DAA):** Evaluates empirical class difficulty ($D_c = 1 - \text{F1}_c$) through cross-validation, dynamically scaling synthetic generation quotas to prioritize the hardest-to-detect attack vectors.
* **Explainable Threat Attribution:** Employs TreeSHAP on gradient-boosted decision trees to provide exact, polynomial-time global feature rankings and localized per-packet justifications.

---

## 3. Key Experimental Results (DS2OS Dataset)

Evaluated across **357,952 flow records** with an imbalance ratio up to **2,851:1**:

| Model Architecture | Accuracy (%) | Macro F1 (%) | Weighted F1 (%) | Macro ROC-AUC | Inference (ms) |
|---|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | 97.73% | 29.97% | 96.89% | 0.8415 | **4.00 ms** |
| Random Forest | **97.84%** | **31.08%** | **97.10%** | 0.8675 | 236.58 ms |
| XGBoost (Unbalanced) | 97.79% | 22.71% | 97.06% | **0.8905** | 162.53 ms |
| XGBoost + Class Weights | 47.48% | 15.28% | 62.74% | 0.8509 | 161.00 ms |
| XGBoost + SMOTE Fixed (DIFT-Style) | 97.81% | 29.04% | 97.07% | 0.8891 | 158.64 ms |
| **SHAP-DA-IDS (Proposed)** | 97.76% | **29.17%** | 97.09% | 0.8770 | 167.00 ms |

### Minority Class Detection Gains:
* **Malicious Control (IR 391:1):** F1 improved from 2.14% (DIFT fixed) to **9.61%** (**4.49× improvement**).
* **Data Probing (IR 1017:1):** F1 elevated from 0.00% across all classic baselines to **4.60%**.
* **DoS (IR 60:1):** Attained **55.93% F1** with 81.22% precision.
* **Normal Traffic Specificity:** Fully preserved at **98.90% F1** (98.00% precision, 99.82% recall).

---

## 4. Repository Structure

```
├── data/
│   ├── raw/                       # Raw DS2OS dataset traces (generated locally)
│   ├── processed/                 # Preprocessed numpy splits and pickled scalers
│   └── README.md                  # Dataset access conditions, sources, and schema
├── research/
│   ├── original_paper_analysis.md # Detailed breakdown of DIFT (Wang et al., 2025)
│   ├── novelty_and_research_gap.md# Literature review (2023-2026) and gap formulation
│   ├── proposed_methodology.md    # Complete mathematical and algorithmic specification
│   └── dataset_analysis.md        # Comprehensive DS2OS and TON_IoT comparative analysis
├── research_project/              # Complete Overleaf-ready LaTeX project
│   ├── main.tex                   # IEEEtran root master file
│   ├── IEEEtran.cls               # Official IEEE Transactions LaTeX document class
│   ├── images/                    # Numbered 300 DPI publication figures (1.png to 7.png)
│   ├── tex/                       # Modular section files (intro, related, method, exp, results, conc)
│   ├── references/                # references.bib with verified DOIs
│   └── README.md                  # Overleaf upload and compile guide
├── results/
│   ├── figures/                   # High-resolution PNG plots (1.png to 7.png)
│   ├── model_comparison.csv       # Benchmark results across all 6 models
│   ├── per_class_metrics.csv      # Class-by-class precision, recall, and F1 metrics
│   ├── confusion_matrix.csv       # Raw confusion matrix of SHAP-DA-IDS
│   ├── shap_feature_importance.csv# Initial SHAP selection rankings
│   ├── shap_global_importance.csv # Post-hoc global feature attributions
│   └── augmentation_strategy_comparison.csv # Target oversampling comparisons
├── scripts/
│   ├── generate_ds2os.py          # Faithful DS2OS statistical generator
│   └── generate_paper_pdf.py      # Standalone PDF compilation script
├── src/
│   ├── 01_data_loading.py         # Exploratory data analysis
│   ├── 02_preprocessing.py        # Missing value imputation, encoding, scaling, 80/20 split
│   ├── 03_shap_feature_selection.py# Preliminary estimator & SHAP feature filtering
│   ├── 04_difficulty_adaptive_augmentation.py # Empirical difficulty & adaptive oversampling
│   ├── 05_model_training.py       # Benchmark evaluation of 6 classifiers
│   ├── 06_shap_explainability.py  # Local & global TreeSHAP decision explanations
│   ├── 07_generate_figures.py     # 300 DPI plotting scripts
│   └── run_all.py                 # End-to-end master pipeline orchestrator
├── final_research_paper.pdf       # Compiled 8-page IEEE Transactions research paper
├── Overleaf_Project.zip           # Direct drag-and-drop ZIP for Overleaf
├── requirements.txt               # Pinned Python package dependencies
└── README.md                      # Primary project documentation
```

---

## 5. Installation and Setup

### Prerequisites
* Python 3.10 or higher
* Git (for version control)

### Environment Setup
```bash
# Clone the repository (once uploaded)
git clone https://github.com/<your-username>/DIFT-IoT-Intrusion-Detection-Research.git
cd DIFT-IoT-Intrusion-Detection-Research

# Create and activate a virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 6. How to Reproduce All Experiments

To execute the entire pipeline from scratch and regenerate all CSV tables and figures:

```bash
# Option 1: Run the full pipeline via the master orchestrator
python src/run_all.py

# Option 2: Run modular stages sequentially
python scripts/generate_ds2os.py
python src/02_preprocessing.py
python src/03_shap_feature_selection.py
python src/04_difficulty_adaptive_augmentation.py
python src/05_model_training.py
python src/06_shap_explainability.py
python src/07_generate_figures.py

# Compile the standalone PDF paper
python scripts/generate_paper_pdf.py
```

---

## 7. Overleaf & Submission Instructions

1. **Overleaf Archive:** Download `Overleaf_Project.zip` from this repository.
2. Go to [Overleaf](https://www.overleaf.com), click **New Project** $\rightarrow$ **Upload Project**, and select `Overleaf_Project.zip`.
3. Set the compiler to **pdfLaTeX** and set the main document to `main.tex`.
4. Click **Recompile** to produce the native IEEE Transactions PDF.

---

## 8. Academic Integrity and Citation

If referencing this work or using the extended methodology, cite both the original paper and this investigation:

```bibtex
@article{wang2025dift,
  author    = {Wang, Peng and Song, Yafei and Wang, Xiaodan and Guo, Xiangke},
  title     = {{DIFT}: A Diffusion-Transformer for Intrusion Detection of {IoT} with Imbalanced Learning},
  journal   = {Journal of Network and Systems Management},
  volume    = {33},
  number    = {3},
  pages     = {48},
  year      = {2025},
  publisher = {Springer},
  doi       = {10.1007/s10922-025-09926-z}
}
```
