# Overleaf Research Paper Project

## Title
**SHAP-DA-IDS: Difficulty-Adaptive Augmentation with Explainable Feature Selection for Imbalanced IoT Intrusion Detection**

## Format
IEEE Transactions format using `IEEEtran.cls`.

---

## Directory Structure
```
research_project/
├── main.tex                       # Root entry point for Overleaf
├── IEEEtran.cls                   # Official IEEE Transactions LaTeX document class
├── images/                        # All high-resolution 300 DPI figures
│   ├── 1.png                      # Dataset class distribution and imbalance analysis
│   ├── 2.png                      # Proposed SHAP-DA-IDS pipeline architecture
│   ├── 3.png                      # Overall model performance comparison (Macro F1 & Acc)
│   ├── 4.png                      # SHAP-DA-IDS confusion matrix (raw counts & normalized)
│   ├── 5.png                      # Per-class F1-score comparison on minority attack classes
│   ├── 6.png                      # Global SHAP feature importance attribution
│   └── 7.png                      # Augmentation strategy comparison (fixed vs. adaptive)
├── tex/                           # Modular section LaTeX documents
│   ├── main.tex                   # Subdirectory entry point
│   ├── introduction.tex           # Section I: Introduction & Contribution Table
│   ├── related_work.tex           # Section II: Literature review & gap analysis
│   ├── methodology.tex            # Section III: Mathematical formulation of SHAP-DA-IDS
│   ├── experiments.tex            # Section IV: Dataset, cleaning, baselines, metrics
│   ├── results.tex                # Section V: Empirical results & SHAP analysis
│   └── conclusion.tex             # Section VI: Limitations & Future work
└── references/
    └── references.bib             # Complete BibTeX bibliography with real DOIs
```

---

## How to Open in Overleaf
1. Download `Overleaf_Project.zip` from the project directory.
2. In Overleaf, click **New Project** $\rightarrow$ **Upload Project**.
3. Select `Overleaf_Project.zip`.
4. Ensure the main document is set to `main.tex`.
5. Click **Recompile** with the default **pdfLaTeX** compiler.
