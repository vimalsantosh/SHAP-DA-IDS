"""
07_generate_figures.py — Publication-Quality Figure Generation
===============================================================
Generates all figures for the research paper:
  1.png  — Dataset class distribution
  2.png  — SHAP-DA-IDS pipeline architecture
  3.png  — Model comparison (macro F1)
  4.png  — Confusion matrix (best model: SHAP-DA-IDS)
  5.png  — Per-class F1 comparison (all models, minority classes)
  6.png  — SHAP global feature importance
  7.png  — Augmentation strategy comparison (fixed vs adaptive)

All figures saved to results/figures/ at 300 DPI.
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
matplotlib.use("Agg")  # Non-interactive backend for file output
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR = r"c:\Users\vimal\OneDrive\Desktop\7_sem\CNS\paper"
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
IMAGES_DIR = os.path.join(BASE_DIR, "research_project", "images")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# Publication style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLORS = {
    "primary": "#2196F3",
    "secondary": "#FF5722",
    "accent": "#4CAF50",
    "dark": "#212121",
    "gray": "#9E9E9E",
    "light": "#F5F5F5",
    "palette": sns.color_palette("Set2", 8),
}


def save_fig(fig, number, name):
    """Save figure to both figures/ and images/ directories."""
    fname = f"{number}.png"
    for d in [FIGURES_DIR, IMAGES_DIR]:
        fig.savefig(os.path.join(d, fname), dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  ✓ Saved {fname} ({name})")


def fig1_class_distribution():
    """Figure 1: Dataset class distribution (imbalance visualisation)."""
    class_data = {
        "Normal": 347935, "DoS": 5780, "Scan": 1547,
        "Malicious\nControl": 889, "Malicious\nOperation": 805,
        "Spying": 532, "Data\nProbing": 342, "Wrong\nSetup": 122,
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    labels = list(class_data.keys())
    counts = list(class_data.values())
    colors = [COLORS["primary"]] + [COLORS["secondary"]] * 7
    
    # Main bar chart (all classes)
    bars = ax1.bar(labels, counts, color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_title("DS2OS Dataset Class Distribution", fontweight="bold", pad=10)
    ax1.set_ylabel("Number of Samples")
    ax1.set_xlabel("Class")
    ax1.tick_params(axis="x", rotation=0)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                 f"{count:,}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    
    # Add legend
    normal_patch = mpatches.Patch(color=COLORS["primary"], label="Normal Traffic")
    attack_patch = mpatches.Patch(color=COLORS["secondary"], label="Attack Classes (7)")
    ax1.legend(handles=[normal_patch, attack_patch], loc="upper right")
    ax1.set_yscale("log")
    ax1.set_ylim(1, 1e6)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    
    # Zoom: minority classes only
    minority_labels = labels[1:]
    minority_counts = counts[1:]
    bars2 = ax2.bar(minority_labels, minority_counts,
                    color=COLORS["secondary"], edgecolor="white", linewidth=0.5)
    ax2.set_title("Minority Attack Classes (Zoomed)", fontweight="bold", pad=10)
    ax2.set_ylabel("Number of Samples")
    ax2.set_xlabel("Attack Class")
    
    for bar, count in zip(bars2, minority_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 f"{count:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    
    # Annotate imbalance ratios
    for bar, count in zip(bars2, minority_counts):
        ratio = 347935 / count
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() / 2,
                 f"{ratio:.0f}x", ha="center", va="center",
                 fontsize=8, color="white", fontweight="bold")
    
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    
    fig.suptitle("DS2OS Intrusion Detection Dataset: Class Imbalance Analysis",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig(fig, 1, "Class Distribution")


def fig2_pipeline_architecture():
    """Figure 2: SHAP-DA-IDS pipeline architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    
    # Title
    ax.text(7, 7.6, "SHAP-DA-IDS: Proposed Pipeline Architecture",
            ha="center", va="center", fontsize=14, fontweight="bold")
    
    # Define boxes
    boxes = [
        # (x, y, width, height, label, color, sublabel)
        (0.3, 5.5, 2.0, 1.2, "Imbalanced\nIoT Dataset\n(DS2OS)", "#90CAF9", 
         "357,952 samples\n8 classes, IR up to 2851:1"),
        (2.9, 5.5, 2.4, 1.2, "Phase 1\nSHAP Feature\nSelection", "#A5D6A7",
         "RF → SHAP values\nSelect top-k features"),
        (5.9, 5.5, 2.4, 1.2, "Phase 2\nDifficulty\nEstimation", "#FFCC80",
         "Per-class F1 from RF\ndifficulty = 1 − F1"),
        (8.9, 5.5, 2.4, 1.2, "Phase 3\nAdaptive\nAugmentation", "#F48FB1",
         "SMOTE target ∝ difficulty\nHarder class → more samples"),
        (11.4, 5.5, 2.3, 1.2, "Phase 4\nClassification\n(XGBoost)", "#CE93D8",
         "Train on balanced\nselected-feature data"),
    ]
    
    for (x, y, w, h, label, color, sublabel) in boxes:
        # Main box
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="gray",
                              linewidth=1.5, zorder=2, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.65, label, ha="center", va="center",
                fontsize=9, fontweight="bold", zorder=3, wrap=True)
        # Sub-label box
        sub_rect = plt.Rectangle((x + 0.05, y - 0.95), w - 0.1, 0.85,
                                  facecolor="white", edgecolor=color,
                                  linewidth=1, zorder=2, alpha=0.8)
        ax.add_patch(sub_rect)
        ax.text(x + w/2, y - 0.52, sublabel, ha="center", va="center",
                fontsize=7.5, color="#333333", zorder=3)
    
    # Arrows between boxes
    arrow_props = dict(arrowstyle="->", color="#424242", lw=1.5)
    arrow_positions = [
        (2.3, 6.1, 2.9, 6.1),
        (5.3, 6.1, 5.9, 6.1),
        (8.3, 6.1, 8.9, 6.1),
        (11.3, 6.1, 11.4, 6.1),
    ]
    for (x1, y1, x2, y2) in arrow_positions:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=arrow_props, zorder=4)
    
    # Explainability box
    exp_box = plt.Rectangle((8.9, 2.8), 4.8, 1.2, facecolor="#B2DFDB",
                             edgecolor="gray", linewidth=1.5, zorder=2, alpha=0.9)
    ax.add_patch(exp_box)
    ax.text(11.3, 3.4, "Phase 5: SHAP Explainability\n(TreeExplainer → feature attributions)",
            ha="center", va="center", fontsize=9, fontweight="bold", zorder=3)
    
    # Arrow from Phase 4 to Explainability
    ax.annotate("", xy=(13.0, 4.0), xytext=(13.0, 5.5),
                arrowprops=dict(arrowstyle="->", color="#424242", lw=1.5), zorder=4)
    
    # Outputs
    outputs = [
        (0.3, 2.5, 2.4, 1.0, "Accuracy, Macro F1\nPrecision, Recall\nAUC-ROC", "#E0E0E0"),
        (3.0, 2.5, 2.4, 1.0, "Per-class F1\nConfusion Matrix\n(minority focus)", "#E0E0E0"),
        (5.7, 2.5, 2.4, 1.0, "SHAP Global &\nPer-class Feature\nImportance", "#E0E0E0"),
    ]
    ax.text(1.5, 1.6, "Outputs & Evaluation Metrics",
            ha="center", va="center", fontsize=10, fontweight="bold", color="#333333")
    
    for (x, y, w, h, label, color) in outputs:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#999",
                              linewidth=1, zorder=2, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=8, zorder=3)
    
    # Novelty annotations
    novelty_x = [3.9, 7.1, 10.1]
    novelty_labels = ["★ Novel:\nSHAP-guided\nfeature selection", 
                      "★ Novel:\nDifficulty-adaptive\naugmentation target",
                      "★ Novel:\nExplainable\npredictions"]
    for nx, nl in zip(novelty_x, novelty_labels):
        ax.text(nx, 7.1, nl, ha="center", va="center", fontsize=7.5,
                color="#D32F2F", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFEBEE",
                          edgecolor="#D32F2F", alpha=0.8))
    
    plt.tight_layout()
    save_fig(fig, 2, "Pipeline Architecture")


def fig3_model_comparison():
    """Figure 3: Model performance comparison (macro F1 bar chart)."""
    comp_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    if not os.path.exists(comp_path):
        print("  Skipping Figure 3 (model_comparison.csv not found)")
        return
    
    df = pd.read_csv(comp_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    models = df["model"].tolist()
    colors_bars = [COLORS["accent"] if "SHAP-DA-IDS" in m 
                   else (COLORS["primary"] if "SMOTE Fixed" in m 
                         else COLORS["gray"]) for m in models]
    
    # Macro F1
    ax1 = axes[0]
    bars = ax1.barh(models, df["macro_f1"], color=colors_bars, edgecolor="white", height=0.6)
    ax1.set_xlabel("Macro F1 Score (%)")
    ax1.set_title("Macro F1 Score Comparison\n(Equal weight to all classes)", fontweight="bold")
    ax1.set_xlim(0, 110)
    for bar, v in zip(bars, df["macro_f1"]):
        ax1.text(v + 0.5, bar.get_y() + bar.get_height()/2,
                 f"{v:.2f}%", va="center", fontsize=9, fontweight="bold")
    ax1.grid(axis="x", alpha=0.3, linestyle="--")
    
    # Accuracy
    ax2 = axes[1]
    bars2 = ax2.barh(models, df["accuracy"], color=colors_bars, edgecolor="white", height=0.6)
    ax2.set_xlabel("Accuracy (%)")
    ax2.set_title("Overall Accuracy Comparison", fontweight="bold")
    ax2.set_xlim(80, 102)
    for bar, v in zip(bars2, df["accuracy"]):
        ax2.text(v + 0.1, bar.get_y() + bar.get_height()/2,
                 f"{v:.2f}%", va="center", fontsize=9, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3, linestyle="--")
    
    # Legend
    patches = [
        mpatches.Patch(color=COLORS["accent"], label="SHAP-DA-IDS (Proposed)"),
        mpatches.Patch(color=COLORS["primary"], label="SMOTE Fixed (DIFT-style)"),
        mpatches.Patch(color=COLORS["gray"], label="Baseline Methods"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
    
    fig.suptitle("Model Performance Comparison on DS2OS Dataset",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, 3, "Model Comparison")


def fig4_confusion_matrix():
    """Figure 4: Confusion matrix for best model."""
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.csv")
    if not os.path.exists(cm_path):
        print("  Skipping Figure 4 (confusion_matrix.csv not found)")
        return
    
    cm_df = pd.read_csv(cm_path, index_col=0)
    class_names = list(cm_df.index)
    cm = cm_df.values
    
    # Normalize
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], linewidths=0.5)
    axes[0].set_title("Confusion Matrix (Raw Counts)", fontweight="bold")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].tick_params(axis="y", rotation=0)
    
    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], linewidths=0.5, vmin=0, vmax=1)
    axes[1].set_title("Confusion Matrix (Normalised)", fontweight="bold")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].tick_params(axis="y", rotation=0)
    
    fig.suptitle("SHAP-DA-IDS Confusion Matrix on DS2OS Dataset",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, 4, "Confusion Matrix")


def fig5_per_class_f1():
    """Figure 5: Per-class F1 comparison (focus on minority classes)."""
    pc_path = os.path.join(RESULTS_DIR, "per_class_metrics.csv")
    if not os.path.exists(pc_path):
        print("  Skipping Figure 5 (per_class_metrics.csv not found)")
        return
    
    df = pd.read_csv(pc_path)
    
    # Focus on minority classes (all except Normal)
    minority_classes = [c for c in df["class"].unique() if c != "Normal"]
    
    # Pivot for plotting
    pivot = df[df["class"].isin(minority_classes)].pivot(
        index="class", columns="model", values="f1"
    )
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    n_models = len(pivot.columns)
    n_classes = len(pivot.index)
    x = np.arange(n_classes)
    width = 0.12
    
    colors_per_model = {
        "Logistic Regression": "#90A4AE",
        "Random Forest": "#78909C",
        "XGBoost": "#546E7A",
        "XGBoost + Class Weights": "#42A5F5",
        "XGBoost + SMOTE Fixed (DIFT-style)": "#2196F3",
        "SHAP-DA-IDS (Ours)": "#F44336",
    }
    
    for i, model_name in enumerate(pivot.columns):
        values = pivot[model_name].values
        offset = (i - n_models/2 + 0.5) * width
        color = colors_per_model.get(model_name, f"C{i}")
        bars = ax.bar(x + offset, values, width, label=model_name,
                      color=color, alpha=0.9, edgecolor="white")
    
    ax.set_xlabel("Attack Class")
    ax.set_ylabel("F1 Score (%)")
    ax.set_title("Per-Class F1 Score Comparison\n(Minority Attack Classes Only)",
                 fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=20, ha="right")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0, 115)
    
    plt.tight_layout()
    save_fig(fig, 5, "Per-class F1")


def fig6_shap_importance():
    """Figure 6: SHAP global feature importance."""
    shap_path = os.path.join(RESULTS_DIR, "shap_global_importance.csv")
    if not os.path.exists(shap_path):
        print("  Skipping Figure 6 (shap_global_importance.csv not found)")
        return
    
    df = pd.read_csv(shap_path).head(10)  # Top 10
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = [COLORS["accent"]] * len(df)
    bars = ax.barh(df["feature"][::-1], df["mean_abs_shap"][::-1],
                   color=colors, edgecolor="white", height=0.6)
    
    ax.set_xlabel("Mean |SHAP Value| (Feature Impact)")
    ax.set_title("Global SHAP Feature Importance\n(SHAP-DA-IDS / XGBoost on DS2OS)",
                 fontweight="bold")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    
    for bar, v in zip(bars, df["mean_abs_shap"][::-1]):
        ax.text(v + max(df["mean_abs_shap"]) * 0.01,
                bar.get_y() + bar.get_height()/2,
                f"{v:.4f}", va="center", fontsize=9)
    
    ax.text(0.98, 0.02,
            "Higher = More important for classification\n"
            "SHAP values are model-agnostic attributions",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, style="italic", color=COLORS["gray"])
    
    plt.tight_layout()
    save_fig(fig, 6, "SHAP Feature Importance")


def fig7_augmentation_strategy():
    """Figure 7: Augmentation strategy comparison (fixed vs adaptive)."""
    strat_path = os.path.join(RESULTS_DIR, "augmentation_strategy_comparison.csv")
    if not os.path.exists(strat_path):
        print("  Skipping Figure 7 (augmentation_strategy_comparison.csv not found)")
        return
    
    df = pd.read_csv(strat_path)
    
    # Exclude majority class (Normal) for clarity
    df_min = df[df["class"] != "Normal"].copy()
    df_min = df_min.sort_values("original_count", ascending=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(df_min))
    width = 0.25
    
    # Left: sample counts comparison
    ax1 = axes[0]
    ax1.bar(x - width, df_min["original_count"], width, label="Original Count",
            color="#90A4AE", edgecolor="white")
    ax1.bar(x, df_min["dift_target"], width, label="DIFT Fixed Target",
            color="#42A5F5", edgecolor="white")
    ax1.bar(x + width, df_min["adaptive_target"], width, label="SHAP-DA-IDS Adaptive Target",
            color="#F44336", edgecolor="white", alpha=0.9)
    
    ax1.set_xlabel("Attack Class")
    ax1.set_ylabel("Number of Samples")
    ax1.set_title("Augmentation Target Comparison\n(DIFT Fixed vs. Adaptive)", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_min["class"], rotation=20, ha="right")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    
    # Right: preliminary F1 and difficulty scores
    ax2 = axes[1]
    colors2 = [COLORS["secondary"] if f < 0.7 else COLORS["accent"]
                for f in df_min["preliminary_f1"]]
    bars = ax2.bar(x, df_min["preliminary_f1"] * 100, color=colors2, edgecolor="white")
    ax2.set_xlabel("Attack Class")
    ax2.set_ylabel("Preliminary Classifier F1 (%)")
    ax2.set_title("Preliminary F1 → Difficulty Score\n(Lower F1 = Harder class = More augmentation)",
                  fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_min["class"], rotation=20, ha="right")
    
    # Difficulty score annotation
    for bar, f1 in zip(bars, df_min["preliminary_f1"]):
        diff = 1 - f1
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5,
                 f"D={diff:.2f}", ha="center", fontsize=8, fontweight="bold",
                 color=COLORS["secondary"])
    
    ax2.set_ylim(0, 115)
    ax2.axhline(y=70, color="red", linestyle="--", alpha=0.5, label="Difficulty threshold (F1<70%)")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    
    patches = [
        mpatches.Patch(color=COLORS["secondary"], label="High difficulty (F1 < 70%)"),
        mpatches.Patch(color=COLORS["accent"], label="Low difficulty (F1 ≥ 70%)"),
    ]
    ax2.legend(handles=patches + [plt.Line2D([0], [0], color="red", linestyle="--",
                                              label="F1 = 70% threshold")],
               fontsize=8)
    
    fig.suptitle("Difficulty-Adaptive Augmentation: Core Novelty of SHAP-DA-IDS",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, 7, "Augmentation Strategy Comparison")


def main():
    print("Generating publication-quality figures...")
    print(f"Output directory: {FIGURES_DIR}")
    
    fig1_class_distribution()
    fig2_pipeline_architecture()
    fig3_model_comparison()
    fig4_confusion_matrix()
    fig5_per_class_f1()
    fig6_shap_importance()
    fig7_augmentation_strategy()
    
    print("\n✓ All figures generated.")
    print(f"  Saved to: {FIGURES_DIR}")
    print(f"  Copied to: {IMAGES_DIR}")


if __name__ == "__main__":
    main()
