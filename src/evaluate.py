"""Plots: model comparison bars, ROC curves, PR curves, feature importance,
PCA scree plot. All saved to results/."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from . import config
from .features import FEATURE_COLUMNS


def _save(fig, name):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] saved -> {path}")


def plot_model_comparison(results):
    names = [n for n in results if not n.startswith("Baseline")]
    metrics = ["f1", "roc_auc", "pr_auc"]
    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(metrics):
        vals = [results[n][m] for n in names]
        ax.bar(x + i * width, vals, width, label=m.upper())
    base = results.get("Baseline_TopK_Similarity")
    if base:
        ax.axhline(base["f1"], ls="--", c="gray",
                   label=f"Top-K similarity F1 ({base['f1']:.2f})")
    ax.set_xticks(x + width)
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylim(0, 1)
    ax.set_title("Classifier comparison (5-fold grouped CV)")
    ax.legend()
    _save(fig, "model_comparison.png")


def plot_curves(df, oof_probs):
    y = df["label"].values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for name, prob in oof_probs.items():
        fpr, tpr, _ = roc_curve(y, prob)
        ax1.plot(fpr, tpr, label=name)
        prec, rec, _ = precision_recall_curve(y, prob)
        ax2.plot(rec, prec, label=name)
    # similarity baseline
    fpr, tpr, _ = roc_curve(y, df["semantic_similarity"])
    ax1.plot(fpr, tpr, "k--", label="similarity only")
    prec, rec, _ = precision_recall_curve(y, df["semantic_similarity"])
    ax2.plot(rec, prec, "k--", label="similarity only")
    ax1.plot([0, 1], [0, 1], c="lightgray")
    ax1.set(title="ROC curves (out-of-fold)", xlabel="FPR", ylabel="TPR")
    ax2.set(title="Precision-Recall curves", xlabel="Recall", ylabel="Precision")
    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8)
    _save(fig, "roc_pr_curves.png")


def plot_feature_importance(df):
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=300,
                                random_state=config.RANDOM_STATE, n_jobs=-1)
    rf.fit(df[FEATURE_COLUMNS], df["label"])
    imp = rf.feature_importances_
    order = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(np.array(FEATURE_COLUMNS)[order], imp[order])
    ax.set_title("Random Forest feature importance")
    _save(fig, "feature_importance.png")


def plot_pca_scree(evr):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(evr) + 1), np.cumsum(evr), "o-")
    ax.set(title="PCA cumulative explained variance",
           xlabel="Components", ylabel="Cumulative variance")
    ax.axhline(0.95, ls="--", c="gray")
    _save(fig, "pca_scree.png")
