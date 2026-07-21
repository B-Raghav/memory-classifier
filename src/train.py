"""Train and compare 5 classifiers with stratified k-fold CV.

Models: Logistic Regression, k-NN, Decision Tree, Random Forest, AdaBoost.
Handles class imbalance with SMOTE (inside the CV pipeline, so no leakage)
and reports accuracy, precision, recall, F1, ROC-AUC, and PR-AUC.

Also evaluates the similarity-only baseline (rank memories by cosine
similarity to the query; classify top-K as important) and runs a PCA
dimensionality analysis. Grouped CV by conversation prevents memories from
the same conversation appearing in both train and test folds.
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import config
from .features import FEATURE_COLUMNS


def get_models():
    rs = config.RANDOM_STATE
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000,
                                                 class_weight="balanced",
                                                 random_state=rs),
        "kNN": KNeighborsClassifier(n_neighbors=15),
        "DecisionTree": DecisionTreeClassifier(max_depth=8,
                                               class_weight="balanced",
                                               random_state=rs),
        "RandomForest": RandomForestClassifier(n_estimators=300,
                                               class_weight="balanced",
                                               random_state=rs, n_jobs=-1),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=rs),
    }


def _make_pipeline(model):
    steps = [("scaler", StandardScaler())]
    if config.USE_SMOTE:
        steps.append(("smote", SMOTE(random_state=config.RANDOM_STATE)))
    steps.append(("clf", model))
    return ImbPipeline(steps)


def _scores(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan"),
        "pr_auc": average_precision_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan"),
    }


def cross_validate(df):
    X = df[FEATURE_COLUMNS].values
    y = df["label"].values
    groups = df["conversation_id"].values
    cv = StratifiedGroupKFold(n_splits=config.N_FOLDS, shuffle=True,
                              random_state=config.RANDOM_STATE)

    all_results, oof = {}, {}
    for name, model in get_models().items():
        fold_scores, y_prob_oof = [], np.zeros(len(y))
        for tr, te in cv.split(X, y, groups):
            pipe = _make_pipeline(model)
            pipe.fit(X[tr], y[tr])
            prob = pipe.predict_proba(X[te])[:, 1]
            pred = (prob >= 0.5).astype(int)
            y_prob_oof[te] = prob
            fold_scores.append(_scores(y[te], pred, prob))
        mean = {k: float(np.mean([f[k] for f in fold_scores]))
                for k in fold_scores[0]}
        std = {k + "_std": float(np.std([f[k] for f in fold_scores]))
               for k in fold_scores[0]}
        all_results[name] = {**mean, **std}
        oof[name] = y_prob_oof
        print(f"[train] {name:<18} "
              f"F1={mean['f1']:.3f}  ROC-AUC={mean['roc_auc']:.3f}  "
              f"PR-AUC={mean['pr_auc']:.3f}")

    # --- similarity-only baseline: top-K per conversation --------------
    base_pred = np.zeros(len(df), dtype=int)
    for _, idx in df.groupby("conversation_id").groups.items():
        idx = list(idx)
        sims = df.loc[idx, "semantic_similarity"]
        top = sims.nlargest(min(config.TOP_K_BASELINE, len(idx))).index
        base_pred[[df.index.get_loc(i) for i in top]] = 1
    all_results["Baseline_TopK_Similarity"] = _scores(
        y, base_pred, df["semantic_similarity"].values)
    b = all_results["Baseline_TopK_Similarity"]
    print(f"[train] {'TopK-Similarity':<18} "
          f"F1={b['f1']:.3f}  ROC-AUC={b['roc_auc']:.3f}  "
          f"PR-AUC={b['pr_auc']:.3f}")

    return all_results, oof


def pca_analysis(df):
    X = StandardScaler().fit_transform(df[FEATURE_COLUMNS].values)
    pca = PCA().fit(X)
    evr = pca.explained_variance_ratio_
    print("[pca] explained variance ratio:",
          np.round(evr, 3).tolist())
    return evr


def fit_final_model(df, model_name):
    """Fit the best model on all data and persist it for the LLM demo."""
    pipe = _make_pipeline(get_models()[model_name])
    pipe.fit(df[FEATURE_COLUMNS].values, df["label"].values)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, "best_model.pkl")
    with open(path, "wb") as f:
        pickle.dump({"model_name": model_name, "pipeline": pipe,
                     "features": FEATURE_COLUMNS}, f)
    print(f"[train] saved final model ({model_name}) -> {path}")
    return path


def save_results(results, evr):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, "metrics.json")
    with open(path, "w") as f:
        json.dump({"cv_results": results,
                   "pca_explained_variance_ratio": list(map(float, evr))},
                  f, indent=2)
    pd.DataFrame(results).T.to_csv(
        os.path.join(config.RESULTS_DIR, "metrics.csv"))
    print(f"[train] metrics saved -> {path}")
