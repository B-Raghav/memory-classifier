import os
import sys
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

# Add root directory to path
sys.path.insert(0, os.path.abspath('.'))
from src import config
from src.train import get_models, _scores, _make_pipeline

def run_ablation():
    features_csv = os.path.join(config.DATA_DIR, "features.csv")
    if not os.path.exists(features_csv):
        print("Error: features.csv not found. Please run the pipeline first.")
        return

    df = pd.read_csv(features_csv)
    print(f"Loaded {len(df)} memories from features.csv")

    # Define the ablated feature set (all features except semantic_similarity)
    ablated_features = [
        "recency", "session_gap", "access_frequency",
        "role_user", "sentence_length", "is_question", "entity_count",
        "first_person", "position_in_session"
    ]
    print(f"Running ablation study on {len(ablated_features)} features (dropped semantic_similarity):")
    print(ablated_features)

    X = df[ablated_features].values
    y = df["label"].values
    groups = df["conversation_id"].values
    cv = StratifiedGroupKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    results = {}
    for name, model in get_models().items():
        fold_scores = []
        for tr, te in cv.split(X, y, groups):
            pipe = _make_pipeline(model)
            pipe.fit(X[tr], y[tr])
            prob = pipe.predict_proba(X[te])[:, 1]
            pred = (prob >= 0.5).astype(int)
            fold_scores.append(_scores(y[te], pred, prob))
        
        mean = {k: float(np.mean([f[k] for f in fold_scores])) for k in fold_scores[0]}
        results[name] = mean
        print(f"[ablation] {name:<18} F1={mean['f1']:.3f}  ROC-AUC={mean['roc_auc']:.3f}  PR-AUC={mean['pr_auc']:.3f}")

    # Also compute similarity-only baseline for reference
    # similarity baseline cannot be run without semantic_similarity, but we can display the original baseline
    print("\nComparison with original pipeline (with semantic similarity):")
    print("AdaBoost (Original): F1=0.557, ROC-AUC=0.827, PR-AUC=0.589")
    print("LogisticRegression (Original): F1=0.553, ROC-AUC=0.812, PR-AUC=0.525")
    print("Similarity-only Baseline: F1=0.334, ROC-AUC=0.623, PR-AUC=0.413")

if __name__ == "__main__":
    run_ablation()
