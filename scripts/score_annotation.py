#!/usr/bin/env python3
"""
Score human annotations: compute inter-annotator agreement (Cohen's kappa),
identify disagreements for adjudication, and score the heuristic against
the adjudicated gold standard.

Usage:
    # After both annotators have filled in their human_label columns:
    python scripts/score_annotation.py

    # After adjudicating disagreements (filling final_label in disagreements.csv):
    python scripts/score_annotation.py
"""

import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, confusion_matrix,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANNOTATION_DIR = os.path.join(PROJECT_ROOT, "results", "annotation")
MASTER_KEY = os.path.join(ANNOTATION_DIR, "master_key.csv")
DISAGREEMENTS_CSV = os.path.join(ANNOTATION_DIR, "disagreements.csv")
OUTPUT_JSON = os.path.join(PROJECT_ROOT, "results", "manual_validation.json")


def load_annotator_files():
    """Find and load all annotation_*.csv files."""
    files = sorted([
        f for f in os.listdir(ANNOTATION_DIR)
        if f.startswith("annotation_") and f.endswith(".csv")
    ])
    if len(files) < 2:
        print(f"Error: need at least 2 annotation files in {ANNOTATION_DIR}")
        sys.exit(1)

    annotators = {}
    for fname in files:
        name = fname.replace("annotation_", "").replace(".csv", "")
        df = pd.read_csv(os.path.join(ANNOTATION_DIR, fname))
        if "human_label" not in df.columns:
            print(f"Error: {fname} missing 'human_label' column")
            sys.exit(1)
        # Drop rows where annotator hasn't labeled yet
        labeled = df[df["human_label"].notna() & (df["human_label"] != "")]
        labeled = labeled.copy()
        labeled["human_label"] = labeled["human_label"].astype(int)
        annotators[name] = labeled.set_index("row_id")["human_label"]
        print(f"  {name}: {len(labeled)} rows labeled")

    return annotators


def compute_agreement(annotators):
    """Compute pairwise Cohen's kappa and percent agreement."""
    names = list(annotators.keys())
    a1_name, a2_name = names[0], names[1]
    a1, a2 = annotators[a1_name], annotators[a2_name]

    # Align on shared row_ids
    shared = a1.index.intersection(a2.index)
    if len(shared) == 0:
        print("Error: no overlapping row_ids between annotators")
        sys.exit(1)

    y1 = a1.loc[shared].values
    y2 = a2.loc[shared].values

    kappa = cohen_kappa_score(y1, y2)
    pct_agree = np.mean(y1 == y2)
    cm = confusion_matrix(y1, y2, labels=[0, 1])

    print(f"\n{'='*50}")
    print(f"INTER-ANNOTATOR AGREEMENT ({a1_name} vs {a2_name})")
    print(f"{'='*50}")
    print(f"  Rows compared:      {len(shared)}")
    print(f"  Percent agreement:  {pct_agree:.3f}")
    print(f"  Cohen's kappa:      {kappa:.3f}")
    print()

    # Interpretation
    if kappa >= 0.8:
        reading = "almost perfect"
    elif kappa >= 0.6:
        reading = "substantial"
    elif kappa >= 0.4:
        reading = "moderate"
    elif kappa >= 0.2:
        reading = "fair"
    else:
        reading = "slight"
    print(f"  Interpretation:     {reading} agreement")
    print()
    print(f"  Confusion matrix ({a1_name} rows vs {a2_name} columns):")
    print(f"                      {a2_name}=0   {a2_name}=1")
    print(f"    {a1_name}=0        {cm[0,0]:>5d}     {cm[0,1]:>5d}")
    print(f"    {a1_name}=1        {cm[1,0]:>5d}     {cm[1,1]:>5d}")

    # Identify disagreements
    disagree_ids = shared[y1 != y2]
    return {
        "kappa": kappa,
        "percent_agreement": pct_agree,
        "reading": reading,
        "n_compared": len(shared),
        "n_disagreements": len(disagree_ids),
        "annotator_1": a1_name,
        "annotator_2": a2_name,
    }, disagree_ids, a1_name, a2_name


def write_disagreements(disagree_ids, annotators, master_key_df, a1_name, a2_name):
    """Write disagreements CSV for adjudication."""
    rows = []
    for rid in disagree_ids:
        text = master_key_df.loc[master_key_df["row_id"] == rid, "text"].values[0]
        rows.append({
            "row_id": rid,
            "text": text,
            f"label_{a1_name}": annotators[a1_name].loc[rid],
            f"label_{a2_name}": annotators[a2_name].loc[rid],
            "final_label": "",  # to be filled during adjudication
        })
    dis_df = pd.DataFrame(rows)
    dis_df.to_csv(DISAGREEMENTS_CSV, index=False)
    print(f"\n  Wrote {len(dis_df)} disagreements → {DISAGREEMENTS_CSV}")
    print("  Fill the 'final_label' column together, then re-run this script.")
    return dis_df


def build_gold_and_score(annotators, master_key_df, a1_name, a2_name):
    """
    Build gold standard from agreed labels + adjudicated disagreements,
    then score the heuristic against it.
    """
    a1, a2 = annotators[a1_name], annotators[a2_name]
    shared = a1.index.intersection(a2.index)

    # Start with agreements
    agreed_mask = a1.loc[shared] == a2.loc[shared]
    gold = a1.loc[shared][agreed_mask].copy()

    # Load adjudicated disagreements if available
    if os.path.exists(DISAGREEMENTS_CSV):
        dis_df = pd.read_csv(DISAGREEMENTS_CSV)
        adjudicated = dis_df[
            dis_df["final_label"].notna() & (dis_df["final_label"] != "")
        ]
        if len(adjudicated) > 0:
            adj_labels = adjudicated.set_index("row_id")["final_label"].astype(int)
            gold = pd.concat([gold, adj_labels])
            print(f"\n  Adjudicated labels loaded: {len(adjudicated)}")
        else:
            print("\n  No adjudicated labels yet — scoring against agreements only.")
    else:
        print("\n  No disagreements.csv found — scoring against agreements only.")

    # Align with master key to get heuristic labels
    mk = master_key_df.set_index("row_id")
    scored_ids = gold.index.intersection(mk.index)
    y_gold = gold.loc[scored_ids].values
    y_heuristic = mk.loc[scored_ids, "label"].values

    acc = accuracy_score(y_gold, y_heuristic)
    prec = precision_score(y_gold, y_heuristic, zero_division=0)
    rec = recall_score(y_gold, y_heuristic, zero_division=0)
    f1 = f1_score(y_gold, y_heuristic, zero_division=0)

    print(f"\n{'='*50}")
    print("HEURISTIC vs GOLD STANDARD")
    print(f"{'='*50}")
    print(f"  Gold standard size:  {len(scored_ids)}")
    print(f"  Accuracy:            {acc:.3f}")
    print(f"  Precision:           {prec:.3f}")
    print(f"  Recall:              {rec:.3f}")
    print(f"  F1-Score:            {f1:.3f}")

    cm = confusion_matrix(y_gold, y_heuristic, labels=[0, 1])
    print()
    print("  Confusion matrix (gold rows vs heuristic columns):")
    print("                      heuristic=0   heuristic=1")
    print(f"    gold=0              {cm[0,0]:>5d}         {cm[0,1]:>5d}")
    print(f"    gold=1              {cm[1,0]:>5d}         {cm[1,1]:>5d}")

    # Find illustrative disagreements for the report
    examples = []
    for rid in scored_ids:
        g = int(gold.loc[rid])
        h = int(mk.loc[rid, "label"])
        if g != h:
            text = mk.loc[rid, "text"]
            examples.append({
                "row_id": int(rid),
                "text": text,
                "gold_label": g,
                "heuristic_label": h,
                "type": "false_positive" if h == 1 else "false_negative",
            })

    return {
        "gold_standard_size": int(len(scored_ids)),
        "accuracy": round(acc, 3),
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3),
        "example_disagreements": examples[:10],
    }


def main():
    if not os.path.exists(MASTER_KEY):
        print(f"Error: {MASTER_KEY} not found. Run prepare_annotation.py first.")
        sys.exit(1)

    master_key_df = pd.read_csv(MASTER_KEY)
    print(f"Loaded master key: {len(master_key_df)} rows")

    annotators = load_annotator_files()
    names = list(annotators.keys())

    if len(names) < 2:
        print("Need at least 2 annotators.")
        sys.exit(1)

    # Step 1: Inter-annotator agreement
    agreement_stats, disagree_ids, a1_name, a2_name = compute_agreement(annotators)

    # Step 2: Write disagreements for adjudication (if any)
    if len(disagree_ids) > 0:
        write_disagreements(
            disagree_ids, annotators, master_key_df, a1_name, a2_name
        )

    # Step 3: Score heuristic against gold standard
    heuristic_stats = build_gold_and_score(
        annotators, master_key_df, a1_name, a2_name
    )

    # Save everything
    output = {
        "inter_annotator_agreement": agreement_stats,
        "heuristic_vs_gold": heuristic_stats,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Full results saved → {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
