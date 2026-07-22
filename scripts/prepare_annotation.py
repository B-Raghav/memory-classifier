#!/usr/bin/env python3
"""
Prepare blinded, shuffled annotation files for human labeling.

Usage:
    python scripts/prepare_annotation.py --annotators raghav harsha

Creates:
    results/annotation/annotation_raghav.csv
    results/annotation/annotation_harsha.csv
    results/annotation/master_key.csv   (DO NOT open until scoring)

Each annotator file contains columns: row_id, text
The annotator adds a column: human_label (1 = retain, 0 = discard)

The master_key.csv maps row_id back to the original label and conversation_id,
so the scoring script can compare without the annotators ever seeing the
heuristic's answer.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

VALIDATION_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "label_validation_sample.csv",
)
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "annotation",
)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare blinded annotation CSVs for human labelers."
    )
    parser.add_argument(
        "--annotators", nargs="+", required=True,
        help="Names of the annotators (e.g. raghav harsha)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for shuffling (default: 42)",
    )
    args = parser.parse_args()

    if not os.path.exists(VALIDATION_CSV):
        print(f"Error: {VALIDATION_CSV} not found. Run the pipeline first.")
        sys.exit(1)

    df = pd.read_csv(VALIDATION_CSV)
    print(f"Loaded {len(df)} rows from {VALIDATION_CSV}")

    # Assign stable row IDs before shuffling
    df["row_id"] = np.arange(len(df))

    # Shuffle deterministically
    rng = np.random.RandomState(args.seed)
    shuffled_idx = rng.permutation(len(df))
    df_shuffled = df.iloc[shuffled_idx].reset_index(drop=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save master key (maps row_id → original label, conversation_id, text)
    # Annotators must NOT open this file.
    master_key = df_shuffled[["row_id", "text", "label", "conversation_id"]].copy()
    master_path = os.path.join(OUTPUT_DIR, "master_key.csv")
    master_key.to_csv(master_path, index=False)
    print(f"Saved master key → {master_path}")
    print("  ⚠  DO NOT open master_key.csv until you run score_annotation.py")

    # Create one blinded file per annotator
    # Only row_id and text — no label, no conversation_id
    blinded = df_shuffled[["row_id", "text"]].copy()
    blinded["human_label"] = ""  # annotator fills this in

    for name in args.annotators:
        out_path = os.path.join(OUTPUT_DIR, f"annotation_{name}.csv")
        blinded.to_csv(out_path, index=False)
        print(f"Saved annotation file → {out_path}")

    print()
    print("=" * 60)
    print("ANNOTATION GUIDELINE")
    print("=" * 60)
    print()
    print("For each row, read the 'text' column and fill 'human_label':")
    print()
    print("  1 (retain)  — The turn states something DURABLE about the")
    print("    speaker: name, age, location, job, family, pets, hobbies,")
    print("    dietary preferences, health, ongoing plans, strong stable")
    print("    preferences. Rule of thumb: 'Would I want this in my notes")
    print("    if I were talking to this person again next week?'")
    print()
    print("  0 (discard) — Backchannel/filler, a question, generic opinion")
    print("    with no personal anchor, or a transient state ('I'm tired")
    print("    today', 'the weather's nice').")
    print()
    print("EDGE CASES (decide these in advance during your pilot):")
    print("  • Transient vs durable: retain only if consequences last")
    print("    past this session.")
    print("  • Multi-fact turns: if ANY clause has a durable fact → 1.")
    print("  • Both speakers have personas — treat each speaker's own")
    print("    disclosures as retainable.")
    print("  • Third-party facts ('my sister is a nurse') → 1, it's")
    print("    stable relational context.")
    print()
    print("PROTOCOL:")
    print("  1. Pilot: both annotators label the FIRST 20 rows together,")
    print("     compare, discuss disagreements, refine the guideline.")
    print("  2. Label the remaining 180 rows independently. No peeking")
    print("     at each other's file or the master_key.")
    print("  3. Run: python scripts/score_annotation.py")
    print("  4. Adjudicate disagreements together, fill 'final_label'")
    print("     in results/annotation/disagreements.csv")
    print("  5. Run score_annotation.py again for final metrics.")
    print("=" * 60)


if __name__ == "__main__":
    main()
