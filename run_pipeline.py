"""End-to-end pipeline runner.

Usage:
    python run_pipeline.py                 # real MSC data (download first)
    python run_pipeline.py --synthetic     # smoke test without the dataset
    python run_pipeline.py --demo          # also run the Ollama LLM demo
    python run_pipeline.py --demo --dry-run-demo   # demo without Ollama
"""

import argparse
import sys

from src import config
from src.data_loader import generate_synthetic, load_msc
from src.features import build_dataset, save_dataset
from src.labeling import export_validation_sample
from src.train import (cross_validate, fit_final_model, pca_analysis,
                       save_results)
from src.evaluate import (plot_curves, plot_feature_importance,
                          plot_model_comparison, plot_pca_scree)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true",
                    help="use synthetic conversations (no dataset needed)")
    ap.add_argument("--max-conversations", type=int,
                    default=config.MAX_CONVERSATIONS)
    ap.add_argument("--demo", action="store_true",
                    help="run the Ollama LLM demo after training")
    ap.add_argument("--dry-run-demo", action="store_true",
                    help="run demo without calling Ollama")
    args = ap.parse_args()

    # 1. Load conversations -------------------------------------------------
    if args.synthetic:
        print("=== Loading SYNTHETIC conversations (smoke test) ===")
        conversations = generate_synthetic(n_conversations=200)
    else:
        print("=== Loading MSC dataset ===")
        try:
            conversations = load_msc(max_conversations=args.max_conversations)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
    print(f"Loaded {len(conversations)} multi-session conversations")

    # 2. Features + heuristic labels ----------------------------------------
    print("\n=== Building features and labels ===")
    df = build_dataset(conversations)
    save_dataset(df)
    export_validation_sample(df)

    # 3. Train + evaluate -----------------------------------------------------
    print("\n=== Cross-validated training (5 classifiers) ===")
    results, oof = cross_validate(df)
    evr = pca_analysis(df)
    save_results(results, evr)

    # 4. Plots ----------------------------------------------------------------
    print("\n=== Generating plots ===")
    plot_model_comparison(results)
    plot_curves(df, oof)
    plot_feature_importance(df)
    plot_pca_scree(evr)

    # 5. Final model ------------------------------------------------------------
    best = max((n for n in results if not n.startswith("Baseline")),
               key=lambda n: results[n]["f1"])
    print(f"\nBest classifier by F1: {best}")
    fit_final_model(df, best)

    # 6. Optional LLM demo --------------------------------------------------
    if args.demo:
        print("\n=== LLM demo (Ollama) ===")
        from src.llm_demo import run_demo
        run_demo(df, dry_run=args.dry_run_demo)

    print("\nDone. See the results/ folder for metrics, plots, and the model.")


if __name__ == "__main__":
    main()
