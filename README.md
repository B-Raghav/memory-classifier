# Memory Importance Classification for LLM Agents

Can supervised ML predict which conversational memories will matter in the
future, better than plain similarity-based retrieval? This project frames
memory selection as **binary classification** (retain vs. discard) on the
**Multi-Session Chat (MSC)** dataset, compares five classifiers, and
demonstrates the winner inside a local LLM (Ollama / Llama 3) against three
baselines.

---

## 1. Quick start

```bash
# 0. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 1. Install dependencies
pip install -r requirements.txt

# 2. Smoke test WITHOUT the dataset (30 seconds, verifies everything works)
python run_pipeline.py --synthetic

# 3. Download the real MSC dataset (~30 MB, fully open-source, no API keys)
python scripts/download_msc.py

# 4. Run the full pipeline on MSC
python run_pipeline.py
```

Everything lands in `results/`: metrics (JSON + CSV), four plots, the saved
best model, and a manual-validation sample of the heuristic labels.

Requires **Python 3.9+**. The first real run downloads the
`all-MiniLM-L6-v2` embedding model (~90 MB) from HuggingFace automatically.

---

## 2. What the pipeline does

For each multi-session conversation:

1. **Split** — the *final session* is treated as "the future". All earlier
   turns become candidate **memories**; the first user turn of the final
   session is the **query**.
2. **Label (heuristic)** — a memory is labeled *important (1)* if the final
   session references it via **any** of three signals:
   - semantic: cosine similarity ≥ 0.60 to any future turn
   - lexical: content-word Jaccard overlap ≥ 0.30
   - entity: shares ≥ 1 capitalized "entity" token

   The lexical/entity signals are independent of the embedding space, which
   reduces circularity between labels and the similarity feature. A
   stratified 200-example sample is exported to
   `results/label_validation_sample.csv` with an empty `human_label` column
   for manual validation.
3. **Features (10 per memory)** — recency, session gap, simulated access
   frequency, cosine similarity to the query, speaker role, sentence length,
   is-question, entity count, first-person marker, position in session.
4. **Train** — Logistic Regression, k-NN, Decision Tree, Random Forest,
   AdaBoost. Preprocessing = StandardScaler (+ SMOTE inside the CV pipeline,
   so no leakage). Evaluation = **StratifiedGroupKFold** (grouped by
   conversation so memories from one conversation never straddle
   train/test), reporting accuracy, precision, recall, F1, ROC-AUC, and
   PR-AUC. PCA scree analysis included.
5. **Baseline** — fixed top-K similarity retrieval, evaluated with the same
   metrics for direct comparison.
6. **Persist** — the best model (by F1) is refit on all data and saved to
   `results/best_model.pkl`.

---

## 3. LLM demo (Ollama)

Shows how memory selection strategy changes LLM responses. Four conditions:
`no_memory`, `store_all`, `top_k` (similarity), and `ml_selected` (the
trained classifier).

```bash
# Install Ollama from https://ollama.com then pull the model:
ollama pull llama3
ollama serve            # if not already running

# Run pipeline + demo:
python run_pipeline.py --demo

# Or test the demo plumbing without Ollama installed:
python run_pipeline.py --synthetic --demo --dry-run-demo
```

Outputs go to `results/llm_demo_outputs.json` (per-conversation: the query,
memories selected by each strategy, and the LLM's response) for qualitative
comparison or LLM-as-judge scoring.

---

## 4. Project structure

```
memory-importance-classifier/
├── run_pipeline.py            # main entry point
├── requirements.txt
├── scripts/
│   └── download_msc.py        # downloads + extracts MSC (~30 MB)
├── src/
│   ├── config.py              # all thresholds/paths/hyperparameters
│   ├── data_loader.py         # MSC loader + synthetic generator
│   ├── embeddings.py          # SentenceTransformers (hashing fallback)
│   ├── labeling.py            # heuristic importance labels + validation export
│   ├── features.py            # feature engineering
│   ├── train.py               # 5 classifiers, SMOTE, grouped k-fold CV
│   ├── evaluate.py            # plots
│   └── llm_demo.py            # Ollama demo, 4 memory strategies
├── data/                      # dataset + features.csv (created at runtime)
└── results/                   # metrics, plots, model, demo outputs
```

## 5. Command reference

| Command | What it does |
|---|---|
| `python run_pipeline.py --synthetic` | Full pipeline on synthetic data (no downloads) |
| `python scripts/download_msc.py` | Download + extract MSC |
| `python run_pipeline.py` | Full pipeline on MSC |
| `python run_pipeline.py --max-conversations 100` | Faster run on a subset |
| `python run_pipeline.py --demo` | Also run the Ollama demo |
| `python run_pipeline.py --demo --dry-run-demo` | Demo without calling Ollama |

All thresholds (similarity/Jaccard cutoffs, K, folds, SMOTE on/off, model
names) live in `src/config.py`.

## 6. Notes, caveats & extensions

- **Embedding fallback**: if `sentence-transformers` is missing or the model
  can't be downloaded, a hashing bag-of-words embedder keeps the pipeline
  runnable. **Report results only with the real embeddings.**
- **MSC loader robustness**: the loader parses the ParlAI JSONL format
  defensively and groups sessions by `initial_data_id`. If ParlAI changes
  the archive layout, inspect `data/msc/` and adjust
  `src/data_loader.load_msc()` (it raises a clear error if grouping fails).
- **Label sensitivity**: rerun with different `LABEL_SIM_THRESHOLD` /
  `LABEL_JACCARD_THRESHOLD` values and report how the positive rate and F1
  change — this is a strong addition to the report.
- **Ablation to include in the report**: drop `semantic_similarity` from
  `FEATURE_COLUMNS` in `src/features.py` and retrain. If the classifier
  still beats the similarity baseline, the "ML learns more than similarity"
  claim is much stronger.
- The `access_frequency` feature is *simulated* (memory's similarity to
  later past user turns) since no live retrieval system exists at training
  time — state this explicitly in your write-up.

## 7. References

- Xu, J., Szlam, A., & Weston, J. (2022). *Beyond Goldfish Memory: Long-Term
  Open-Domain Conversation.* ACL 2022. (MSC dataset)
- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings
  using Siamese BERT-Networks.* EMNLP 2019.
