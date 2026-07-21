import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Memory Importance Classification for LLM Agents\n",
    "\n",
    "Can supervised ML predict which conversational memories will matter in the future, better than plain similarity-based retrieval?\n",
    "\n",
    "This notebook demonstrates the end-to-end pipeline: loading the Multi-Session Chat (MSC) dataset, extracting features, performing exploratory data analysis and dimensionality reduction (PCA), training five ML classifiers using StratifiedGroupKFold cross-validation, comparing them with a similarity baseline, and running an Ollama LLM demo with four memory strategies."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 0: Set Up Environment and Imports"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import sys\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.decomposition import PCA\n",
    "\n",
    "# Ensure project root is in path\n",
    "sys.path.insert(0, os.path.abspath('.'))\n",
    "\n",
    "from src import config\n",
    "from src.data_loader import load_msc, generate_synthetic\n",
    "from src.features import build_dataset, FEATURE_COLUMNS\n",
    "from src.train import cross_validate, get_models, fit_final_model, pca_analysis\n",
    "from src.evaluate import plot_model_comparison, plot_curves, plot_feature_importance, plot_pca_scree\n",
    "from src.llm_demo import run_demo\n",
    "\n",
    "print(\"Environment initialized. Using device settings from config:\")\n",
    "print(f\"Ollama model: {config.OLLAMA_MODEL}\")\n",
    "print(f\"Embedding model: {config.EMBEDDING_MODEL}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 1: Load Conversations\n",
    "\n",
    "We load the Multi-Session Chat (MSC) conversations. Memories are extracted from earlier sessions, and the final session serves as the source of future references (labels) and the current query."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "try:\n",
    "    # Load 100 conversations from the real MSC dataset\n",
    "    conversations = load_msc(max_conversations=100)\n",
    "    print(f\"Loaded {len(conversations)} conversations from real MSC dataset.\")\n",
    "except FileNotFoundError:\n",
    "    print(\"Real MSC dataset not found. Falling back to synthetic conversations for testing.\")\n",
    "    conversations = generate_synthetic(n_conversations=50)\n",
    "    print(f\"Generated {len(conversations)} synthetic conversations.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 2: Feature Engineering and Label Generation\n",
    "\n",
    "We process the raw conversations to extract 10 key features per memory and generate heuristic binary importance labels. The labels identify whether a memory is referenced in the final session via semantic similarity, lexical overlap, or shared entities."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = build_dataset(conversations)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 3: Exploratory Data Analysis & Dimensionality Analysis\n",
    "\n",
    "Let's look at the correlation of the features with the target label and run a Principal Component Analysis (PCA) to examine the explained variance."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Correlation of engineered features with the target importance label\n",
    "correlation = df[FEATURE_COLUMNS + ['label']].corr()['label'].sort_values(ascending=False)\n",
    "print(\"Correlation of features with importance label:\")\n",
    "print(correlation)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Run PCA\n",
    "X = StandardScaler().fit_transform(df[FEATURE_COLUMNS].values)\n",
    "pca = PCA().fit(X)\n",
    "evr = pca.explained_variance_ratio_\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(7, 4))\n",
    "ax.plot(np.arange(1, len(evr) + 1), np.cumsum(evr), \"o-\")\n",
    "ax.set(title=\"PCA Cumulative Explained Variance\",\n",
    "       xlabel=\"Number of Components\", ylabel=\"Cumulative Variance\")\n",
    "ax.axhline(0.95, ls=\"--\", c=\"gray\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 4: Model Cross-Validation and Comparison\n",
    "\n",
    "We train 5 classification models using 5-fold StratifiedGroupKFold cross-validation (grouped by conversation to prevent data leakage) and compare their performance against a TopK-Similarity baseline."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "results, oof = cross_validate(df)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 5: Visualizing Evaluation Metrics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot model comparison\n",
    "plot_model_comparison(results)\n",
    "\n",
    "# Show comparison plot inline\n",
    "from IPython.display import Image, display\n",
    "display(Image(filename=os.path.join(config.RESULTS_DIR, 'model_comparison.png')))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot out-of-fold ROC and PR curves\n",
    "plot_curves(df, oof)\n",
    "display(Image(filename=os.path.join(config.RESULTS_DIR, 'roc_pr_curves.png')))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot Random Forest Feature Importance\n",
    "plot_feature_importance(df)\n",
    "display(Image(filename=os.path.join(config.RESULTS_DIR, 'feature_importance.png')))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 6: Fit Final Model and Run LLM Demo\n",
    "\n",
    "We save the best classifier, and then run the LLM demo using Ollama (Llama 3.1) to observe how memory filtering affects conversational generation quality."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Determine the best classifier name by F1-score\n",
    "best_clf = max((n for n in results if not n.startswith(\"Baseline\")), key=lambda n: results[n][\"f1\"])\n",
    "print(f\"Fitting final model for best classifier: {best_clf}\")\n",
    "fit_final_model(df, best_clf)\n",
    "\n",
    "# Run the LLM demo using the saved model\n",
    "run_demo(df, dry_run=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Display some demo results\n",
    "demo_file = os.path.join(config.RESULTS_DIR, \"llm_demo_outputs.json\")\n",
    "if os.path.exists(demo_file):\n",
    "    with open(demo_file) as f:\n",
    "        demo_data = json.load(f)\n",
    "    \n",
    "    # Print comparison for the first conversation\n",
    "    first_conv = demo_data[0]\n",
    "    print(f\"Query: {first_conv['query']}\\n\")\n",
    "    for strategy, info in first_conv['strategies'].items():\n",
    "        print(f\"Strategy: {strategy} (Memories: {info['n_memories']})\")\n",
    "        print(f\"Response: {info['response']}\\n\")"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open("memory_importance_classifier.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)
print("memory_importance_classifier.ipynb created successfully!")
