"""Central configuration for the memory importance classification pipeline."""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MSC_DIR = os.path.join(DATA_DIR, "msc")            # extracted MSC dataset
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FEATURES_CSV = os.path.join(DATA_DIR, "features.csv")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
# ParlAI-hosted archive of the Multi-Session Chat dataset (Xu et al., ACL 2022)
MSC_URL = "http://parl.ai/downloads/msc/msc_v0.1.tar.gz"

# Which sessions to load per conversation (MSC has up to 5 sessions).
# Memories are drawn from all sessions except the last loaded one;
# the last loaded session is the "future" used for labeling + querying.
MAX_CONVERSATIONS = 500        # cap for speed; set None for all
MIN_SESSIONS = 2               # skip conversations with fewer sessions

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # SentenceTransformers model
EMBEDDING_FALLBACK = True              # use hashing embedder if ST unavailable

# ---------------------------------------------------------------------------
# Labeling heuristic (a memory is "important" if referenced in the future)
# ---------------------------------------------------------------------------
LABEL_SIM_THRESHOLD = 0.60      # cosine similarity to any future turn
LABEL_JACCARD_THRESHOLD = 0.30  # content-word overlap with any future turn
LABEL_ENTITY_OVERLAP = 1        # min shared capitalized tokens ("entities")

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
N_FOLDS = 5
USE_SMOTE = True
TOP_K_BASELINE = 5              # fixed top-K similarity retrieval baseline

# ---------------------------------------------------------------------------
# LLM demo (Ollama)
# ---------------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b-instruct-q4_K_M"
DEMO_NUM_CONVERSATIONS = 5
