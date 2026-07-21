"""Sentence embeddings: SentenceTransformers if available, hashing fallback.

The fallback keeps the pipeline runnable in offline environments; results
in the report should always use the real all-MiniLM-L6-v2 embeddings.
"""

import hashlib
import re

import numpy as np

from . import config

_model = None
_using_fallback = False


def _try_load_model():
    global _model, _using_fallback
    if _model is not None or _using_fallback:
        return
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
        print(f"[embeddings] Using SentenceTransformer: {config.EMBEDDING_MODEL}")
    except Exception as e:  # ImportError or download failure
        if not config.EMBEDDING_FALLBACK:
            raise
        _using_fallback = True
        print(f"[embeddings] WARNING: falling back to hashing embedder ({e}). "
              "Install sentence-transformers for real results.")


def _hash_embed(texts, dim=256):
    """Deterministic bag-of-words hashing embedding (offline fallback)."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        for tok in re.findall(r"[a-z']+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            out[i, h % dim] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms


def embed(texts):
    """Embed a list of strings -> (n, d) float array, L2-normalized."""
    _try_load_model()
    if _using_fallback:
        return _hash_embed(texts)
    vecs = _model.encode(texts, show_progress_bar=False,
                         normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)


def cosine_matrix(a, b):
    """Cosine similarity between rows of a and rows of b (both normalized)."""
    return a @ b.T
