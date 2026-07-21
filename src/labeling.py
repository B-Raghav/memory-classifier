"""Heuristic importance labels.

A memory (utterance before the final session) is labeled IMPORTANT (1) if it
is referenced in the final session by ANY of three signals:

  1. Semantic:  cosine similarity >= LABEL_SIM_THRESHOLD to any future turn
  2. Lexical:   content-word Jaccard overlap >= LABEL_JACCARD_THRESHOLD
  3. Entity:    shares >= LABEL_ENTITY_OVERLAP capitalized tokens ("entities")

Signals 2 and 3 are independent of the embedding space, which mitigates
label/feature circularity (the classifier's similarity feature is computed
against the *query* only, while labels use the whole future session).

export_validation_sample() writes a CSV of labeled examples for manual
validation, as described in the proposal.
"""

import csv
import os
import re

import numpy as np

from . import config
from .embeddings import cosine_matrix

_STOPWORDS = set("""a an the and or but if then so to of in on at for with
about as by from is are was were be been being am i you he she it we they
me him her us them my your his its our their this that these those do does
did have has had not no yes what when where who how why which there here
""".split())


def content_words(text):
    return {w for w in re.findall(r"[a-z']+", text.lower())
            if w not in _STOPWORDS and len(w) > 2}


def entity_tokens(text):
    """Capitalized non-sentence-initial tokens as a cheap entity proxy."""
    tokens = text.split()
    return {t.strip(".,!?") for t in tokens[1:]
            if t[:1].isupper() and len(t) > 2}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def label_memories(memory_texts, memory_embs, future_texts, future_embs):
    """Return binary labels + per-signal breakdown for one conversation."""
    n = len(memory_texts)
    labels = np.zeros(n, dtype=int)
    signals = []

    sim = cosine_matrix(memory_embs, future_embs) if len(future_texts) else \
        np.zeros((n, 0))

    fut_words = [content_words(t) for t in future_texts]
    fut_ents = [entity_tokens(t) for t in future_texts]

    for i, mem in enumerate(memory_texts):
        mw, me = content_words(mem), entity_tokens(mem)
        sem = bool(sim.shape[1]) and float(sim[i].max()) >= config.LABEL_SIM_THRESHOLD
        lex = any(jaccard(mw, fw) >= config.LABEL_JACCARD_THRESHOLD
                  for fw in fut_words)
        ent = any(len(me & fe) >= config.LABEL_ENTITY_OVERLAP
                  for fe in fut_ents if me)
        labels[i] = int(sem or lex or ent)
        signals.append({"semantic": sem, "lexical": lex, "entity": ent})
    return labels, signals


def export_validation_sample(df, n=200, path=None,
                             seed=config.RANDOM_STATE):
    """Write a stratified sample for manual label validation."""
    path = path or os.path.join(config.RESULTS_DIR, "label_validation_sample.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pos = df[df.label == 1]
    neg = df[df.label == 0]
    k = min(n // 2, len(pos), len(neg))
    sample = (
        pos.sample(k, random_state=seed)
        .append(neg.sample(k, random_state=seed))
        if hasattr(pos, "append")
        else __import__("pandas").concat(
            [pos.sample(k, random_state=seed), neg.sample(k, random_state=seed)]
        )
    )
    cols = ["conversation_id", "text", "label"]
    sample[cols].assign(human_label="").to_csv(path, index=False,
                                               quoting=csv.QUOTE_ALL)
    print(f"[labeling] Wrote {len(sample)} examples for manual validation: {path}")
