"""Feature engineering.

For each conversation, the *final session* is treated as the future:
  - query      = first user turn of the final session
  - memories   = all turns in earlier sessions
  - label      = memory referenced anywhere in the final session (labeling.py)

Feature vector per memory:
  recency            turns between memory and query
  session_gap        number of sessions between memory and query
  access_frequency   times memory was similar (>=0.5 cos) to any *past* user
                     turn after its creation - a simulated retrieval count
  semantic_similarity cosine similarity between memory and the query
  role_user          1 if spoken by the user
  sentence_length    word count
  is_question        contains '?'
  entity_count       capitalized-token count (cheap NER proxy)
  first_person       contains a first-person fact marker (i/my/we/our)
  position_in_session relative position (0..1) inside its session
"""

import os
import re

import numpy as np
import pandas as pd

from . import config
from .embeddings import embed, cosine_matrix
from .labeling import label_memories, entity_tokens

FEATURE_COLUMNS = [
    "recency", "session_gap", "access_frequency", "semantic_similarity",
    "role_user", "sentence_length", "is_question", "entity_count",
    "first_person", "position_in_session",
]

_FP = re.compile(r"\b(i|my|mine|we|our)\b", re.I)


def build_dataset(conversations, verbose=True):
    rows = []
    for conv_id, conv in enumerate(conversations):
        if len(conv) < 2:
            continue
        past_sessions, future_session = conv[:-1], conv[-1]
        memories = [t for s in past_sessions for t in s
                    if "?" not in t["text"] and len(t["text"].split()) >= 4]
        if not memories or not future_session:
            continue

        query = next((t for t in future_session if t["speaker"] == "user"),
                     future_session[0])
        future_texts = [t["text"] for t in future_session]

        mem_texts = [m["text"] for m in memories]
        mem_embs = embed(mem_texts)
        fut_embs = embed(future_texts)
        query_emb = embed([query["text"]])

        labels, _ = label_memories(mem_texts, mem_embs, future_texts, fut_embs)
        sim_to_query = cosine_matrix(mem_embs, query_emb)[:, 0]

        # simulated access counts: memory vs. later *past* user turns
        past_user_idx = [i for i, m in enumerate(memories)
                         if m["speaker"] == "user"]
        mem_mem_sim = cosine_matrix(mem_embs, mem_embs)

        query_turn = query["turn"]
        query_session = query["session"]

        for i, m in enumerate(memories):
            later_users = [j for j in past_user_idx if memories[j]["turn"] > m["turn"]]
            access = int(sum(mem_mem_sim[i, j] >= 0.5 for j in later_users))

            sess_turns = [t for s in past_sessions for t in s
                          if t["session"] == m["session"]]
            pos = (sess_turns.index(m) / max(len(sess_turns) - 1, 1)
                   if m in sess_turns else 0.0)

            rows.append({
                "conversation_id": conv_id,
                "query_text": query["text"],
                "text": m["text"],
                "recency": query_turn - m["turn"],
                "session_gap": query_session - m["session"],
                "access_frequency": access,
                "semantic_similarity": float(sim_to_query[i]),
                "role_user": int(m["speaker"] == "user"),
                "sentence_length": len(m["text"].split()),
                "is_question": int("?" in m["text"]),
                "entity_count": len(entity_tokens(m["text"])),
                "first_person": int(bool(_FP.search(m["text"]))),
                "position_in_session": pos,
                "label": int(labels[i]),
            })
        if verbose and (conv_id + 1) % 50 == 0:
            print(f"[features] processed {conv_id + 1} conversations")

    df = pd.DataFrame(rows)
    if verbose:
        pos_rate = df.label.mean() if len(df) else 0
        print(f"[features] {len(df)} memories from "
              f"{df.conversation_id.nunique() if len(df) else 0} conversations "
              f"| positive rate = {pos_rate:.3f}")
    return df


def save_dataset(df, path=None):
    path = path or config.FEATURES_CSV
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[features] saved -> {path}")
