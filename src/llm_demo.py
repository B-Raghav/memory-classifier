"""End-to-end demo: adaptive memory selection with a local LLM (Ollama).

Compares four memory strategies on held-out conversations:
  1. no_memory   - answer the query with no conversation history
  2. store_all   - inject ALL past memories
  3. top_k       - inject top-K memories by cosine similarity to the query
  4. ml_selected - inject memories the trained classifier predicts important

Outputs results/llm_demo_outputs.json with the prompt, selected memories,
and the model's response per strategy, so responses can be compared
qualitatively (or scored with an LLM judge / human eval).

Requires Ollama running locally:  https://ollama.com  ->  `ollama run llama3`
"""

import json
import os
import pickle

import numpy as np
import pandas as pd

from . import config
from .features import FEATURE_COLUMNS


def _ollama(prompt):
    import urllib.request
    req = urllib.request.Request(
        config.OLLAMA_URL,
        data=json.dumps({"model": config.OLLAMA_MODEL, "prompt": prompt,
                         "stream": False}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["response"].strip()


def _prompt(memories, query):
    mem_block = ("Relevant facts from earlier conversations:\n"
                 + "\n".join(f"- {m}" for m in memories) + "\n\n") if memories else ""
    return (f"{mem_block}The user says: \"{query}\"\n"
            "Respond naturally in 1-3 sentences, using the facts if relevant.")


def run_demo(df, model_path=None, n_conversations=None, dry_run=False):
    model_path = model_path or os.path.join(config.RESULTS_DIR, "best_model.pkl")
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    pipe = bundle["pipeline"]

    n_conversations = n_conversations or config.DEMO_NUM_CONVERSATIONS
    conv_ids = df["conversation_id"].unique()[:n_conversations]
    outputs = []

    for cid in conv_ids:
        sub = df[df.conversation_id == cid].reset_index(drop=True)
        query = (sub["query_text"].iloc[0]
                 if "query_text" in sub.columns
                 else "Let's catch up - what do you remember about me?")

        probs = pipe.predict_proba(sub[FEATURE_COLUMNS].values)[:, 1]
        ml_sel = sub.text[probs >= 0.5].tolist()
        topk = sub.nlargest(config.TOP_K_BASELINE, "semantic_similarity").text.tolist()

        strategies = {
            "no_memory": [],
            "store_all": sub.text.tolist(),
            "top_k": topk,
            "ml_selected": ml_sel,
        }
        record = {"conversation_id": int(cid), "query": query, "strategies": {}}
        for name, mems in strategies.items():
            prompt = _prompt(mems, query)
            response = "(dry run - Ollama not called)" if dry_run else _ollama(prompt)
            record["strategies"][name] = {
                "n_memories": len(mems),
                "memories": mems[:20],
                "response": response,
            }
            print(f"[demo] conv {cid} | {name:<12} | {len(mems)} memories")
        outputs.append(record)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, "llm_demo_outputs.json")
    with open(path, "w") as f:
        json.dump(outputs, f, indent=2)
    print(f"[demo] saved -> {path}")


if __name__ == "__main__":
    df = pd.read_csv(config.FEATURES_CSV)
    run_demo(df)
