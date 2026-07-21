"""Load Multi-Session Chat (MSC) conversations.

Produces a list of conversations, each a list of sessions, each a list of
turns: {"text": str, "speaker": "user"|"agent", "session": int, "turn": int}.

If the real MSC data is not present, an optional synthetic generator lets you
smoke-test the whole pipeline (--synthetic flag on run_pipeline.py).
"""

import glob
import json
import os
import random

from . import config


# ---------------------------------------------------------------------------
# Real MSC loading
# ---------------------------------------------------------------------------
def _parse_msc_line(line):
    """Parse one JSONL line of MSC into a list of turns (best-effort)."""
    obj = json.loads(line)
    dialog = obj.get("dialog") or obj.get("dialogue") or []
    turns = []
    for i, t in enumerate(dialog):
        if isinstance(t, dict):
            text = t.get("text", "")
            spk = str(t.get("id", t.get("speaker", ""))).lower()
        else:
            text, spk = str(t), ""
        speaker = "user" if ("speaker_1" in spk or "user" in spk or i % 2 == 0) else "agent"
        if text.strip():
            turns.append({"text": text.strip(), "speaker": speaker})
    return turns


def load_msc(max_conversations=None):
    """Load MSC sessions and group them into multi-session conversations.

    The extracted archive contains per-session folders, e.g.:
        data/msc/msc/msc_dialogue/session_2/train.txt   (JSONL)
    Sessions of the same underlying conversation are linked by their
    'metadata' / 'initial_data_id' field; we group on that when available,
    otherwise we fall back to line index alignment across session files.
    """
    pattern = os.path.join(config.MSC_DIR, "**", "session_*", "train.txt")
    session_files = sorted(glob.glob(pattern, recursive=True))
    if not session_files:
        raise FileNotFoundError(
            f"No MSC session files found under {config.MSC_DIR}. "
            "Run: python scripts/download_msc.py"
        )

    # conversation_key -> {session_number: [turns]}
    grouped = {}
    for path in session_files:
        sess_num = int(os.path.basename(os.path.dirname(path)).split("_")[-1])
        with open(path, encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    key = (
                        obj.get("metadata", {}).get("initial_data_id")
                        if isinstance(obj.get("metadata"), dict) else None
                    ) or obj.get("initial_data_id") or f"line_{line_idx}"
                    turns = _parse_msc_line(line)
                except (json.JSONDecodeError, KeyError):
                    continue
                if turns:
                    grouped.setdefault(key, {})[sess_num] = turns

    conversations = []
    for key, sessions in grouped.items():
        if len(sessions) < config.MIN_SESSIONS:
            continue
        conv, turn_counter = [], 0
        for sess_num in sorted(sessions):
            sess = []
            for t in sessions[sess_num]:
                t = dict(t, session=sess_num, turn=turn_counter)
                turn_counter += 1
                sess.append(t)
            conv.append(sess)
        conversations.append(conv)
        if max_conversations and len(conversations) >= max_conversations:
            break

    if not conversations:
        raise ValueError(
            "MSC files were found but no multi-session conversations could be "
            "grouped. The archive layout may differ from what this loader "
            "expects - inspect data/msc/ and adjust load_msc() accordingly."
        )
    return conversations


# ---------------------------------------------------------------------------
# Synthetic fallback (for smoke tests without the dataset)
# ---------------------------------------------------------------------------
_TOPICS = [
    ("dogs", "I have two golden retrievers named Max and Ruby"),
    ("job", "I work as a nurse at the city hospital"),
    ("hiking", "I love hiking in the mountains every weekend"),
    ("cooking", "My favorite dish to cook is homemade lasagna"),
    ("guitar", "I have been learning guitar for three years"),
    ("travel", "I visited Japan last spring and loved Kyoto"),
    ("books", "I mostly read science fiction novels"),
    ("garden", "I grow tomatoes and peppers in my garden"),
]
_FILLER = [
    "That sounds really nice.", "How was your day today?",
    "The weather has been great lately.", "I agree with you completely.",
    "What do you think about that?", "It has been a busy week for me.",
]


def generate_synthetic(n_conversations=200, seed=config.RANDOM_STATE):
    """Generate multi-session conversations where some early facts are
    deliberately referenced again in the final session (positives)."""
    rng = random.Random(seed)
    conversations = []
    for _ in range(n_conversations):
        facts = rng.sample(_TOPICS, k=4)
        referenced = rng.sample(facts, k=rng.randint(1, 2))
        conv, turn = [], 0
        for s in range(3):  # 3 sessions
            sess = []
            n_turns = rng.randint(6, 10)
            for i in range(n_turns):
                speaker = "user" if i % 2 == 0 else "agent"
                if s < 2 and rng.random() < 0.3 and facts:
                    topic, text = facts[rng.randrange(len(facts))]
                elif s == 2 and rng.random() < 0.4 and referenced:
                    topic, base = referenced[rng.randrange(len(referenced))]
                    text = f"Earlier you mentioned {base.lower()}, tell me more about your {topic}."
                else:
                    text = rng.choice(_FILLER)
                sess.append({"text": text, "speaker": speaker,
                             "session": s + 1, "turn": turn})
                turn += 1
            conv.append(sess)
        conversations.append(conv)
    return conversations
