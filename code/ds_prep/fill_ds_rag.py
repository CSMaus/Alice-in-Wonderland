#!/usr/bin/env python3
"""
Fill instruction dataset using RAG over Alice embeddings.

- Reads API key from ../data/key.txt
- Loads embeddings/chunks from ../data/alice_embeddings.npz
- Builds FAISS (cosine) index
- For each item in Alpaca, retrieves top-k book chunks
- If retrieval is weak, it STILL calls the model with empty context so the model
  naturally replies with an in-world "don't know" (no meta wording)
- Writes JSONL to ../data/alice_filled.json with fields: instruction, input, output

"""

import os, json, re
import numpy as np
import jsonlines
import faiss
from openai import OpenAI

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(ROOT, "data")
KEY_PATH = os.path.join(DATA_DIR, "key.txt")
CONFIGS = {"short", "fine", "medium", "coarse"}

EMB_NPZ_BASENAME = os.path.join(DATA_DIR, "alice_embeddings")
NPZ_PATHS = {name: f"{EMB_NPZ_BASENAME}_{name}.npz" for name in CONFIGS}
ALPACA_JSON = os.path.join(DATA_DIR, "alpaca_data_cleaned.json")
OUT_JSONL = os.path.join(DATA_DIR, "alice_world_dataset.json")

EMB_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"  # -mini
TOPK = 3  # but totally will use 9 to include surroundings of chosen
# think about Max Marginal Relevance and re-ranker clear not repetitive answer.
TAU = 0.30   # similarity gate; below this → empty context (model will naturally say it doesn't know)
TEMP = 0.2

PER_INDEX_TOPK = {"short": 3, "fine": 3, "medium": 3, "coarse": 3}
FINAL_K = 12
WINDOW_BY = {"short": 2, "fine": 1, "medium": 1, "coarse": 0}
MMR_LAMBDA = 0.7
MAX_CONTEXT_CHARS = 12000
SYSTEM_PROMPT = (
    "You know only the world of “Alice’s Adventures in Wonderland.” "
    "Answer as if you exist inside that world. Use the provided CONTEXT text, and speak naturally in that world’s tone. "
    "If the question makes no sense in this world or has no answer, reply in a short, natural way that simply shows you don’t know or understand. "
    "Use any natural wording you like (e.g., “I don’t know.” “I can’t say.” “I haven’t heard of it.”), but do not mention other worlds, contexts, books, or outside information. "
    "Keep replies concise (1–3 sentences)."
)

def load_key(p):
    with open(p, "r", encoding="utf-8") as f:
        return f.read().strip()

def load_embeddings(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    X = d["embeddings"].astype("float32")
    chunks = d["chunks"]
    return X, chunks

def build_index(X):
    faiss.normalize_L2(X)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    return index

def load_alpaca(path):
    # Accepts list JSON or JSONL
    if path.endswith(".jsonl"):
        items = []
        with jsonlines.open(path, "r") as r:
            for o in r:
                items.append(o)
        return items
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Some alpaca variants store a list; others under a key
        return data if isinstance(data, list) else data.get("data", [])

def embed_query(client, text):
    resp = client.embeddings.create(model=EMB_MODEL, input=[text])
    return np.array(resp.data[0].embedding, dtype="float32")[None, :]

def retrieve_old(index, qvec, chunks, topk=TOPK):
    # for one file with less customizable settings
    qv = qvec.copy()
    faiss.normalize_L2(qv)
    D, I = index.search(qv, topk)
    sims = D[0].tolist()
    idxs = I[0].tolist()
    ctx = [chunks[i] for i in idxs if 0 <= i < len(chunks)]
    return ctx, sims

def retrieve(index, qvec, chunks, topk=TOPK):
    qv = qvec.copy()
    faiss.normalize_L2(qv)
    D, I = index.search(qv, topk)
    sims = D[0].tolist()
    idxs = I[0].tolist()
    return idxs, sims

def mmr_select(cands, final_k, mmr_lambda):
    # cands: list of dicts {name, idx, score, chunk}
    selected, pool = [], cands[:]
    if not pool: return selected
    mx = max(c["score"] for c in pool) or 1e-6
    for c in pool: c["score"] = c["score"] / mx
    while pool and len(selected) < final_k:
        if not selected:
            best = max(pool, key=lambda c: c["score"])
        else:
            def mmr_val(c):
                # discourage near-duplicates from same index (±1)
                same = [s for s in selected if s["name"] == c["name"]]
                penalty = 1.0 if any(abs(s["idx"] - c["idx"]) <= 1 for s in same) else 0.0
                return mmr_lambda * c["score"] - (1 - mmr_lambda) * penalty
            best = max(pool, key=mmr_val)
        selected.append(best); pool.remove(best)
    return selected

def expand_neighbors_for(picked, chunks_map, window_by):
    seen = set(); final_chunks = []
    for p in picked:
        name, idx = p["name"], p["idx"]
        w = window_by.get(name, 0)
        total = len(chunks_map[name])
        for j in range(max(0, idx - w), min(total, idx + w + 1)):
            key = (name, j)
            if key in seen: continue
            seen.add(key)
            final_chunks.append(str(chunks_map[name][j]))
    return final_chunks

def make_user_prompt(ctx_chunks, instr, inp):
    ctx_block = ""
    if ctx_chunks:
        ctx_block = "---- CONTEXT ----\n" + "\n\n".join(ctx_chunks) + "\n-----------------\n"
    return f"{ctx_block}Instruction: {instr}\nInput: {inp}\nAnswer:"

def make_user_prompt_improved(ctx_chunks, instr, inp):
    # improved - limin tokens
    ctx_block = ""
    if ctx_chunks:
        joined = "\n\n".join(ctx_chunks)
        if len(joined) > MAX_CONTEXT_CHARS:
            joined = joined[:MAX_CONTEXT_CHARS]
        ctx_block = "---- CONTEXT ----\n" + joined + "\n-----------------\n"
    return f"{ctx_block}Instruction: {instr}\nInput: {inp}\nAnswer:"


def main():
    api_key = load_key(KEY_PATH)
    client = OpenAI(api_key=api_key)

    indices = {}  # name -> faiss index
    chunks_map = {}  # name -> np.ndarray of chunks
    for name, path in NPZ_PATHS.items():
        X, chunks = load_embeddings(path)
        indices[name] = build_index(X)
        chunks_map[name] = chunks

    items = load_alpaca(ALPACA_JSON)
    # items = items[:100]  # for test first
    existing_pairs = set()
    if os.path.exists(OUT_JSONL):
        with jsonlines.open(OUT_JSONL, "r") as r:
            for o in r:
                existing_pairs.add((o.get("instruction", ""), o.get("input", "")))
    print(f"[resume] existing items: {len(existing_pairs)}")

    out_path = OUT_JSONL
    n = len(items)
    done = 0
    with jsonlines.open(out_path, "a") as w:
        for it in items:

            instr = (it.get("instruction") or "").strip()
            inp = (it.get("input") or it.get("context") or "").strip()  # Dolly uses "context"

            key = (instr, inp)
            if key in existing_pairs:
                continue

            query = (instr + "\n" + inp).strip() if inp else instr
            if not query:
                continue

            qvec = embed_query(client, query)
            candidates = []
            for name, index in indices.items():
                idxs, sims = retrieve(index, qvec, chunks_map[name], topk=PER_INDEX_TOPK[name])
                for i, s in zip(idxs, sims):
                    if i < 0: continue
                    candidates.append({
                        "name": name,
                        "idx": i,
                        "score": float(s),
                        "chunk": str(chunks_map[name][i]),
                    })

            max_sim = max([c["score"] for c in candidates], default=0.0)
            if max_sim < TAU:
                use_ctx = []  # no per system prompt
            else:
                # MMR filter to FINAL_K across pooled candidates
                picked = mmr_select(sorted(candidates, key=lambda c: c["score"], reverse=True),
                                    FINAL_K, MMR_LAMBDA)
                # per-index windows AFTER MMR selection
                use_ctx = expand_neighbors_for(picked, chunks_map, WINDOW_BY)

            user_msg = make_user_prompt(use_ctx, instr, inp)
            resp = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=TEMP,
            )
            out_text = resp.choices[0].message.content.strip()

            w.write({
                "instruction": instr,
                "input": inp,
                "output": out_text
            })

            done += 1
            if done % 50 == 0 or done == n:
                print(f"[progress] {done}/{n}")

    print(f"[saved] {out_path}")

if __name__ == "__main__":
    main()