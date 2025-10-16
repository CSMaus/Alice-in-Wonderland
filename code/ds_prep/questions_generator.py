#!/usr/bin/env python3
import os, json, random, numpy as np
from openai import OpenAI
import faiss

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(ROOT, "data")
KEY_PATH = os.path.join(DATA_DIR, "key.txt")

NPZ_MED = os.path.join(DATA_DIR, "alice_embeddings_medium.npz")
NPZ_COA = os.path.join(DATA_DIR, "alice_embeddings_coarse.npz")
OUT_JSON = os.path.join(DATA_DIR, "alice_questions.json")

EMB_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"
TEMP = 0.2
DEDUP_SIM = 0.90  # cosine
RAND_FRACTION_COMBINED = 0.10
RAND_SEED = 42
SAVE_EVERY = 10

PROMPT_A_SYS = "You know only the world of “Alice’s Adventures in Wonderland.” Use ONLY the provided CONTEXT."
PROMPT_A_USER = (
    "---- CONTEXT ----\n{ctx}\n-----------------\n"
    "Write 1 to 3 short, self-contained user questions that can be answered from this context alone.\n"
    "Include both general-world and story-specific questions when possible.\n"
    "No answers. Output a JSON array of strings."
)

PROMPT_B_SYS = PROMPT_A_SYS
PROMPT_B_USER = (
    "---- CONTEXT ----\n{ctx}\n-----------------\n"
    "Create 1 to 3 items for an Alpaca-style dataset where:\n"
    "- \"instruction\" is a brief situation/setup inside this world (place, time, mood, who is present),\n"
    "- \"input\" is a short user question asked within that situation,\n"
    "- \"output\" is omitted (leave it empty).\n\n"
    "Output a JSON array of objects with keys exactly: \"instruction\" and \"input\".\n"
    "No answers. No meta."
)

def load_key(p): return open(p, "r", encoding="utf-8").read().strip()
def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return d["chunks"]

def embed_texts(client, texts):
    resp = client.embeddings.create(model=EMB_MODEL, input=texts)
    V = np.array([d.embedding for d in resp.data], dtype="float32")
    faiss.normalize_L2(V)
    return V

def dedup_keep_unique(client, items):
    if not items: return items
    qs = [i["instruction"] + " ||| " + (i["input"] or "") for i in items]
    V = embed_texts(client, qs)
    keep = []
    kept = None
    for i, it in enumerate(items):
        v = V[i:i+1]
        if kept is None:
            kept = v.copy(); keep.append(it); continue
        sims = (v @ kept.T)[0]
        if np.max(sims) < DEDUP_SIM:
            keep.append(it)
            kept = np.vstack([kept, v])
    return keep

def combined_contexts(chunks_med, chunks_coa):
    # old, without limits
    Lm, Lc = len(chunks_med), len(chunks_coa)
    step_m, step_c = 3, 2
    for i in range(0, max(0, Lm - 2), step_m):
        med_span = [str(chunks_med[i]), str(chunks_med[i+1]), str(chunks_med[i+2])]
        j = (i // step_m * step_c + Lc // 2) % max(1, Lc - 1)
        coa_span = [str(chunks_coa[j]), str(chunks_coa[j+1])]
        yield "\n\n".join(med_span + coa_span)

def medium_only_contexts(chunks_med, span=4, step=1):
    L = len(chunks_med)
    for i in range(0, max(0, L - span + 1), step):
        yield "\n\n".join(str(ch) for ch in chunks_med[i:i+span])

def combined_contexts_random(chunks_med, chunks_coa, fraction=0.10, seed=None):
    rng = random.Random(seed)
    Lm, Lc = len(chunks_med), len(chunks_coa)

    # candidate starts for triples (medium) and pairs (coarse)
    med_starts = list(range(0, max(0, Lm - 2)))   # i, i+1, i+2
    coa_starts = list(range(0, max(0, Lc - 1)))   # j, j+1

    # sample size = 10% of all coarse pairs (at least 1)
    n = max(1, int(round(fraction * len(coa_starts))))
    # cannot sample more than available starts
    n = min(n, len(med_starts), len(coa_starts))

    # sample without replacement
    sel_med = rng.sample(med_starts, n)
    sel_coa = rng.sample(coa_starts, n)

    for i, j in zip(sel_med, sel_coa):
        if i + 2 >= Lm or j + 1 >= Lc:
            continue
        med_span = [str(chunks_med[i]), str(chunks_med[i+1]), str(chunks_med[i+2])]
        coa_span = [str(chunks_coa[j]), str(chunks_coa[j+1])]
        yield "\n\n".join(med_span + coa_span)

def save_json_atomic(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def main():
    client = OpenAI(api_key=load_key(KEY_PATH))
    C_med = load_npz(NPZ_MED)
    C_coa = load_npz(NPZ_COA)

    med_ctxs = list(medium_only_contexts(C_med, span=4, step=1))
    comb_ctxs = list(combined_contexts_random(C_med, C_coa, fraction=RAND_FRACTION_COMBINED, seed=RAND_SEED))
    print(f"[info] medium-only contexts: {len(med_ctxs)}", flush=True)
    print(f"[info] mixed contexts (random 10%): {len(comb_ctxs)}", flush=True)

    out = []

    for i, ctx in enumerate(med_ctxs, 1):
        respA = client.chat.completions.create(
            model=CHAT_MODEL, temperature=TEMP,
            messages=[{"role":"system","content":PROMPT_A_SYS},
                      {"role":"user","content":PROMPT_A_USER.format(ctx=ctx)}]
        )
        try:
            qlist = json.loads(respA.choices[0].message.content)
            qlist = [q.strip() for q in qlist if isinstance(q, str) and q.strip()]
        except Exception:
            qlist = []
        for q in qlist:
            out.append({"instruction": q, "input": "", "output": ""})

        respB = client.chat.completions.create(
            model=CHAT_MODEL, temperature=TEMP,
            messages=[{"role":"system","content":PROMPT_B_SYS},
                      {"role":"user","content":PROMPT_B_USER.format(ctx=ctx)}]
        )
        try:
            pairs = json.loads(respB.choices[0].message.content)
            pairs = [p for p in pairs if isinstance(p, dict)
                     and isinstance(p.get("instruction",""), str)
                     and isinstance(p.get("input",""), str)
                     and p["instruction"].strip() and p["input"].strip()]
        except Exception:
            pairs = []
        for p in pairs:
            out.append({"instruction": p["instruction"].strip(),
                        "input": p["input"].strip(),
                        "output": ""})
        if len(out) % SAVE_EVERY == 0:
            save_json_atomic(OUT_JSON, out)
        print(f"[progress][medium] {i}/{len(med_ctxs)} | items={len(out)}", flush=True)

    for j, ctx in enumerate(comb_ctxs, 1):
        respA = client.chat.completions.create(
            model=CHAT_MODEL, temperature=TEMP,
            messages=[{"role":"system","content":PROMPT_A_SYS},
                      {"role":"user","content":PROMPT_A_USER.format(ctx=ctx)}]
        )
        try:
            qlist = json.loads(respA.choices[0].message.content)
            qlist = [q.strip() for q in qlist if isinstance(q, str) and q.strip()]
        except Exception:
            qlist = []
        for q in qlist:
            out.append({"instruction": q, "input": "", "output": ""})

        respB = client.chat.completions.create(
            model=CHAT_MODEL, temperature=TEMP,
            messages=[{"role":"system","content":PROMPT_B_SYS},
                      {"role":"user","content":PROMPT_B_USER.format(ctx=ctx)}]
        )
        try:
            pairs = json.loads(respB.choices[0].message.content)
            pairs = [p for p in pairs if isinstance(p, dict)
                     and isinstance(p.get("instruction",""), str)
                     and isinstance(p.get("input",""), str)
                     and p["instruction"].strip() and p["input"].strip()]
        except Exception:
            pairs = []
        for p in pairs:
            out.append({"instruction": p["instruction"].strip(),
                        "input": p["input"].strip(),
                        "output": ""})
        if len(out) % SAVE_EVERY == 0:
            save_json_atomic(OUT_JSON, out)
        print(f"[progress][mixed] {j}/{len(comb_ctxs)} | items={len(out)}", flush=True)

    out = dedup_keep_unique(client, out)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[saved] {OUT_JSON} | {len(out)} items")

if __name__ == "__main__":
    main()


