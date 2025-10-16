#!/usr/bin/env python3
import os, json, random, sys, atexit, signal
import numpy as np
from openai import OpenAI
import faiss

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(ROOT, "data")
KEY_PATH = os.path.join(DATA_DIR, "key.txt")

NPZ_MED = os.path.join(DATA_DIR, "alice_embeddings_medium.npz")
NPZ_COA = os.path.join(DATA_DIR, "alice_embeddings_coarse.npz")
OUT_JSON = os.path.join(DATA_DIR, "alice_questions.json")
STATE_PATH = os.path.join(DATA_DIR, "alice_questions.state.json")

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
    # no NaN/Inf, no divide-by-zero
    V = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    V = V / norms
    return V

def dedup_keep_unique(client, items, threshold=DEDUP_SIM):
    if not items: return items
    qs = [i["instruction"] + " ||| " + (i["input"] or "") for i in items]
    V = embed_texts(client, qs)
    keep, kept = [], None
    for i, it in enumerate(items):
        v = V[i:i+1]
        if kept is None:
            kept = v.copy(); keep.append(it); continue

        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype("float32", copy=False)
        kept = np.nan_to_num(kept, nan=0.0, posinf=0.0, neginf=0.0).astype("float32", copy=False)
        if kept.size == 0:
            sims = np.array([], dtype="float32")
        else:
            sims = (v @ kept.T).astype("float32")[0]

        if float(np.max(sims)) < threshold:
            keep.append(it)
            kept = np.vstack([kept, v])
    return keep

def medium_only_contexts(chunks_med, span=4, step=1):
    L = len(chunks_med)
    for i in range(0, max(0, L - span + 1), step):
        yield "\n\n".join(str(ch) for ch in chunks_med[i:i+span])

def plan_random_pairs(Lm, Lc, fraction, seed):
    rng = random.Random(seed)
    med_starts = list(range(0, max(0, Lm - 2)))
    coa_starts = list(range(0, max(0, Lc - 1)))
    n = max(1, int(round(fraction * len(coa_starts))))
    n = min(n, len(med_starts), len(coa_starts))
    sel_med = rng.sample(med_starts, n)
    sel_coa = rng.sample(coa_starts, n)
    return sel_med, sel_coa

def load_state(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            s = json.load(f)
            s.setdefault("written_since_save", 0)
            s.setdefault("last_saved", 0)
            return s
    return {
        "rand_fraction": RAND_FRACTION_COMBINED,
        "rand_seed": RAND_SEED,
        "medium_pos": 0,
        "mixed_med_idx": [],
        "mixed_coa_idx": [],
        "mixed_pos": 0,
        "written_since_save": 0,
        "last_saved": 0
    }

def save_state(path, state):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def save_json_atomic(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def load_existing_list(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list): return data
        except Exception:
            pass
    return []

def strip_code_fences(s: str) -> str:
    t = s.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:].lstrip()
    return t

def parse_json_array(text):
    try:
        return json.loads(text)
    except Exception:
        return json.loads(strip_code_fences(text))

def main():
    client = OpenAI(api_key=load_key(KEY_PATH))
    C_med = load_npz(NPZ_MED)
    C_coa = load_npz(NPZ_COA)

    med_ctxs = list(medium_only_contexts(C_med, span=4, step=1))
    print(f"[info] medium-only contexts: {len(med_ctxs)}", flush=True)

    state = load_state(STATE_PATH)
    Lm, Lc = len(C_med), len(C_coa)
    if not state["mixed_med_idx"] or not state["mixed_coa_idx"]:
        sel_med, sel_coa = plan_random_pairs(Lm, Lc, state["rand_fraction"], state["rand_seed"])
        state["mixed_med_idx"] = sel_med
        state["mixed_coa_idx"] = sel_coa
        state["mixed_pos"] = 0
        save_state(STATE_PATH, state)
    print(f"[info] mixed pairs planned: {len(state['mixed_med_idx'])}", flush=True)

    out = load_existing_list(OUT_JSON)
    print(f"[info] existing items in {os.path.basename(OUT_JSON)}: {len(out)}", flush=True)

    def do_save():
        save_json_atomic(OUT_JSON, out)
        save_state(STATE_PATH, state)
        print(f"[save] {OUT_JSON} | items={len(out)} | medium_pos={state['medium_pos']} | mixed_pos={state['mixed_pos']}", flush=True)

    atexit.register(do_save)
    def _sig_handler(signum, frame):
        print(f"[signal] caught {signum}, saving...", flush=True)
        do_save()
        sys.exit(1)
    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    def add_items(new_items):
        nonlocal out, state
        if not new_items:
            return
        prev_len = len(out)
        out.extend(new_items)
        if prev_len == 0:
            do_save()
            state["written_since_save"] = 0
            state["last_saved"] = len(out)
            return
        state["written_since_save"] += len(new_items)
        if len(out) - state.get("last_saved", 0) >= SAVE_EVERY:
            do_save()
            state["written_since_save"] = 0
            state["last_saved"] = len(out)

    for i in range(state["medium_pos"], len(med_ctxs)):
        ctx = med_ctxs[i]

        respA = client.chat.completions.create(
            model=CHAT_MODEL, temperature=TEMP,
            messages=[{"role":"system","content":PROMPT_A_SYS},
                      {"role":"user","content":PROMPT_A_USER.format(ctx=ctx)}]
        )
        try:
            qlist = parse_json_array(respA.choices[0].message.content)
            qlist = [q.strip() for q in qlist if isinstance(q, str) and q.strip()]
        except Exception:
            qlist = []
        add_items([{"instruction": q, "input": "", "output": ""} for q in qlist])

        respB = client.chat.completions.create(
            model=CHAT_MODEL, temperature=TEMP,
            messages=[{"role":"system","content":PROMPT_B_SYS},
                      {"role":"user","content":PROMPT_B_USER.format(ctx=ctx)}]
        )
        try:
            pairs = parse_json_array(respB.choices[0].message.content)
            pairs = [p for p in pairs if isinstance(p, dict)
                     and isinstance(p.get("instruction",""), str)
                     and isinstance(p.get("input",""), str)
                     and p["instruction"].strip() and p["input"].strip()]
        except Exception:
            pairs = []
        add_items([{"instruction": p["instruction"].strip(),
                    "input": p["input"].strip(),
                    "output": ""} for p in pairs])

        state["medium_pos"] = i + 1
        print(f"[progress][medium] {state['medium_pos']}/{len(med_ctxs)} | items={len(out)}", flush=True)

    for k in range(state["mixed_pos"], len(state["mixed_med_idx"])):
        i = state["mixed_med_idx"][k]
        j = state["mixed_coa_idx"][k]
        if i + 2 >= len(C_med) or j + 1 >= len(C_coa):
            state["mixed_pos"] = k + 1
            continue

        ctx = "\n\n".join([str(C_med[i]), str(C_med[i+1]), str(C_med[i+2]),
                           str(C_coa[j]), str(C_coa[j+1])])

        respA = client.chat.completions.create(
            model=CHAT_MODEL, temperature=TEMP,
            messages=[{"role":"system","content":PROMPT_A_SYS},
                      {"role":"user","content":PROMPT_A_USER.format(ctx=ctx)}]
        )
        try:
            qlist = parse_json_array(respA.choices[0].message.content)
            qlist = [q.strip() for q in qlist if isinstance(q, str) and q.strip()]
        except Exception:
            qlist = []
        add_items([{"instruction": q, "input": "", "output": ""} for q in qlist])

        respB = client.chat.completions.create(
            model=CHAT_MODEL, temperature=TEMP,
            messages=[{"role":"system","content":PROMPT_B_SYS},
                      {"role":"user","content":PROMPT_B_USER.format(ctx=ctx)}]
        )
        try:
            pairs = parse_json_array(respB.choices[0].message.content)
            pairs = [p for p in pairs if isinstance(p, dict)
                     and isinstance(p.get("instruction",""), str)
                     and isinstance(p.get("input",""), str)
                     and p["instruction"].strip() and p["input"].strip()]
        except Exception:
            pairs = []
        add_items([{"instruction": p["instruction"].strip(),
                    "input": p["input"].strip(),
                    "output": ""} for p in pairs])

        state["mixed_pos"] = k + 1
        print(f"[progress][mixed] {state['mixed_pos']}/{len(state['mixed_med_idx'])} | items={len(out)}", flush=True)

    out[:] = dedup_keep_unique(client, out, threshold=DEDUP_SIM)
    save_json_atomic(OUT_JSON, out)
    state["last_saved"] = len(out)
    save_state(STATE_PATH, state)
    print(f"[saved] {OUT_JSON} | total items={len(out)}", flush=True)

if __name__ == "__main__":
    main()