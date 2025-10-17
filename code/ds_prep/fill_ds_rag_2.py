#!/usr/bin/env python3
"""
done:
I'll remake it, so when the limit of idk type of questions will be filled, rest of questions would be ONLY
remade to be questions regarding the alice in wonderland only.
In the beginning for each question with answers of type "idk" also should be converted to the question/instruction
related the Alice tale.

but it's still expencive. for whole alpaca 51k I will not use.
need to think about other embeddings model. maybe ollama to be local

I think the best way is to use the book directly and create questions
 for different text chunks of this book - it's much cheaper.
"""

import os, json, re
import numpy as np
import jsonlines
import faiss
from openai import OpenAI
from tqdm.auto import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(ROOT, "data")
KEY_PATH = os.path.join(DATA_DIR, "key.txt")
CONFIGS = {"short", "fine", "medium", "coarse"}

EMB_NPZ_BASENAME = os.path.join(DATA_DIR, "alice_embeddings")
NPZ_PATHS = {name: f"{EMB_NPZ_BASENAME}_{name}.npz" for name in CONFIGS}
ALPACA_JSON = os.path.join(DATA_DIR, "alpaca_data_cleaned.json")
QUESTIONS_JSON = os.path.join(DATA_DIR, "alice_questions.json")
OUT_JSON = os.path.join(DATA_DIR, "alice_dataset_expanded.json")

EMB_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOPK = 3  # but totally will use 9 to include surroundings of chosen
# maybe need think about Max Marginal Relevance and re-ranker clear not repetitive answer.
TAU = 0.30
TEMP = 0.25

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

IDK_PAT = re.compile(
    r"\b("
    r"i\s+(don['’]t|do\s+not)\s+(know|recall|understand|remember|recognize)|"
    r"i\s+(can['’]t|cannot|could\s+not)\s+(say|tell|recall|remember|help)|"
    r"i\s+haven['’]t\s+(heard|the\s+(faintest|foggiest|slightest|remotest)\s+idea|any\s+idea)|"
    r"i\s+have\s+no\s+(idea|notion|clue|knowledge)|"
    r"no\s+(idea|clue|knowledge|notion)|"
    r"not\s+(sure|certain|familiar|aware)|"
    r"never\s+(heard|seen|known|encountered)|"
    r"i\s+am\s+not\s+(sure|certain|aware|familiar|acquainted)|"
    r"how\s+should\s+i\s+know|"
    r"beats\s+me|"
    r"no\s+clue|"
    r"haven['’]t\s+a\s+clue|"
    r"i\s+couldn['’]t\s+say"
    r")\b",
    re.I,
)

REWRITE_SYS = (
    "Rewrite the user prompt so it fits ONLY within the world of “Alice’s Adventures in Wonderland.” "
    "Keep it concise and natural. If retheme is impossible, reply EXACTLY: __DONT_KNOW__"
)
REWRITE_USER_TMPL = (
    "Original instruction:\n{instr}\n\n"
    "Original input (may be empty):\n{inp}\n\n"
    "Produce ONE Wonderland-style pair as JSON:\n"
    '{{"instruction":"<brief in-world situation/setup>", "input":"<user question>"}}\n'
    "No answers. No meta. JSON only."
)

def looks_idk(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t: return False
    if len(t.split()) <= 8 and IDK_PAT.search(t):
        return True
    return bool(IDK_PAT.search(t))

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
    if path.endswith(".jsonl"):
        items = []
        with jsonlines.open(path, "r") as r:
            for o in r:
                items.append(o)
        return items
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else data.get("data", [])

def embed_query(client, text):
    resp = client.embeddings.create(model=EMB_MODEL, input=[text])
    return np.array(resp.data[0].embedding, dtype="float32")[None, :]

def retrieve_old(index, qvec, chunks, topk=TOPK):
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
    selected, pool = [], cands[:]
    if not pool: return selected
    mx = max(c["score"] for c in pool) or 1e-6
    for c in pool: c["score"] = c["score"] / mx
    while pool and len(selected) < final_k:
        if not selected:
            best = max(pool, key=lambda c: c["score"])
        else:
            def mmr_val(c):
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

    indices = {}
    chunks_map = {}
    for name, path in NPZ_PATHS.items():
        X, chunks = load_embeddings(path)
        indices[name] = build_index(X)
        chunks_map[name] = chunks


    items = []
    if os.path.exists(QUESTIONS_JSON):
        with open(QUESTIONS_JSON, "r", encoding="utf-8") as f:
            q_items = json.load(f)
            if isinstance(q_items, list):
                items.extend(q_items)
                print(f"[info] added questions from alice_questions.json: {len(q_items)}")

    alpaca_items = load_alpaca(ALPACA_JSON)
    print(f"[info] added questions from alpaca_data_cleaned: {len(alpaca_items)}")
    alpaca_items = alpaca_items[:10000]

    items.extend(alpaca_items)
    print(f"[info] total items to process: {len(items)}")

    existing_pairs = set()
    if os.path.exists(OUT_JSON):
        with jsonlines.open(OUT_JSON, "r") as r:
            for o in r:
                existing_pairs.add((o.get("instruction", ""), o.get("input", "")))
    print(f"[resume] existing items: {len(existing_pairs)}")

    out_path = OUT_JSON
    done = 0
    with jsonlines.open(out_path, "a") as w:
        pbar = tqdm(items, total=len(items), desc="filling", unit="q")
        for it in pbar:

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
                use_ctx = []
            else:
                # MMR filter to FINAL_K across pooled candidates
                picked = mmr_select(sorted(candidates, key=lambda c: c["score"], reverse=True),
                                    FINAL_K, MMR_LAMBDA)
                use_ctx = expand_neighbors_for(picked, chunks_map, WINDOW_BY)

            user_msg = make_user_prompt_improved(use_ctx, instr, inp)
            resp = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=TEMP,
            )
            out_text = resp.choices[0].message.content.strip()

            if looks_idk(out_text):
                if len(out_text) > 22:
                    w.write({
                        "instruction": instr,
                        "input": inp,
                        "output": out_text
                    })
                r_resp = client.chat.completions.create(
                    model="gpt-4o", temperature=0.2,
                    messages=[
                        {"role": "system", "content": REWRITE_SYS},
                        {"role": "user", "content": REWRITE_USER_TMPL.format(instr=instr, inp=inp)}
                    ]
                )
                r_txt = (r_resp.choices[0].message.content or "").strip()
                try:
                    r_json = json.loads(r_txt)
                    rinstr = (r_json.get("instruction") or "").strip()
                    rinp = (r_json.get("input") or "").strip()
                except Exception as e:
                    print(f"[rewrite failed] {r_txt} | {e}")
                    rinstr, rinp = "", ""
                if not r_txt or r_txt.strip() == "__DONT_KNOW__":
                    rinstr, rinp = "", ""
                    continue

                if rinstr and rinp and rinstr != instr:
                    r_query = (rinstr + "\n" + rinp).strip()
                    rqvec = embed_query(client, r_query)

                    rcands = []
                    for name, index in indices.items():
                        ridxs, rsims = retrieve(index, rqvec, chunks_map[name], topk=PER_INDEX_TOPK[name])
                        for i2, s2 in zip(ridxs, rsims):
                            if i2 < 0: continue
                            rcands.append(
                                {"name": name, "idx": i2, "score": float(s2), "chunk": str(chunks_map[name][i2])})

                    rmax_sim = max([c["score"] for c in rcands], default=0.0)
                    if rmax_sim >= TAU:
                        rpicked = mmr_select(sorted(rcands, key=lambda c: c["score"], reverse=True), FINAL_K,
                                             MMR_LAMBDA)
                        r_ctx = expand_neighbors_for(rpicked, chunks_map, WINDOW_BY)
                    else:
                        r_ctx = []

                    r_user_msg = make_user_prompt_improved(r_ctx, rinstr, rinp)
                    r_ans = client.chat.completions.create(
                        model=CHAT_MODEL, temperature=TEMP,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": r_user_msg}
                        ]
                    )
                    r_out = (r_ans.choices[0].message.content or "").strip()

                    if r_out and not looks_idk(r_out) and len(out_text.split()) <= 15:
                        continue
                    instr, inp, out_text = rinstr, rinp, r_out
                    w.write({
                        "instruction": instr,
                        "input": inp,
                        "output": out_text
                    })
            else:
                w.write({
                    "instruction": instr,
                    "input": inp,
                    "output": out_text
                })

            done += 1
            pbar.set_postfix(written=done)
            if done % 50 == 0:
                try:
                    w._fp.flush()
                    os.fsync(w._fp.fileno())
                except Exception:
                    pass

            # if done % 50 == 0 or done == n:
            #     print(f"[progress] {done}/{n}")

    print(f"[saved] {out_path}")

if __name__ == "__main__":
    main()