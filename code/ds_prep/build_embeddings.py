#!/usr/bin/env python3
"""
Create embeddings for the book 'AliceInWonderlands.txt' using OpenAI GPT-4o API.

Reads API key from ../data/key.txt
Saves results to ../data/alice_embeddings.npz (vectors + text chunks)
"""

import os
import numpy as np
import tiktoken
from openai import OpenAI

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(ROOT, "data")
BOOK_PATH = os.path.join(DATA_DIR, "AliceInWonderlands.txt")
KEY_PATH = os.path.join(DATA_DIR, "key.txt")
OUT_PATH = os.path.join(DATA_DIR, "alice_embeddings.npz")
OUT_BASENAME = os.path.join(DATA_DIR, "alice_embeddings")

MODEL_EMB = "text-embedding-3-small"
# CHUNK_TOKENS = 380  # maybe there is a point to make it 300
# OVERLAP_TOKENS = 80  # and then this about 70? for sentence consistency

CONFIGS = {
    "short":   {"chunk": 150, "overlap": 45},
    "fine":   {"chunk": 300, "overlap": 80},
    "medium": {"chunk": 550, "overlap": 100},
    "coarse": {"chunk": 900, "overlap": 150},
}

def load_key(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def read_book(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()

def chunk_text(text, target_tokens, overlap_tokens):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = []
    i = 0
    while i < len(tokens):
        j = min(len(tokens), i + target_tokens)
        chunk = enc.decode(tokens[i:j])
        chunks.append(chunk.strip())
        if j >= len(tokens):
            break
        i = j - overlap_tokens
    return [c for c in chunks if c]

def main():
    api_key = load_key(KEY_PATH)
    client = OpenAI(api_key=api_key)

    print(f"[info] Reading book: {BOOK_PATH}")
    text = read_book(BOOK_PATH)
    print(f"[info] Length: {len(text)} characters")

    print("[info] Building multi-granularity indexes...")
    for name, cfg in CONFIGS.items():
        ct, ov = cfg["chunk"], cfg["overlap"]
        print(f"[info] ({name}) chunk={ct}, overlap={ov}")
        chunks = chunk_text(text, ct, ov)
        print(f"[info] ({name}) chunks: {len(chunks)}")

        embeddings = []
        BATCH = 256
        for i in range(0, len(chunks), BATCH):
            batch = chunks[i:i + BATCH]
            resp = client.embeddings.create(model=MODEL_EMB, input=batch)
            embeddings.extend([d.embedding for d in resp.data])
            print(f"[progress:{name}] {i + len(batch)}/{len(chunks)}")

        embeddings = np.array(embeddings, dtype="float32")
        out_path = f"{OUT_BASENAME}_{name}.npz"
        np.savez(out_path,
                 embeddings=embeddings,
                 chunks=np.array(chunks),
                 meta=np.array({"chunk_tokens": ct, "overlap_tokens": ov}, dtype=object))
        print(f"[saved] ({name}) -> {out_path} | shape={embeddings.shape}")

if __name__ == "__main__":
    main()