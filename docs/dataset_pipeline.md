# Dataset Preparation Pipeline

## Files
- `build_embeddings.py` - create multi-scale embeddings for *Alice in Wonderland*
- `fill_dataset_rag.py` - generate instruction–response dataset using RAG

## Input
- `data/AliceInWonderlands.txt` - original textbook
- `data/alpaca_data_cleaned.json` - example ds from which instruction and input used for output generation
- `data/key.txt` — OpenAI API key

## Output
- `data/alice_embeddings_{short,fine,medium,coarse}.npz`
- `data/alice_filled.json`

## build_embeddings.py
| Param | Description | Default |
|--------|--------------|----------|
| `MODEL_EMB` | Embedding model | `text-embedding-3-small` |
| `CONFIGS` | Chunk/overlap sizes | short:150/45, fine:300/80, medium:550/100, coarse:900/150 |
| `BATCH` | Embedding batch size | 256 |

Creates `.npz` with arrays `embeddings`, `chunks`, and `meta`.

Process:
1. Load 4 embedding indexes
2. Retrieve Top-3 chunks per index  
3. Apply MMR returning `FINAL_K` number of chunks
4. Expand neighbors with `WINDOW_BY`
5. Generate output with GPT-4o
6. Save first 100 items to `alice_filled.json`

## System Prompt
*under edit rn

## Run
```bash
python code/ds_prep/build_embeddings.py
python code/ds_prep/fill_dataset_rag.py
