# Dataset Preparation Pipeline
Scripts to generate full datasets for LLM full retraining and per-character fine-tuning (LoRA/QLoRA). 
Uses instructions and some instruction+input pairs from *Cleaned Alpaca DS* and embeddings from **Alice in Wonderlands** to generate corresponding outputs.
In addition, generated instruction and instruction+input json file also used to generate answers for training dataset.

## Files
- `build_embeddings.py` - create multi-scale embeddings for *Alice in Wonderland*
- `fill_dataset_rag.py` - generate instruction–response dataset using RAG
- `questions_generator.py` - generate instruction and instruction+input DS using RAG to use it later in addition to alpaca DS

## Original data files
- `data/AliceInWonderlands.txt` - original textbook
- `data/alpaca_data_cleaned.json` - example ds from which instruction and input used for output generation
- `data/key.txt` — OpenAI API key

## Generated data files
- `data/alice_embeddings_{short,fine,medium,coarse}.npz`
- `data/alice_world_dataset.json` - DS for full NN retrain to be focused only on book (not a character DS)
- `data/alice_questions.state.json.` - state to safe continue generation
- `data/alice_questions.json` - instruction and instruction+input generated from book using RAG

### build_embeddings.py
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

### questions_generator.py
Generates new questions and situations from Alice book text chunks.
Uses medium (4 consecutive) and coarse (random 10%) embeddings for variety.

**Pass 1**: Sequential 4×medium chunks — ensures full book coverage.<br>
**Pass 2**: Random mixed 3×medium + 2×coarse chunks — adds variety and long-range context.<br>
Each context produces two types: 
- Instruction-only questions (no input).
- Instruction + Input pairs (situation + question).<br>

Outputs are empty — they’re filled later by `fill_ds_rag.py`.

## System Prompt
*under edit rn

## Code
```bash
python code/ds_prep/build_embeddings.py
python code/ds_prep/fill_dataset_rag.py
python code/ds_prep/questions_generator.py
```
## **Credit**  
This project uses the **Cleaned Alpaca** dataset (Apache 2.0) as a source for instruction/input pairs.  
Original Alpaca (before cleaning) is licensed CC BY-NC 4.0.  
