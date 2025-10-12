# Alice-in-Wonderland
AI for few characters from Alice's Adventures in Wonderland by Lewis Carroll in different situation. Helper gpt4o API
Final stage: full tiny-llama retrain for basic *world* chat, and LoRA per character: separate DS for few characters in different situations

## Current state
Uses GPT-4o API to generate dataset from the book text with multi-scale embeddings and RAG.  
First 100 dataset samples generated for testing.

## Next steps
- finish dataset generation
- retrain Tiny-Llama fully on world-only data
- prepare LoRA adapters (one per character)
- add dialogue system for character testing
- evaluate consistency and responses
- prepare Cpp-based application

## Project structure
code/
 └── ds_prep/
      ├── build_embeddings.py
      └── fill_dataset_rag.py
data/
 ├── AliceInWonderlands.txt
 ├── alpaca_data_cleaned.json
 ├── alice_embeddings_*.npz
 └── alice_filled.json
docs/
 └── dataset_pipeline.md
