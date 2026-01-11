import os

# --- LLM CONFIG ---
LLM_MODEL = "qwen2.5:3b"  # Runs fast on CPU
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # Small, fast embeddings
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2" # Specialized for reranking

# --- PATHS ---
DATA_DIR = "./data"
VECTOR_DB_DIR = "./storage/chroma_db"