import os
import glob
from langchain_community.document_loaders import (
    PyPDFLoader, 
    Docx2txtLoader, 
    TextLoader, 
    CSVLoader, 
    JSONLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from backend.config import DATA_DIR, VECTOR_DB_DIR, EMBEDDING_MODEL

def load_file(filepath):
    """Selects the right loader based on file extension."""
    ext = os.path.splitext(filepath)[1].lower()
    
    try:
        if ext == ".pdf":
            return PyPDFLoader(filepath).load()
        elif ext == ".docx":
            return Docx2txtLoader(filepath).load()
        elif ext == ".txt":
            return TextLoader(filepath).load()
        elif ext == ".csv":
            return CSVLoader(filepath).load()
        elif ext == ".json":
            # Requires 'jq' installed or simple schema. Using text mode for generic JSON.
            return TextLoader(filepath).load() 
        else:
            print(f"Skipping unsupported file: {filepath}")
            return []
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return []

def run_ingestion():
    print("--- 1. Scanning Data Directory ---")
    all_docs = []
    
    # Walk through all files in data directory
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            filepath = os.path.join(root, file)
            print(f"Loading: {file}")
            docs = load_file(filepath)
            all_docs.extend(docs)

    if not all_docs:
        print("No documents found. Please add files to 'data/' folder.")
        return

    print(f"--- 2. Splitting {len(all_docs)} Documents ---")
    # Chunking handles long context
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)

    print("--- 3. Creating/Updating Vector Store ---")
    # Uses HuggingFace Embeddings (CPU friendly)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=VECTOR_DB_DIR
    )
    print("--- Ingestion Complete ---")

if __name__ == "__main__":
    run_ingestion()