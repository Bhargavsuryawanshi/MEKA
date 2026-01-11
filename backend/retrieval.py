from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
from backend.config import VECTOR_DB_DIR, EMBEDDING_MODEL, RERANKER_MODEL

# Load Cross-Encoder Model once (it's heavy)
print("Loading Reranker Model...")
reranker = CrossEncoder(RERANKER_MODEL)

def get_hybrid_retriever():
    """Builds the search engine combining Vector (Meaning) and BM25 (Keyword)."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    vectorstore = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
    
    # 1. Semantic Search
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    # 2. Keyword Search (BM25)
    # We fetch docs to build the keyword index
    all_docs = vectorstore.get()['documents']
    if not all_docs:
        return vector_retriever # Fallback if empty
        
    # Quick reconstruction of Document objects for BM25
    from langchain_core.documents import Document
    docs = [Document(page_content=t) for t in all_docs]
    keyword_retriever = BM25Retriever.from_documents(docs)
    keyword_retriever.k = 10
    
    # 3. Ensemble (Hybrid)
    ensemble = EnsembleRetriever(
        retrievers=[vector_retriever, keyword_retriever],
        weights=[0.5, 0.5]
    )
    return ensemble

def rerank_documents(query, documents, top_k=5):
    """
    Uses a Cross-Encoder to score relevance.
    Cross-Encoders are more accurate than Vectors for specific relevance.
    """
    if not documents:
        return []
        
    doc_texts = [d.page_content for d in documents]
    pairs = [[query, text] for text in doc_texts]
    
    # Predict scores
    scores = reranker.predict(pairs)
    
    # Sort documents by score
    scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    
    # Return top K docs
    return [doc for doc, score in scored_docs[:top_k]]