# EXPLANATION.md

## Architectural Decisions & Rationale

This document provides a comprehensive explanation of the engineering decisions behind **MEKA (Multi-Agent Expert Knowledge Assistant)**. The primary objective was to build a robust, locally-executable, multi-agent RAG (Retrieval-Augmented Generation) system that operates efficiently on CPU hardware without incurring external API costs.

---

## Table of Contents

1. [Core Language Model Selection](#1-core-language-model-selection)
2. [Orchestration Framework](#2-orchestration-framework)
3. [Retrieval Strategy](#3-retrieval-strategy)
4. [Relevance Filtering](#4-relevance-filtering)
5. [Vector Database](#5-vector-database)
6. [Backend Architecture](#6-backend-architecture)
7. [Document Ingestion](#7-document-ingestion)
8. [Trade-offs & Limitations](#8-trade-offs--limitations)

---

## 1. Core Language Model Selection

### **Choice: Qwen 2.5 (3B Parameters) via Ollama**

#### Why Qwen 2.5 and not Llama-3-8B?

**Hardware Constraints:**
- The system requirement mandated local execution on CPU infrastructure
- Llama-3-8B requires significantly more RAM (16GB+ recommended) and computational resources
- Token generation speed on standard CPUs would be prohibitively slow for interactive use

**Performance Characteristics:**
- Qwen 2.5 3B represents the current state-of-the-art for Small Language Models (SLMs)
- Demonstrates superior performance in reasoning and coding tasks compared to older, larger models
- Achieves this while being approximately 60% smaller than Llama-3-8B

**Practical Results:**
- The system runs smoothly with acceptable latency for real-time interaction
- Larger models would cause UI timeouts and degraded user experience
- Balanced trade-off between model capability and resource efficiency

---

## 2. Orchestration Framework

### **Choice: LangGraph (Stateful Graph Architecture)**

#### Why LangGraph instead of standard LangChain Chains?

**State Management Requirements:**
- Traditional RAG implementations follow a linear pipeline: `Retrieve → Generate`
- MEKA requires complex multi-step reasoning with interdependent stages:
  1. **Plan** the task decomposition
  2. **Retrieve** relevant documents
  3. **Rerank** documents for relevance
  4. **Summarize** and synthesize information

**Data Passing Between Agents:**
- Standard chains struggle with passing complex state objects (e.g., planning results, ranked documents)
- LangGraph's `StateGraph` provides a shared memory architecture where:
  - Each agent can read from and write to centralized state
  - Complex workflows can branch, merge, and loop based on conditions
  - State persistence enables proper audit trails and debugging

**Architectural Advantages:**
- Superior for agentic workflows requiring decision-making and conditional logic
- Enables better observability and debugging through state inspection
- More maintainable than nested callback chains

---

## 3. Retrieval Strategy

### **Choice: Hybrid Search (EnsembleRetriever with ChromaDB + BM25)**

#### The Problem with Vector-Only Search

**Semantic Limitations:**
- Pure vector search excels at capturing conceptual meaning
- However, it often misses exact keyword matches
- **Example:** Searching for "Error 505" might return general "Server Failures" content but miss the specific error code documentation

#### The Hybrid Solution

**Vector Search (ChromaDB):**
- Captures semantic similarity for conceptual queries
- Example: "How do I fix a server crash?" → Returns architecture and troubleshooting guides

**Keyword Search (BM25):**
- Exact lexical matching for precise terminology
- Example: "Error 505", "Article 32", "Section 4.2.1" → Returns specific references

**Ensemble Configuration:**
- 50/50 weight distribution between vector and keyword retrieval
- Ensures comprehensive coverage: breadth of understanding + precision of specific details
- Mitigates individual retriever weaknesses through complementary strengths

---

## 4. Relevance Filtering

### **Choice: Cross-Encoder Reranking (sentence-transformers/ms-marco-MiniLM-L-6-v2)**

#### Why Reranking is Critical

**The "Lost in the Middle" Problem:**
- Initial retrievers typically fetch 10-20 candidate documents
- Many contain irrelevant or tangential information
- Feeding all candidates to the LLM introduces noise, leading to:
  - Hallucinations
  - Context dilution
  - Increased latency

#### Bi-Encoder vs. Cross-Encoder

| Aspect | Bi-Encoder (Retriever) | Cross-Encoder (Reranker) |
|--------|------------------------|--------------------------|
| **Speed** | Fast (milliseconds) | Slower (seconds) |
| **Accuracy** | Approximate similarity | Precise relevance scoring |
| **Architecture** | Separate query/document encoding | Joint query-document analysis |
| **Use Case** | Broad candidate retrieval | Final relevance ranking |

#### Implementation Strategy

1. **High Recall Phase:** Retrieve 10-20 candidate documents using ensemble retriever
2. **High Precision Phase:** Cross-encoder scores each query-document pair (0.0 to 1.0 relevance)
3. **Filtering:** Select top 3 highest-scoring chunks for LLM context
4. **Result:** Dramatically improved answer quality with minimal hallucination

---

## 5. Vector Database

### **Choice: ChromaDB (langchain-chroma integration)**

#### Key Decision Factors

**Local & Serverless:**
- Unlike Milvus or Weaviate, ChromaDB operates as a local file-based database
- No Docker containers, external services, or complex infrastructure required
- Installation: `pip install chromadb` — immediate operability

**Developer Experience:**
- Zero-configuration setup for development and deployment
- Easy to version control and replicate across environments
- Ideal for academic and research contexts

**Feature Support:**
- Efficient metadata storage alongside vector embeddings
- Filtered search capabilities
- Native LangChain integration

**Limitations Acknowledged:**
- Not suitable for production-scale distributed systems
- Limited horizontal scalability compared to enterprise solutions
- Acceptable trade-off for the project's local-first architecture

---

## 6. Backend Architecture

### **Choice: Asynchronous Job Queue Pattern (FastAPI BackgroundTasks + Polling)**

#### The Latency Challenge

**Problem:**
- Multi-agent CPU workflows take 15-30 seconds to complete
- Standard synchronous HTTP requests timeout after 10-30 seconds
- Browser connections become unstable during long operations

#### Architectural Solution

**Implementation Pattern:**

```
┌─────────┐                    ┌──────────┐                 ┌─────────┐
│ Client  │ ─── POST /query ──>│  FastAPI │ ── Background ──>│  Agent  │
│         │ <── 202 + job_id ──│          │                 │ System  │
│         │                    │          │                 │         │
│         │ ─── GET /status ──>│          │ <── Update ─────│         │
│         │ <── Progress Log ──│          │                 │         │
│         │                    └──────────┘                 └─────────┘
│         │ (Polling every 1s)
└─────────┘
```

**Benefits:**

1. **UX Stability:** Frontend remains responsive during processing
2. **Live Progress:** Users see real-time logs ("Planner finished...", "Retrieving documents...")
3. **Professional Audit Trail:** Complete visibility into system operations
4. **Error Handling:** Graceful failure recovery and user notification

---

## 7. Document Ingestion

### **Choice: Multi-Format Loaders (PyPDF, Docx2txt, CSVLoader)**

#### Design Rationale

**Real-World Knowledge Base Reality:**
- Enterprise knowledge exists in heterogeneous formats
- A text-only system would be impractical for production use
- Requirements demand ingestion of:
  - PDF technical manuals
  - Word documents (reports, procedures)
  - CSV data tables
  - Plain text files

**Implementation Approach:**

```python
File Extension → Appropriate Loader
.pdf          → PyPDF/PDFPlumber
.docx         → Docx2txt
.csv          → CSVLoader
.txt/.md      → TextLoader
```

**Text Chunking Strategy:**
- Chunk size: 1000 characters
- Overlap: 200 characters
- Preserves context across chunk boundaries
- Optimized for 3B model context window constraints

---

## 8. Trade-offs & Limitations

### Engineering Honesty: What We Sacrificed and Why

#### **Speed vs. Quality**

**Trade-off:**
- Cross-encoder reranking adds 3-5 seconds to processing time
- Could be eliminated for faster responses

**Decision:**
- Prioritized answer accuracy over speed
- Users prefer correct answers over fast incorrect ones
- Professional use cases demand reliability

#### **Context Window Constraints**

**Challenge:**
- 3B model has limited context window (~4k-8k tokens)
- Compared to GPT-4 (128k tokens) or Claude (200k tokens)

**Mitigation Strategy:**
- Aggressive reranking to surface only most relevant content
- Chunk size optimization
- Risk: Model may lose context for very complex queries

#### **Memory Architecture**

**Current Implementation:**
- Conversation history stored in server RAM
- Stateless across server restarts

**Production Considerations:**
- Would require migration to Redis or PostgreSQL
- Trade-off made for architectural simplicity in demo/research context
- Sufficient for single-user or development scenarios

#### **Scalability Limits**

**Acknowledged Constraints:**
- ChromaDB not suitable for multi-tenant production
- CPU inference limits concurrent user capacity
- Acceptable for research, prototyping, and small-team deployment

---

## Conclusion

MEKA's architecture represents a series of deliberate engineering decisions optimized for:

1. **Local-First Operation:** No external API dependencies or costs
2. **CPU Efficiency:** Runs on standard hardware without GPU requirements
3. **Answer Quality:** Multi-stage filtering ensures high-precision responses
4. **Developer Accessibility:** Simple setup and transparent operation

Each component was selected to balance practical constraints with production-quality results, creating a system that is both academically interesting and practically deployable.

