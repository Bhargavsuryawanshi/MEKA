import operator
from typing import Annotated, List, TypedDict
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END

from backend.retrieval import get_hybrid_retriever, rerank_documents
from backend.config import LLM_MODEL

# --- SETUP ---
llm = ChatOllama(model=LLM_MODEL, temperature=0)
web_search = DuckDuckGoSearchRun()

# --- STATE ---
class AgentState(TypedDict):
    question: str
    plan: List[str]
    retrieved_docs: List[str] # Raw docs
    ranked_context: List[str] # Reranked docs
    web_evidence: str
    final_answer: str
    logs: Annotated[List[str], operator.add] # Append-only logs

# --- AGENT 1: ORCHESTRATOR (PLANNER) ---
def planner(state: AgentState):
    print("--> Agent: Planner")
    query = state['question']
    
    prompt = f"""
    You are a technical planner. Break down this complex query into 3 distinct search/reasoning steps.
    Return ONLY the steps as a numbered list.
    Query: {query}
    """
    response = llm.invoke(prompt)
    steps = response.content.split("\n")
    
    return {
        "plan": steps,
        "logs": ["planner: Decomposed query into sub-tasks."]
    }

# --- AGENT 2: RETRIEVER (HYBRID + WEB) ---
def retriever(state: AgentState):
    print("--> Agent: Retriever")
    query = state['question']
    
    # 1. Internal Documents (Hybrid)
    hybrid_engine = get_hybrid_retriever()
    docs = hybrid_engine.invoke(query)
    
    # 2. External Web (Supplemental)
    web_res = web_search.invoke(query)
    
    return {
        "retrieved_docs": docs, # Keep objects for reranker
        "web_evidence": web_res,
        "logs": [f"retriever: Found {len(docs)} internal docs and web info."]
    }

# --- AGENT 3: RERANKER (CROSS-ENCODER) ---
def reranker_agent(state: AgentState):
    print("--> Agent: Reranker")
    docs = state['retrieved_docs']
    query = state['question']
    
    # Apply Cross-Encoder Logic
    top_docs = rerank_documents(query, docs, top_k=3)
    
    # Convert to text for the LLM
    context_text = [d.page_content for d in top_docs]
    
    return {
        "ranked_context": context_text,
        "logs": ["reranker: Filtered noise using Cross-Encoder."]
    }

# --- AGENT 4: SUMMARIZER (REASONING) ---
def summarizer(state: AgentState):
    print("--> Agent: Summarizer")
    
    prompt = ChatPromptTemplate.from_template("""
    You are an Expert Assistant. Answer the user query based strictly on the provided context.
    
    Query: {question}
    
    Strategic Plan: {plan}
    
    Internal Evidence (Highest Confidence):
    {internal}
    
    Web Evidence (Supplemental):
    {web}
    
    Instructions:
    1. Provide a direct answer.
    2. Cite which internal document the info came from if possible.
    3. Explain your reasoning.
    """)
    
    chain = prompt | llm
    res = chain.invoke({
        "question": state['question'],
        "plan": "\n".join(state['plan']),
        "internal": "\n\n".join(state['ranked_context']),
        "web": state['web_evidence']
    })
    
    return {
        "final_answer": res.content,
        "logs": ["summarizer: Synthesized final answer."]
    }

# --- GRAPH BUILDER ---
workflow = StateGraph(AgentState)

workflow.add_node("planner", planner)
workflow.add_node("retriever", retriever)
workflow.add_node("reranker", reranker_agent)
workflow.add_node("summarizer", summarizer)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "retriever")
workflow.add_edge("retriever", "reranker")
workflow.add_edge("reranker", "summarizer")
workflow.add_edge("summarizer", END)

app = workflow.compile()