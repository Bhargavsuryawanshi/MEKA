from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from backend.graph import app as graph_app
import uuid
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="MEKA API")

# Allow Frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for jobs (In production, use Redis/Postgres)
jobs = {}

class QueryRequest(BaseModel):
    query: str

def process_graph(job_id: str, query: str):
    """Runs the LangGraph workflow in background."""
    jobs[job_id]["status"] = "running"
    
    initial_state = {
        "question": query,
        "plan": [],
        "retrieved_docs": [],
        "ranked_context": [],
        "web_evidence": "",
        "final_answer": "",
        "logs": []
    }
    
    try:
        # Stream events to capture intermediate logs
        final_state = initial_state
        for event in graph_app.stream(initial_state):
            # Update state as agents finish
            for key, value in event.items():
                if "logs" in value:
                    jobs[job_id]["logs"].extend(value["logs"])
                # Save intermediate data
                if "plan" in value: jobs[job_id]["plan"] = value["plan"]
                if "ranked_context" in value: jobs[job_id]["context"] = value["ranked_context"]
                if "final_answer" in value: jobs[job_id]["answer"] = value["final_answer"]
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["answer"] = jobs[job_id]["answer"] # Ensure final answer is set
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)

@app.post("/query")
async def submit_query(req: QueryRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "pending", 
        "logs": [], 
        "plan": [], 
        "context": [], 
        "answer": None
    }
    
    # Start agent workflow in background
    background_tasks.add_task(process_graph, job_id, req.query)
    return {"job_id": job_id}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    return jobs.get(job_id, {"status": "not_found"})

@app.get("/history")
async def get_history():
    # Filter only completed jobs
    return {k: v for k, v in jobs.items() if v["status"] == "completed"}