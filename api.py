from fastapi import FastAPI
from pydantic import BaseModel
from app import run_orchestrator

app = FastAPI(title="Multi-Agent Orchestration API")

class QueryInput(BaseModel):
    query: str

@app.post("/run-agents")
def run_agents(data: QueryInput):
    return run_orchestrator(data.query)