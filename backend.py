from fastapi import FastAPI
from pydantic import BaseModel
from app import MultiAgentSystem

app = FastAPI()


class TaskRequest(BaseModel):
    objective: str


@app.post("/v1/execute")
async def run_pipeline(request: TaskRequest):
    system = MultiAgentSystem()

    state = {
        "query": request.objective,
        "research": "",
        "critique": "",
        "email": "",
        "logs": [],
    }

    state = system.research_agent(state)
    state = system.critic_agent(state)
    state = system.email_agent(state)

    return {
        "status": "success",
        "logs": state["logs"],
        "data": {
            "research": state["research"],
            "critique": state["critique"],
            "email": state["email"],
        },
    }
