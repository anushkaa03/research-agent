from fastapi import FastAPI
from pydantic import BaseModel

from BACKEND.app.graph import research_graph
from BACKEND.app.memory import memory

app = FastAPI()


class ResearchRequest(BaseModel):
    query: str


@app.get("/")
def home():

    return {
        "message": "Autonomous Research Agent Running"
    }


@app.post("/research")
def research(request: ResearchRequest):

    result = research_graph.invoke(
        {
            "query": request.query,
            "plan": "",
            "search_results": [],
            "extracted_text": [],
            "final_answer": ""
        }
    )

    return {
        "query": request.query,
        "plan": result["plan"],
        "answer": result["final_answer"]
    }


@app.get("/memory")
def get_memory():

    return {
        "stored_facts": memory.get_facts()
    }