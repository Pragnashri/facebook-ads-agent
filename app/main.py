from fastapi import FastAPI
from pydantic import BaseModel
from app.agent import load_rag_agent

app = FastAPI()
agent = load_rag_agent()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_question(query: Query):
    result = agent.invoke(query.question)  # or agent(query.question) if invoke() fails
    return {"answer": result["result"]}
