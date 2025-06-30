Facebook Ads Agent (Offline RAG System)
An offline, lightweight AI agent using RAG to answer Facebook ads questions from blog data. Runs locally with TinyLlama and FastAPI.

Features:
Offline & open-source
TinyLlama + LangChain + ChromaDB
FastAPI backend with Swagger UI
Local blog knowledge

Tech Stack
FastAPI, LangChain, ChromaDB
TinyLlama (via Hugging Face)
SentenceTransformers (MiniLM)

Structure
facebook-ads-agent/
├── app/
│   ├── main.py
│   └── agent.py
├── blogs/
├── requirements.txt
└── README.md

Setup
git clone https://github.com/Pragnashri/facebook-ads-agent.git
cd facebook-ads-agent
python -m venv venv
venv\Scripts\activate  
# Windows
pip install -r requirements.txt
uvicorn app.main:app --reload

API Usage
POST /ask
{ "question": "How do Facebook ads perform?" }
Test in: http://127.0.0.1:8000/docs

