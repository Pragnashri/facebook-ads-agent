import os
os.environ["PERSIST_DIRECTORY_TELEMETRY_DISABLED"] = "true"

from agent import load_rag_agent


agent = load_rag_agent()
print("✅ Agent loaded")

question = "How to write good Facebook ads?"
response = agent.invoke(question)
print("🧠 Response:", response)
