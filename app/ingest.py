import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Constants
CHROMA_DIR = "chroma_db"
DATA_DIR = os.path.join("app", "blogs")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Step 1: Load data safely
docs = []
for filename in os.listdir(DATA_DIR):
    filepath = os.path.join(DATA_DIR, filename)
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            docs.append(Document(page_content=content, metadata={"source": filename}))
    except Exception as e:
        print(f"❌ Failed to load {filepath}: {e}")

if not docs:
    raise ValueError("No documents found. Please add .txt files to app/blogs/")

# Step 2: Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)

# Step 3: Create embeddings
embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# Step 4: Create and persist vector DB
vectordb = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory=CHROMA_DIR)
vectordb.persist()

print("✅ Ingestion completed successfully.")
