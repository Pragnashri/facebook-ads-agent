import os
from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA

# 🔒 Disable Chroma Telemetry
os.environ["PERSIST_DIRECTORY_TELEMETRY_DISABLED"] = "true"

CHROMA_DIR = "chroma_db"
BLOGS_DIR = Path(__file__).parent / "blogs"

# ✅ Load LLM
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
llm = HuggingFacePipeline(pipeline=pipe)

# ✅ Load embedding model
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# ✅ Load or build vectorstore
def build_or_load_vectorstore():
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        # Load existing vectorstore
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
    
    # Else, build new vectorstore
    documents = []
    for file in BLOGS_DIR.glob("*.txt"):
        loader = TextLoader(str(file))
        documents.extend(loader.load())

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    vectordb = Chroma.from_documents(docs, embedding_function=embedding, persist_directory=CHROMA_DIR)
    vectordb.persist()
    return vectordb

# 🔁 Main load function
def load_rag_agent():
    vectordb = build_or_load_vectorstore()
    retriever = vectordb.as_retriever(search_kwargs={"k": 1})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain
