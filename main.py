import os
import httpx
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration from environment variables or defaults
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
DOCUMENTS_DIR = "./documents"

# Allow your frontend URL (React usually runs on http://localhost:3000)
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

# Initialize the Ollama client
llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_URL)
# Initialize the embeddings model
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)

# Vector store variable
vectorstore = None


def create_vector_store():
    """Loads documents and creates a ChromaDB vector store for RAG."""
    global vectorstore
    documents = []
    
    print("Step 1: Checking for documents in ./documents directory...")
    if not os.path.exists(DOCUMENTS_DIR):
        print("Documents directory not found. Skipping RAG functionality.")
        return
        
    for filename in os.listdir(DOCUMENTS_DIR):
        file_path = os.path.join(DOCUMENTS_DIR, filename)
        if filename.endswith(".pdf"):
            print(f"Loading PDF file: {filename}")
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif filename.endswith(".docx"):
            print(f"Loading DOCX file: {filename}")
            loader = Docx2txtLoader(file_path)
            documents.extend(loader.load())
    
    if not documents:
        print("No documents found. RAG functionality will be disabled.")
        return

    print(f"Found {len(documents)} total documents.")
    
    print("Step 2: Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    
    print(f"Step 3: Creating embeddings and building the vector store. This may take a while...")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    print("Vector store creation complete.")


async def check_ollama_status():
    """Checks if the Ollama server is running and accessible."""
    try:
        print(f"Checking connection to Ollama at {OLLAMA_URL}...")
        async with httpx.AsyncClient() as client:
            response = await client.get(OLLAMA_URL)
            response.raise_for_status()
        print("Ollama server is running.")
        return True
    except httpx.HTTPError as e:
        print(f"Ollama server not running or unreachable. HTTP Error: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while checking Ollama status: {e}")
        return False


# Lifespan context manager for startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    is_ollama_ready = await check_ollama_status()
    if is_ollama_ready:
        print("Starting RAG setup in a background thread...")
        # Run the blocking function in a separate thread to not freeze the event loop
        await asyncio.to_thread(create_vector_store)
        print("RAG setup complete.")
    else:
        print("Backend server is running, but Ollama is not available. Check your Ollama setup and try again.")
    print("Application startup complete. Ready for requests.")
    yield


app = FastAPI(title="Gemma Chatbot Proxy", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    messages: list
    stream: bool = False


class RAGChatRequest(BaseModel):
    query: str


@app.post("/chat")
async def chat(req: ChatRequest):
    """Proxy chat requests to Ollama's /api/chat endpoint."""
    payload = {
        "model": LLM_MODEL,
        "messages": req.messages,
        "stream": req.stream,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            r = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
            r.raise_for_status()
            # The Ollama API response is {..., "message": {"role": "assistant", "content": "..."}}
            # We extract the content and return it in a 'reply' key for the frontend.
            return {"reply": r.json()["message"]["content"]}
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=f"Ollama API Error: {str(e)}")


@app.post("/rag_chat")
async def rag_chat(req: RAGChatRequest):
    """Handles RAG-based chat requests using the document vector store."""
    if not vectorstore:
        raise HTTPException(status_code=503, detail="RAG is not initialized. Please ensure documents are in ./documents and Ollama is running.")
    
    print(f"Received RAG query: {req.query}")
    
    retriever = vectorstore.as_retriever()
    
    template = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know.
    
    Question: {input}
    Context: {context}
    
    Answer:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    try:
        response = retrieval_chain.invoke({"input": req.query})
        # The response structure from LangChain is a dictionary with the final answer
        # The key is typically 'answer' or 'output'. We'll use 'answer' here.
        return {"reply": response["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG Chain Error: {str(e)}")


@app.get("/api/health")
async def health():
    """Simple health check endpoint."""
    return {"status": "ok"}