from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from enhanced_rag_v2 import EnhancedPhysicalTherapyRAG
import uvicorn
import logging
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Physical Therapy RAG API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system
try:
    rag_system = EnhancedPhysicalTherapyRAG()
    logger.info("RAG system initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG system: {str(e)}")
    raise

class Query(BaseModel):
    text: str
    chat_history: Optional[List[tuple]] = None

@app.post("/api/query")
async def process_query(query: Query):
    try:
        response = await rag_system.process_query(query.text, query.chat_history)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def get_status():
    try:
        doc_count = rag_system.get_document_count()
        return {
            "status": "healthy",
            "documents_loaded": doc_count,
            "embeddings_model": "OpenAI",
            "retrieval_type": "hybrid (semantic + BM25)",
            "reranking": "enabled (cross-encoder)"
        }
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files last to avoid conflicts with API routes
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True) 