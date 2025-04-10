from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import logging
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
import chromadb
import uvicorn
import re
from typing import List, Dict, Any
import json
from pathlib import Path
from langchain.vectorstores import BM25
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Medical Knowledge Base")

# Mount templates
templates = Jinja2Templates(directory="templates")

class Query(BaseModel):
    query: str
    conversation_history: List[Dict[str, str]] = []

class Source(BaseModel):
    content: str
    page: str
    source: str

class Response(BaseModel):
    answer: str
    sources: List[Source]

# Initialize conversation history
conversation_history = []

def clean_text(text: str) -> str:
    """Clean and format the text content."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove figure references
    text = re.sub(r'FIGURE\s+\d+\.\d+.*?(?=\n|$)', '', text)
    # Clean up line breaks
    text = text.replace('\n', ' ').strip()
    return text

def enhance_query(query: str, conversation_history: List[Dict[str, str]]) -> str:
    """Enhance the query for better search results."""
    # Add context from conversation history
    context = ""
    if conversation_history:
        last_exchange = conversation_history[-1]
        context = f"Previous question: {last_exchange.get('question', '')} "
        context += f"Previous answer: {last_exchange.get('answer', '')} "
    
    # Add medical context if not present
    if not any(word in query.lower() for word in ['treatment', 'manage', 'therapy', 'medicine']):
        if 'treat' in query.lower():
            query += " treatment management therapy medicine recommendations"
    
    return f"{context}Current question: {query}"

def format_response_with_sources(results: List[Any], query: str) -> Dict[str, Any]:
    """Format the response in a conversational way using the LLM."""
    llm = ChatOpenAI(temperature=0.7)
    
    # Extract and clean content from results
    formatted_results = []
    seen_content = set()  # To avoid duplicate content
    
    for doc in results:
        content = clean_text(doc.page_content)
        # Skip if content is too short or duplicate
        if len(content) < 20 or content in seen_content:
            continue
            
        seen_content.add(content)
        formatted_results.append({
            "content": content,
            "page": str(doc.metadata.get("page", "Unknown")),  # Convert page to string
            "source": doc.metadata.get("source", "Unknown")
        })
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a knowledgeable medical professional who has read Nelson's Essentials of Pediatrics. 
        Answer questions in a conversational, friendly manner while maintaining medical accuracy.
        Use the provided information from the textbook to support your answers.
        If you're not sure about something, say so rather than making assumptions.
        Keep your answers concise but informative."""),
        ("user", """Based on the following information from the textbook, please answer this question: {query}
        
        Relevant information:
        {context}
        
        Please provide a natural, conversational response that incorporates this information.""")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    # Combine all results into context
    context = "\n".join([r["content"] for r in formatted_results])
    
    response = chain.invoke({
        "query": query,
        "context": context
    })
    
    return {
        "answer": response,
        "sources": formatted_results[:3]  # Return top 3 sources
    }

def init_vectorstore():
    """Initialize the vector store from pre-indexed data."""
    try:
        # Check if we have pre-indexed data
        index_path = Path("index_data")
        if not index_path.exists():
            logger.error("Index data directory not found")
            return None
            
        vector_store_path = index_path / "vector_store"
        if not vector_store_path.exists():
            logger.error("Vector store not found")
            return None
            
        # Load the vector store
        vectorstore = Chroma(
            persist_directory=str(vector_store_path),
            embedding_function=embeddings
        )
        logger.info("Vector store loaded successfully")
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error initializing vector store: {str(e)}")
        return None

def init_bm25():
    """Initialize BM25 from pre-indexed data."""
    try:
        # Check if we have pre-indexed data
        index_path = Path("index_data")
        if not index_path.exists():
            logger.error("Index data directory not found")
            return None
            
        bm25_path = index_path / "bm25_index.pkl"
        if not bm25_path.exists():
            logger.error("BM25 index not found")
            return None
            
        # Load the BM25 index
        bm25 = BM25()
        bm25.load_index(str(bm25_path))
        logger.info("BM25 index loaded successfully")
        return bm25
        
    except Exception as e:
        logger.error(f"Error initializing BM25: {str(e)}")
        return None

def init_document_store():
    """Initialize document store from pre-indexed data."""
    try:
        # Check if we have pre-indexed data
        index_path = Path("index_data")
        if not index_path.exists():
            logger.error("Index data directory not found")
            return None
            
        doc_store_path = index_path / "document_store.pkl"
        if not doc_store_path.exists():
            logger.error("Document store not found")
            return None
            
        # Load the document store
        doc_store = DocumentStore()
        doc_store.load(str(doc_store_path))
        logger.info("Document store loaded successfully")
        return doc_store
        
    except Exception as e:
        logger.error(f"Error initializing document store: {str(e)}")
        return None

# Initialize components
embeddings = OpenAIEmbeddings()
vectorstore = init_vectorstore()
bm25 = init_bm25()
doc_store = init_document_store()

if not all([vectorstore, bm25, doc_store]):
    logger.error("Failed to initialize one or more components")
    raise RuntimeError("Failed to initialize required components")

# Initialize RAG system
rag_system = EnhancedRAGSystem(
    vectorstore=vectorstore,
    bm25=bm25,
    doc_store=doc_store
)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/query")
async def query_knowledge_base(query: Query):
    """Query the knowledge base with conversation history."""
    try:
        # Enhance the query with conversation history
        enhanced_query = enhance_query(query.query, query.conversation_history)
        
        # Get relevant documents
        results = vectorstore.similarity_search(enhanced_query, k=5)
        
        # Format the response with sources
        response = format_response_with_sources(results, query.query)
        
        # Update conversation history
        query.conversation_history.append({
            "question": query.query,
            "answer": response["answer"]
        })
        
        return Response(
            answer=response["answer"],
            sources=[Source(**source) for source in response["sources"]]
        )
    except Exception as e:
        logger.error(f"Error querying knowledge base: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 