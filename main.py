from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Medical RAG System API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize conversation history
conversation_history = []

# Define models
class Query(BaseModel):
    question: str = Field(..., description="The user's question")
    conversation_history: List[dict] = Field(default_factory=list, description="Previous conversation messages")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the symptoms of diabetes?",
                "conversation_history": []
            }
        }

class Response(BaseModel):
    answer: str = Field(..., description="The system's response")
    sources: List[str] = Field(..., description="List of source documents used")
    conversation_history: List[dict] = Field(..., description="Updated conversation history")

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Common symptoms of diabetes include...",
                "sources": ["source1.pdf", "source2.pdf"],
                "conversation_history": []
            }
        }

# Add root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to the Medical RAG System API",
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "query": "/query"
        }
    }

# Add health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Add chat endpoint
@app.get("/chat")
async def chat_ui():
    return {"message": "Chat UI endpoint - implement your chat interface here"}

# Enhance query with conversation history
def enhance_query(query: str, conversation_history: List[dict]) -> str:
    if not conversation_history:
        return query
    
    # Extract previous questions and answers
    context = "\n".join([
        f"Previous question: {msg['question']}\nPrevious answer: {msg['answer']}"
        for msg in conversation_history[-3:]  # Use last 3 exchanges for context
    ])
    
    return f"""Based on our previous conversation:
{context}

Current question: {query}"""

# Format response with sources
def format_response_with_sources(response: str, sources: List[str]) -> str:
    # Create a conversational response using the language model
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful medical assistant. 
        Format the response in a friendly, conversational way while maintaining accuracy.
        If there are sources, mention them naturally in the conversation."""),
        ("human", f"Here's the raw response: {response}\nSources: {', '.join(sources)}\nPlease format this into a natural conversation.")
    ])
    
    chain = prompt | ChatOpenAI(temperature=0.7) | StrOutputParser()
    return chain.invoke({})

# Query the knowledge base
@app.post("/query", response_model=Response)
async def query_knowledge_base(query: Query):
    try:
        # Enhance the query with conversation history
        enhanced_query = enhance_query(query.question, query.conversation_history)
        
        # Here you would typically:
        # 1. Query your vector store
        # 2. Get relevant documents
        # 3. Generate a response
        
        # For now, return a mock response
        mock_response = "I'm sorry, but the knowledge base is not fully initialized yet. Please try again later."
        mock_sources = ["medical_database.pdf"]
        
        # Format the response
        formatted_response = format_response_with_sources(mock_response, mock_sources)
        
        # Update conversation history
        updated_history = query.conversation_history + [{
            "question": query.question,
            "answer": formatted_response
        }]
        
        return Response(
            answer=formatted_response,
            sources=mock_sources,
            conversation_history=updated_history
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080) 