from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

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

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings()
vector_store = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# Initialize language model
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
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

# Root endpoint that serves the chat interface
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Chat endpoint
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
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({})

# Query the knowledge base
@app.post("/query", response_model=Response)
async def query_knowledge_base(query: Query):
    try:
        # Enhance the query with conversation history
        enhanced_query = enhance_query(query.question, query.conversation_history)
        
        # Search for relevant documents
        docs = vector_store.similarity_search(enhanced_query, k=3)
        
        # Create a prompt with the retrieved documents
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful medical assistant. Answer the question based on the provided context.
            If you don't know the answer, say you don't know. Don't make up information.
            Be conversational and friendly in your response."""),
            ("human", """Context: {context}
            
            Question: {question}""")
        ])
        
        # Create the chain
        chain = (
            {"context": lambda x: "\n\n".join([doc.page_content for doc in docs]),
             "question": lambda x: x["question"]}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Get the response
        response = chain.invoke({"question": enhanced_query})
        
        # Get source documents
        sources = [doc.metadata.get("source", "Unknown source") for doc in docs]
        
        # Format the response
        formatted_response = format_response_with_sources(response, sources)
        
        # Update conversation history
        updated_history = query.conversation_history + [{
            "question": query.question,
            "answer": formatted_response
        }]
        
        return Response(
            answer=formatted_response,
            sources=sources,
            conversation_history=updated_history
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080) 