from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Initialize conversation history
conversation_history = []

class Query(BaseModel):
    question: str = Field(..., description="The question to be answered")
    conversation_history: Optional[List[dict]] = Field(default_factory=list, description="Previous conversation history")

class Response(BaseModel):
    answer: str = Field(..., description="The answer to the question")
    sources: List[str] = Field(default_factory=list, description="Sources used to generate the answer")
    conversation_history: List[dict] = Field(default_factory=list, description="Updated conversation history")

def enhance_query(query: str, conversation_history: List[dict]) -> str:
    """Enhance the query with context from conversation history."""
    if not conversation_history:
        return query
    
    # Get the last few exchanges for context
    recent_history = conversation_history[-3:]  # Last 3 exchanges
    context = "\n".join([
        f"Q: {exchange['question']}\nA: {exchange['answer']}"
        for exchange in recent_history
    ])
    
    return f"""Previous conversation:
{context}

Current question: {query}

Please consider the context from the previous conversation when answering."""

def format_response_with_sources(response: str, sources: List[str]) -> str:
    """Format the response to be more conversational and include sources."""
    if not sources:
        return response
    
    source_text = "\n\nSources:\n" + "\n".join(f"- {source}" for source in sources)
    return f"{response}{source_text}"

@app.post("/query", response_model=Response)
async def query_knowledge_base(query: Query):
    try:
        # Enhance query with conversation history
        enhanced_query = enhance_query(query.question, query.conversation_history)
        
        # Initialize the language model
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create a prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Answer questions based on the provided context and previous conversation history. Be conversational and friendly."),
            ("human", "{query}")
        ])
        
        # Create the chain
        chain = prompt | llm | StrOutputParser()
        
        # Get the response
        response = chain.invoke({"query": enhanced_query})
        
        # Update conversation history
        conversation_history.append({
            "question": query.question,
            "answer": response
        })
        
        # For now, return empty sources as we haven't implemented source tracking
        return Response(
            answer=response,
            sources=[],
            conversation_history=conversation_history
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080) 