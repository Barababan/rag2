from enhanced_rag_v2 import EnhancedPhysicalTherapyRAG
import logging
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import chromadb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def test_rag_query():
    try:
        # Check if vector store exists
        persist_dir = os.path.join(os.getcwd(), "chroma_db")
        if not os.path.exists(persist_dir):
            logger.error(f"Vector store directory not found at {persist_dir}")
            return False
            
        # Initialize components directly
        embeddings = OpenAIEmbeddings()
        
        # Initialize ChromaDB with settings
        settings = chromadb.config.Settings(
            anonymized_telemetry=False,
            allow_reset=True,
            is_persistent=True,
            persist_directory=persist_dir
        )
        
        # Create vector store
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            client_settings=settings
        )
        
        # Get collection info
        collection_count = len(vectorstore._client.list_collections())
        logger.info(f"Number of collections: {collection_count}")
        
        # Sample query about pediatrics
        query = "What are the common symptoms of fever in children and how should parents manage it?"
        logger.info(f"Querying: {query}")
        
        # Get relevant documents
        results = vectorstore.similarity_search(query, k=3)
        
        if not results:
            logger.warning("No results found. This might indicate that the vector store is empty.")
            return False
            
        # Print results
        logger.info("\nRelevant passages found:")
        for i, doc in enumerate(results, 1):
            logger.info(f"\n--- Result {i} ---")
            logger.info(f"Source: {doc.metadata.get('source', 'Unknown')}")
            logger.info(f"Page: {doc.metadata.get('page', 'Unknown')}")
            logger.info(f"Content: {doc.page_content[:500]}...")
            
        return True
        
    except Exception as e:
        logger.error(f"Error testing RAG query: {str(e)}")
        return False

if __name__ == "__main__":
    test_rag_query() 