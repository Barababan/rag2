from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def test_openai_connection():
    try:
        # Print API key length for verification (without revealing the key)
        api_key = os.getenv('OPENAI_API_KEY')
        logger.info(f"API key length: {len(api_key) if api_key else 'No key found'}")
        
        # Initialize embeddings
        embeddings = OpenAIEmbeddings()
        
        # Try to get embeddings for a simple text
        test_text = "This is a test."
        result = embeddings.embed_query(test_text)
        
        # If we get here, it worked
        logger.info("✅ OpenAI API connection successful!")
        logger.info(f"Received embedding vector of length: {len(result)}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing OpenAI API: {str(e)}")
        return False

if __name__ == "__main__":
    test_openai_connection() 