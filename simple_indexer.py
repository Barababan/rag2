import os
import logging
import pickle
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
import warnings

# Load environment variables
load_dotenv()

# Suppress specific pypdf warnings
warnings.filterwarnings("ignore", message=".*PageLabels.*")

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"indexer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class SimpleDocumentStore:
    """A simple document store that saves and loads documents."""
    
    def __init__(self):
        self.documents = []
        
    def add_document(self, document):
        """Add a document to the store."""
        self.documents.append(document)
        
    def save(self, path):
        """Save documents to a file."""
        with open(path, 'wb') as f:
            pickle.dump(self.documents, f)
            
    def load(self, path):
        """Load documents from a file."""
        with open(path, 'rb') as f:
            self.documents = pickle.load(f)
            
    def get_documents(self):
        """Get all documents."""
        return self.documents

def process_book(pdf_path: str, output_dir: str = "index_data") -> bool:
    """Process a book PDF and create embeddings."""
    try:
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Initialize embeddings
        logger.info("Initializing embeddings...")
        embeddings = OpenAIEmbeddings()
        
        # Load and split the document
        logger.info(f"Processing {pdf_path}...")
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        if not pages:
            logger.error(f"No pages found in {pdf_path}")
            return False
            
        logger.info(f"Found {len(pages)} pages")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(pages)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Create vector store
        book_name = os.path.splitext(os.path.basename(pdf_path))[0]
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=os.path.join(output_dir, book_name)
        )
        
        # Save the vector store
        vector_store.persist()
        logger.info(f"Saved vector store for {book_name}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {str(e)}")
        return False

def main():
    """Main function to process books."""
    # Get books directory
    books_dir = Path("books")
    if not books_dir.exists():
        logger.error("Books directory not found")
        return
    
    # Process each PDF in the books directory
    for book_file in books_dir.glob("*.pdf"):
        logger.info(f"Found book: {book_file.name}")
        success = process_book(str(book_file))
        if success:
            logger.info(f"Successfully processed {book_file.name}")
        else:
            logger.error(f"Failed to process {book_file.name}")

if __name__ == "__main__":
    main() 