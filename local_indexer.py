import os
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from enhanced_rag_v2 import EnhancedRAGSystem

# Load environment variables
load_dotenv()

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

def process_book(book_path: str, output_dir: str = "index_data"):
    """Process a book and save the index data."""
    try:
        # Initialize RAG system
        logger.info("Initializing RAG system...")
        rag_system = EnhancedRAGSystem()
        
        # Process the book
        logger.info(f"Processing book: {book_path}")
        rag_system.process_document(book_path)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save vector store
        vector_store_path = output_path / "vector_store"
        rag_system.vectorstore.save_local(str(vector_store_path))
        logger.info(f"Vector store saved to {vector_store_path}")
        
        # Save BM25 index
        bm25_path = output_path / "bm25_index.pkl"
        rag_system.bm25.save_index(str(bm25_path))
        logger.info(f"BM25 index saved to {bm25_path}")
        
        # Save document store
        doc_store_path = output_path / "document_store.pkl"
        rag_system.document_store.save(str(doc_store_path))
        logger.info(f"Document store saved to {doc_store_path}")
        
        logger.info("Book processing completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error processing book: {str(e)}")
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