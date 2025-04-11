import os
import logging
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_document(file_path: str, vector_store: Chroma):
    """Process a single document and add it to the vector store."""
    try:
        logger.info(f"Processing document: {file_path}")
        
        # Load the document
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(pages)
        
        # Add chunks to the vector store
        vector_store.add_documents(chunks)
        logger.info(f"Added {len(chunks)} chunks from {file_path}")
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")

def main():
    # Initialize embeddings and vector store
    embeddings = HuggingFaceEmbeddings()
    vector_store = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    
    # Process all PDF files in the books directory
    books_dir = "books"
    if not os.path.exists(books_dir):
        logger.error(f"Books directory '{books_dir}' not found")
        return
    
    for filename in os.listdir(books_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(books_dir, filename)
            process_document(file_path, vector_store)
    
    # Persist the vector store
    vector_store.persist()
    logger.info("Vector store updated successfully")

if __name__ == "__main__":
    main() 