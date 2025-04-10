import os
import logging
from typing import List, Dict
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_vector_stores(index_dir: str = "index_data") -> Dict[str, Chroma]:
    """Load all vector stores from the index directory."""
    vector_stores = {}
    if not os.path.exists(index_dir):
        logging.error(f"Index directory {index_dir} not found")
        return vector_stores
        
    for book_dir in os.listdir(index_dir):
        book_path = os.path.join(index_dir, book_dir)
        if os.path.isdir(book_path):
            try:
                vector_store = Chroma(
                    persist_directory=book_path,
                    embedding_function=OpenAIEmbeddings()
                )
                vector_stores[book_dir] = vector_store
                logging.info(f"Loaded vector store for {book_dir}")
            except Exception as e:
                logging.error(f"Error loading vector store for {book_dir}: {str(e)}")
                
    return vector_stores

def create_qa_chain(vector_store: Chroma):
    """Create a question-answering chain for a vector store."""
    llm = ChatOpenAI(temperature=0)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

def format_source_document(doc, max_length: int = 200) -> str:
    """Format a source document with context."""
    # Get the source and page information
    source = doc.metadata.get('source', 'Unknown source')
    page = doc.metadata.get('page', 'Unknown page')
    
    # Get the content and truncate if too long
    content = doc.page_content.strip()
    if len(content) > max_length:
        content = content[:max_length] + "..."
    
    # Format the output
    return f"""
Source: {source}
Page: {page}
Context:
{content}
"""

def main():
    # Load vector stores
    vector_stores = load_vector_stores()
    if not vector_stores:
        logging.error("No vector stores found. Please run the indexer first.")
        return
        
    # Create QA chains for each book
    qa_chains = {
        book_name: create_qa_chain(vector_store)
        for book_name, vector_store in vector_stores.items()
    }
    
    # Interactive query loop
    print("\nWelcome to the Medical Knowledge Base Query System!")
    print("Available books:")
    for i, book_name in enumerate(qa_chains.keys(), 1):
        print(f"{i}. {book_name}")
    
    while True:
        try:
            # Get book selection
            book_idx = int(input("\nSelect a book number (or 0 to exit): ")) - 1
            if book_idx == -1:
                break
                
            book_name = list(qa_chains.keys())[book_idx]
            qa_chain = qa_chains[book_name]
            
            # Get query
            query = input("\nEnter your question (or 'back' to select another book): ")
            if query.lower() == 'back':
                continue
                
            # Get answer
            print("\nSearching...")
            result = qa_chain({"question": query, "chat_history": []})
            
            # Display answer
            print("\nAnswer:", result["answer"])
            print("\nSources:")
            for i, doc in enumerate(result["source_documents"], 1):
                print(f"\nSource {i}:")
                print(format_source_document(doc))
                
        except (ValueError, IndexError):
            print("Invalid selection. Please try again.")
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            print("An error occurred. Please try again.")

if __name__ == "__main__":
    main() 