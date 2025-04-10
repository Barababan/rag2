from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize the RAG system
def init_rag_system():
    vector_stores = {}
    bm25_retrievers = {}
    index_dir = "index_data"
    
    if not os.path.exists(index_dir):
        return vector_stores, bm25_retrievers
        
    for book_dir in os.listdir(index_dir):
        book_path = os.path.join(index_dir, book_dir)
        if os.path.isdir(book_path):
            try:
                # Initialize vector store
                vector_store = Chroma(
                    persist_directory=book_path,
                    embedding_function=OpenAIEmbeddings()
                )
                vector_stores[book_dir] = vector_store
                
                # Initialize BM25 retriever
                documents = vector_store.get()
                bm25_retriever = BM25Retriever.from_documents(documents)
                bm25_retrievers[book_dir] = bm25_retriever
                
            except Exception as e:
                print(f"Error loading vector store for {book_dir}: {str(e)}")
                
    return vector_stores, bm25_retrievers

# Initialize the system
vector_stores, bm25_retrievers = init_rag_system()

# Create hybrid retrievers
hybrid_retrievers = {}
for book_name, vector_store in vector_stores.items():
    # Get vector store retriever
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Get BM25 retriever
    bm25_retriever = bm25_retrievers.get(book_name)
    
    # Create hybrid retriever
    if bm25_retriever:
        hybrid_retriever = ContextualCompressionRetriever(
            base_compressor=LLMChainExtractor.from_llm(ChatOpenAI(temperature=0)),
            base_retriever=vector_retriever
        )
        hybrid_retrievers[book_name] = hybrid_retriever
    else:
        hybrid_retrievers[book_name] = vector_retriever

# Create QA chains
qa_chains = {
    book_name: ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0),
        retriever=retriever,
        return_source_documents=True
    )
    for book_name, retriever in hybrid_retrievers.items()
}

def format_source_document(doc, max_length: int = 200) -> dict:
    """Format a source document with context."""
    source = doc.metadata.get('source', 'Unknown source')
    page = doc.metadata.get('page', 'Unknown page')
    content = doc.page_content.strip()
    if len(content) > max_length:
        content = content[:max_length] + "..."
    
    return {
        "source": source,
        "page": page,
        "context": content
    }

@app.route('/')
def home():
    return render_template('index.html', books=list(qa_chains.keys()))

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    book_name = data.get('book')
    question = data.get('question')
    
    if not book_name or not question:
        return jsonify({"error": "Missing book name or question"}), 400
        
    if book_name not in qa_chains:
        return jsonify({"error": "Book not found"}), 404
        
    try:
        result = qa_chains[book_name]({"question": question, "chat_history": []})
        
        sources = [
            format_source_document(doc)
            for doc in result["source_documents"]
        ]
        
        return jsonify({
            "answer": result["answer"],
            "sources": sources
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port) 