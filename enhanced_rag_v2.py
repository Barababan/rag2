import os
import logging
import re
from typing import List, Dict, Any, Union, Optional
from langchain_community.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from chromadb.config import Settings
from sentence_transformers import CrossEncoder
import numpy as np
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
import threading
import time
from dotenv import load_dotenv
import gc
import uuid
from datetime import datetime
import chromadb
import asyncio
from pydantic import Field, BaseModel

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
    
# Load environment variables
load_dotenv()

# Debug logging for OpenAI API key
api_key = os.getenv('OPENAI_API_KEY')
if api_key:
    print(f"OpenAI API key found: {api_key[:10]}...")
else:
    print("OpenAI API key not found in environment variables")

class EnhancedPhysicalTherapyRAG:
    def __init__(self):
        self.logger = self._setup_logger()
        self.embeddings = self._initialize_embeddings()
        self.vectorstore = self._initialize_vectorstore()
        self.bm25 = self._initialize_bm25()
        self.cross_encoder = self._initialize_cross_encoder()
        self.conversational_chain = self._initialize_conversational_chain()
        self.metadata_store = {}
        self.document_store = []
        self.chunk_size = 500
        self.chunk_overlap = 50
        self._memory_lock = threading.Lock()
        self._batch_size = 50  # Reduced from 100 to 50
        self._max_document_store_size = 1000  # Limit document store size
        
    def add_documents(self, documents: List[Document], source: str = None) -> None:
        """Add documents to both vector store and BM25 index with memory optimization."""
        try:
            with self._memory_lock:
                # Filter complex metadata
                for doc in documents:
                    doc.metadata = self._filter_complex_metadata(doc.metadata)
                
                # Add to vector store
                self.vectorstore.add_documents(documents)
                
                # Update BM25 with memory management
                if self.bm25 and documents:
                    # Limit document store size
                    if len(self.document_store) + len(documents) > self._max_document_store_size:
                        # Remove oldest documents
                        excess = len(self.document_store) + len(documents) - self._max_document_store_size
                        self.document_store = self.document_store[excess:]
                        self.logger.info(f"Trimmed document store to {len(self.document_store)} documents")
                    
                    # Add documents to document store
                    self.document_store.extend(documents)
                    
                    # Tokenize all documents
                    all_tokenized_docs = []
                    for doc in self.document_store:
                        try:
                            tokens = word_tokenize(doc.page_content.lower())
                            all_tokenized_docs.append(tokens)
                        except Exception as e:
                            self.logger.warning(f"Error tokenizing document: {str(e)}")
                            continue
                    
                    if all_tokenized_docs:
                        # Reinitialize BM25 with all documents
                        self.bm25 = BM25Okapi(all_tokenized_docs)
                
                # Force garbage collection
                gc.collect()
                
                # Log success
                self.logger.info(f"Added {len(documents)} documents to stores" + 
                               (f" from {source}" if source else ""))
                
        except Exception as e:
            self.logger.error(f"Error adding documents: {str(e)}")
            raise
            
    def _extract_metadata(self, doc: Document, source: str = None) -> Dict[str, Any]:
        """Extract enhanced metadata from document."""
        try:
            # Basic metadata
            metadata = {
                'id': str(uuid.uuid4()),
                'timestamp': datetime.now().isoformat(),
                'source': source or 'unknown',
                'length': len(doc.page_content),
                'type': 'pdf'
            }
            
            # Extract headers and sections
            headers = re.findall(r'^(?:Chapter|Section|Part)\s+\d+[.:]\s*([^\n]+)', doc.page_content, re.MULTILINE)
            if headers:
                metadata['headers'] = headers[:3]  # Store top 3 headers
            
            # Extract key terms with TF-IDF-like scoring
            try:
                key_terms = self._extract_key_terms(doc.page_content, max_terms=5)
                metadata['key_terms'] = key_terms
            except Exception as e:
                self.logger.warning(f"Error extracting key terms: {str(e)}")
                metadata['key_terms'] = []
            
            # Calculate quality metrics
            quality_score = self._calculate_quality_score(doc)
            metadata['quality_score'] = quality_score
            
            # Extract medical terminology
            medical_terms = re.findall(r'\b(patient|treatment|therapy|diagnosis|assessment|rehabilitation|exercise|pain|movement|function)\b', 
                                     doc.page_content, re.I)
            metadata['medical_terms'] = list(set(medical_terms))
            
            # Extract references
            references = re.findall(r'\[\d+\]|\(\d+\)', doc.page_content)
            metadata['reference_count'] = len(references)
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error extracting metadata: {str(e)}")
            return {
                'id': str(uuid.uuid4()),
                'timestamp': datetime.now().isoformat(),
                'source': source or 'unknown',
                'length': len(doc.page_content),
                'type': 'pdf',
                'quality_score': 0.0
            }
            
    def _calculate_quality_score(self, doc: Document) -> float:
        """Calculate enhanced document quality score."""
        try:
            score = 0.0
            
            # Content length score (0-0.2)
            content_length = len(doc.page_content)
            if content_length > 500:
                score += 0.2
            elif content_length > 200:
                score += 0.1
            
            # Reference score (0-0.2)
            references = re.findall(r'\[\d+\]|\(\d+\)', doc.page_content)
            if len(references) > 5:
                score += 0.2
            elif len(references) > 2:
                score += 0.1
            
            # Medical terminology score (0-0.3)
            medical_terms = re.findall(r'\b(patient|treatment|therapy|diagnosis|assessment|rehabilitation|exercise|pain|movement|function)\b', 
                                     doc.page_content, re.I)
            if len(set(medical_terms)) > 10:
                score += 0.3
            elif len(set(medical_terms)) > 5:
                score += 0.2
            elif len(set(medical_terms)) > 2:
                score += 0.1
            
            # Structure score (0-0.3)
            has_headers = bool(re.search(r'^(?:Chapter|Section|Part)\s+\d+[.:]\s*[^\n]+', doc.page_content, re.MULTILINE))
            has_lists = bool(re.search(r'^\s*[-•*]\s+', doc.page_content, re.MULTILINE))
            has_tables = bool(re.search(r'\|\s*[^|]+\s*\|', doc.page_content))
            
            if has_headers and has_lists:
                score += 0.3
            elif has_headers or has_lists:
                score += 0.2
            elif has_tables:
                score += 0.1
            
            return min(score, 1.0)  # Ensure score doesn't exceed 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating quality score: {str(e)}")
            return 0.0

    def _setup_logger(self):
        """Set up the logger for the RAG system."""
        logger = logging.getLogger("enhanced_rag")
        logger.setLevel(logging.INFO)
        
        # Create handlers
        if not logger.handlers:
            # File handler
            log_file = f'logs/rag_system_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            os.makedirs('logs', exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Create formatters and add it to handlers
            log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(log_format)
            console_handler.setFormatter(log_format)
            
            # Add handlers to the logger
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger

    def _initialize_embeddings(self):
        """Initialize the embeddings model."""
        try:
            embeddings = OpenAIEmbeddings()
            self.logger.info("Embeddings model initialized successfully")
            return embeddings
        except Exception as e:
            self.logger.error(f"Error initializing embeddings: {str(e)}")
            raise

    def _initialize_vectorstore(self):
        """Initialize the vector store with conservative memory settings."""
        try:
            self.logger.info("Initializing vector store...")
            persist_dir = os.path.join(os.getcwd(), "chroma_db")
            
            # More conservative ChromaDB settings
            settings = chromadb.config.Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True,
                persist_directory=persist_dir,
            )
            
            return Chroma(
                persist_directory=persist_dir,
                embedding_function=self.embeddings,
                client_settings=settings
            )
        except Exception as e:
            self.logger.error(f"Error initializing vector store: {str(e)}")
            raise

    def _initialize_bm25(self):
        """Initialize BM25 for keyword-based search."""
        try:
            # Initialize with a dummy document to avoid division by zero
            dummy_doc = ["dummy", "document", "for", "initialization"]
            tokenized_docs = [dummy_doc]
            bm25 = BM25Okapi(tokenized_docs)
            self.logger.info("BM25 initialized successfully with dummy document")
            return bm25
        except Exception as e:
            self.logger.error(f"Error initializing BM25: {str(e)}")
            raise

    def _initialize_cross_encoder(self):
        """Initialize cross-encoder for reranking."""
        try:
            cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            self.logger.info("Cross-encoder initialized successfully")
            return cross_encoder
        except Exception as e:
            self.logger.error(f"Error initializing cross-encoder: {str(e)}")
            raise

    def _initialize_conversational_chain(self):
        """Initialize the conversational chain."""
        try:
            # Create an enhanced prompt template
            prompt_template = """Answer the question based on this context:

{context}

Question: {question}

Please provide a detailed answer with specific information from the sources. If you're unsure about any part, please indicate that.

Answer:"""
            
            prompt = PromptTemplate.from_template(prompt_template)
            
            # Create a function to format documents with metadata
            def format_docs(docs):
                if not docs:
                    return "No relevant information found."
                
                formatted_docs = []
                for doc in docs:
                    # Extract metadata
                    source = doc.metadata.get('source', 'Unknown')
                    page = doc.metadata.get('page', '')
                    headers = doc.metadata.get('headers', [])
                    
                    # Format the document
                    formatted_doc = f"[Source: {source}"
                    if page:
                        formatted_doc += f", Page: {page}"
                    if headers:
                        formatted_doc += f", Sections: {', '.join(headers)}"
                    formatted_doc += "]\n\n"
                    
                    # Add content
                    formatted_doc += doc.page_content
                    formatted_docs.append(formatted_doc)
                
                return "\n\n---\n\n".join(formatted_docs)
            
            # Create the chain using ConversationalRetrievalChain
            chain = ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7),
                retriever=self.retriever,
                combine_docs_chain_kwargs={"prompt": prompt},
                return_source_documents=True
            )
            
            self.logger.info("Conversational chain initialized successfully")
            return chain
            
        except Exception as e:
            self.logger.error(f"Error initializing conversational chain: {str(e)}")
            raise

    def _extract_key_terms(self, text: str, max_terms: int = 5) -> List[str]:
        """Extract key terms from text using simple tokenization."""
        try:
            # Simple tokenization without relying on NLTK's punkt_tab
            words = text.lower().split()
            word_freq = {}
            
            # Count word frequencies
            for word in words:
                # Skip short words and non-alphanumeric
                if word.isalnum() and len(word) > 2:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Sort by frequency and get top terms
            sorted_terms = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [term for term, _ in sorted_terms[:max_terms]]
            
        except Exception as e:
            self.logger.error(f"Error extracting key terms: {str(e)}")
            return []

    def hybrid_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform hybrid search combining semantic and keyword-based approaches."""
        try:
            # Get semantic search results
            semantic_results = self.vectorstore.similarity_search(query, k=k)
            
            # Get BM25 results
            bm25_results = []
            if self.bm25 and self.document_store:
                try:
                    tokenized_query = word_tokenize(query.lower())
                    bm25_scores = self.bm25.get_scores(tokenized_query)
                    if len(bm25_scores) > 0:
                        top_indices = np.argsort(bm25_scores)[-k:][::-1]
                        # Ensure indices are within bounds
                        valid_indices = [i for i in top_indices if i < len(self.document_store)]
                        bm25_results = [self.document_store[i] for i in valid_indices]
                except Exception as e:
                    self.logger.warning(f"Error in BM25 search: {str(e)}")
            
            # Combine results using document IDs to avoid duplicates
            seen_ids = set()
            unique_results = []
            
            for doc in semantic_results + bm25_results:
                doc_id = doc.metadata.get('id', '')
                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    unique_results.append(doc)
            
            # Rerank using cross-encoder
            if self.cross_encoder and unique_results:
                try:
                    pairs = [(query, doc.page_content) for doc in unique_results]
                    scores = self.cross_encoder.predict(pairs)
                    scored_results = list(zip(unique_results, scores))
                    scored_results.sort(key=lambda x: x[1], reverse=True)
                    reranked_results = [doc for doc, _ in scored_results[:k]]
                except Exception as e:
                    self.logger.warning(f"Error in cross-encoder reranking: {str(e)}")
                    reranked_results = unique_results[:k]
            else:
                reranked_results = unique_results[:k]
            
            return reranked_results
            
        except Exception as e:
            self.logger.error(f"Error in hybrid search: {str(e)}")
            # Fallback to basic semantic search
            return self.vectorstore.similarity_search(query, k=k)

    @property
    def retriever(self) -> BaseRetriever:
        """Get the hybrid retriever."""
        class HybridRetriever(BaseRetriever, BaseModel):
            rag: Any = Field(description="The RAG instance")
            
            def __init__(self, rag_instance):
                super().__init__(rag=rag_instance)
            
            def _get_relevant_documents(self, query: str) -> List[Document]:
                return self.rag.hybrid_search(query)
            
            async def _aget_relevant_documents(self, query: str) -> List[Document]:
                return self._get_relevant_documents(query)
        
        return HybridRetriever(self)

    async def process_query(self, query: str, chat_history: List[tuple] = None) -> str:
        """Process a query using the enhanced RAG system."""
        try:
            # Convert chat history to the format expected by the chain
            formatted_history = []
            if chat_history:
                for human, ai in chat_history:
                    formatted_history.append((human, ai))
            
            # Process query with the chain using ainvoke instead of acall
            result = await self.conversational_chain.ainvoke({
                "question": query,
                "chat_history": formatted_history
            })
            
            # Extract answer and sources
            answer = result["answer"]
            sources = result.get("source_documents", [])
            
            # Format response with unique sources
            seen_sources = set()
            response = f"{answer}\n\nSources:"
            for doc in sources:
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "")
                source_key = f"{source}_{page}"
                
                if source_key not in seen_sources:
                    seen_sources.add(source_key)
                    response += f"\n- {source}"
                    if page:
                        response += f" (Page {page})"
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return "I apologize, but I encountered an error while processing your query. Please try again."

    def get_document_count(self) -> int:
        """Get the total number of documents in the system."""
        try:
            return len(self.document_store)
        except Exception as e:
            self.logger.error(f"Error getting document count: {str(e)}")
            return 0

    async def process_document(self, text: str, metadata: dict) -> None:
        """Process a single document with enhanced metadata and quality scoring."""
        try:
            # Create document with basic metadata
            doc = Document(page_content=text, metadata=metadata)
            
            # Extract enhanced metadata
            enhanced_metadata = self._extract_metadata(doc, source=metadata.get('source'))
            
            # Create new document with enhanced metadata
            enhanced_doc = Document(
                page_content=text,
                metadata=enhanced_metadata
            )
            
            # Process in smaller chunks with optimized settings for large documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=250,  # Increased from 100 to 250 for better context
                chunk_overlap=20,  # Increased from 10 to 20 for better continuity
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]  # Added ". " for better sentence splitting
            )
            
            # Split text into chunks
            chunks = text_splitter.split_text(text)
            total_chunks = len(chunks)
            self.logger.info(f"Split document into {total_chunks} chunks")
            
            # Process chunks in optimized batches
            batch_size = 3  # Process 3 chunks at a time for better efficiency
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                # Create documents for batch
                batch_docs = []
                for j, chunk in enumerate(batch):
                    chunk_metadata = enhanced_metadata.copy()
                    chunk_metadata['chunk_index'] = i + j
                    chunk_metadata['total_chunks'] = total_chunks
                    batch_docs.append(Document(page_content=chunk, metadata=chunk_metadata))
                
                # Add documents to stores
                self.add_documents(batch_docs, source=metadata.get('source'))
                
                # Clean up
                del batch_docs
                gc.collect()
                gc.collect()
                
                # Add shorter delay between batches
                await asyncio.sleep(1)  # Reduced from 2 to 1 second
                
                # Log progress
                if (i + batch_size) % 10 == 0 or (i + batch_size) >= len(chunks):
                    self.logger.info(f"Processed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")
            
            self.logger.info(f"Successfully processed document from {metadata.get('source', 'unknown source')}")
            
        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            raise

    def _filter_complex_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Filter complex metadata to only include simple types."""
        filtered = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                filtered[key] = value
            elif isinstance(value, list):
                # Convert lists to comma-separated strings
                filtered[key] = ', '.join(map(str, value))
        return filtered

    async def process_document_batch(self, documents: List[Document]) -> None:
        """Process a batch of documents asynchronously."""
        try:
            self.logger.info(f"Starting to process batch of {len(documents)} documents")
            
            # Add to vector store with progress tracking
            self.logger.info("Adding documents to vector store...")
            try:
                self.vectorstore.add_documents(documents)
                self.logger.info("Successfully added documents to vector store")
            except Exception as vs_error:
                self.logger.error(f"Error adding to vector store: {str(vs_error)}")
                raise
            
            # Update BM25 with memory management
            if self.bm25 and documents:
                self.logger.info("Updating BM25 index...")
                try:
                    with self._memory_lock:
                        # Limit document store size
                        if len(self.document_store) + len(documents) > self._max_document_store_size:
                            excess = len(self.document_store) + len(documents) - self._max_document_store_size
                            self.document_store = self.document_store[excess:]
                            self.logger.info(f"Trimmed document store to {len(self.document_store)} documents")
                        
                        # Add documents to document store
                        self.document_store.extend(documents)
                        self.logger.info(f"Added {len(documents)} documents to document store")
                        
                        # Tokenize all documents
                        self.logger.info("Tokenizing documents for BM25...")
                        all_tokenized_docs = []
                        for doc in self.document_store:
                            try:
                                tokens = word_tokenize(doc.page_content.lower())
                                all_tokenized_docs.append(tokens)
                            except Exception as e:
                                self.logger.warning(f"Error tokenizing document: {str(e)}")
                                continue
                        
                        if all_tokenized_docs:
                            # Reinitialize BM25 with all documents
                            self.bm25 = BM25Okapi(all_tokenized_docs)
                            self.logger.info(f"BM25 index updated with {len(all_tokenized_docs)} documents")
                except Exception as bm25_error:
                    self.logger.error(f"Error updating BM25: {str(bm25_error)}")
                    raise
            
            # Force garbage collection
            gc.collect()
            self.logger.info("Batch processing completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in process_document_batch: {str(e)}")
            raise 