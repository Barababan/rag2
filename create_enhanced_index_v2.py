import os
import logging
import asyncio
import gc
from datetime import datetime
from typing import List, Dict, Any, Tuple
from enhanced_rag_v2 import EnhancedPhysicalTherapyRAG
from dotenv import load_dotenv
from langchain.schema import Document
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from io import StringIO
import concurrent.futures
from functools import partial
from itertools import chain
import tqdm

# Load environment variables
load_dotenv()

# Configure logging
log_file = f'logs/rag_system_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
os.makedirs('logs', exist_ok=True)

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create and configure file handler
file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
file_handler.setFormatter(formatter)

# Create and configure console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Configure root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Remove any existing handlers to avoid duplicates
for handler in logger.handlers[:]:
    if isinstance(handler, (logging.FileHandler, logging.StreamHandler)):
        logger.removeHandler(handler)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def extract_page_text(pdf_path: str, page_nums: List[int]) -> List[Tuple[int, str]]:
    """Extract text from multiple pages at once."""
    try:
        results = []
        for page_num in page_nums:
            text = extract_text(pdf_path, page_numbers=[page_num], codec='utf-8', maxpages=1)
            results.append((page_num, text))
        return results
    except Exception as e:
        logger.error(f"Error extracting text from pages {page_nums}: {str(e)}")
        return [(num, "") for num in page_nums]

async def process_text_batch(texts: List[Tuple[int, str]], total_pages: int, rag_system: EnhancedPhysicalTherapyRAG, source: str) -> None:
    """Process a batch of texts with combined embeddings."""
    try:
        # Filter out empty texts
        valid_texts = [(page_num, text) for page_num, text in texts if text.strip()]
        
        logger.info(f"Processing batch with {len(valid_texts)} valid texts out of {len(texts)} total")
        
        if not valid_texts:
            logger.warning("No valid texts found in batch")
            return
        
        # Process documents in larger batches for better throughput
        batch_size = 100  # Increased from 50 to 100
        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i:i + batch_size]
            logger.info(f"Processing sub-batch {i//batch_size + 1} of {(len(valid_texts) + batch_size - 1)//batch_size}")
            
            # Create documents with metadata
            documents = []
            for page_num, text in batch:
                if not text.strip():
                    logger.warning(f"Empty text found for page {page_num + 1}")
                    continue
                    
                logger.info(f"Creating document for page {page_num + 1} with {len(text)} characters")
                metadata = {
                    "source": source,
                    "page": page_num + 1,
                    "total_pages": total_pages,
                    "processed_at": datetime.now().isoformat(),
                    "batch_size": len(valid_texts),
                    "text_length": len(text)
                }
                documents.append(Document(page_content=text, metadata=metadata))
            
            # Process batch
            logger.info(f"Sending {len(documents)} documents for processing")
            try:
                await rag_system.process_document_batch(documents)
                logger.info(f"Successfully processed sub-batch with {len(documents)} documents")
            except Exception as batch_error:
                logger.error(f"Error processing sub-batch: {str(batch_error)}")
                logger.error(f"First document preview: {documents[0].page_content[:100]}..." if documents else "No documents")
                raise
            
            # Force garbage collection after batch
            gc.collect()
            logger.info("Garbage collection completed")
            
            # Reduced delay between batches
            await asyncio.sleep(0.1)  # Reduced from 0.25 to 0.1 seconds
            
    except Exception as e:
        logger.error(f"Error processing text batch: {str(e)}")
        raise

async def process_pdf_in_batches(pdf_path: str, rag_system: EnhancedPhysicalTherapyRAG) -> Dict:
    try:
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Get total page count
        with open(pdf_path, 'rb') as file:
            parser = PDFParser(file)
            doc = PDFDocument(parser)
            total_pages = len(list(PDFPage.create_pages(doc)))
        
        logger.info(f"Found {total_pages} pages in {pdf_path}")
        
        # Process pages in smaller batches with reduced concurrency
        batch_size = 50  # Reduced from 200 to 50 for better memory management
        max_concurrent_batches = 10  # Reduced from 30 to 10 to prevent overload
        
        # Create batches of page numbers
        all_batches = [list(range(i, min(i + batch_size, total_pages))) 
                      for i in range(0, total_pages, batch_size)]
        
        logger.info(f"Created {len(all_batches)} batches of size {batch_size}")
        
        # Create progress bar
        pbar = tqdm.tqdm(total=total_pages, desc="Processing pages")
        
        # Process batches with concurrent executors
        for i in range(0, len(all_batches), max_concurrent_batches):
            current_batches = all_batches[i:i + max_concurrent_batches]
            batch_num = i//max_concurrent_batches + 1
            total_batch_groups = (len(all_batches) + max_concurrent_batches - 1)//max_concurrent_batches
            logger.info(f"Starting batch group {batch_num} of {total_batch_groups}")
            
            # Extract text from pages concurrently
            logger.info(f"Extracting text from pages in batch group {batch_num}")
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_batches) as executor:
                extract_func = partial(extract_page_text, pdf_path)
                text_results = list(executor.map(extract_func, current_batches))
            
            # Flatten results while preserving page numbers
            all_texts = list(chain.from_iterable(text_results))
            logger.info(f"Extracted text from {len(all_texts)} pages in batch group {batch_num}")
            
            # Process texts concurrently with smaller batch size
            tasks = []
            logger.info(f"Creating processing tasks for batch group {batch_num}")
            for j in range(0, len(all_texts), batch_size):
                batch_texts = all_texts[j:j + batch_size]
                logger.info(f"Creating task for sub-batch {j//batch_size + 1} in batch group {batch_num}")
                task = asyncio.create_task(
                    process_text_batch(
                        batch_texts,
                        total_pages,
                        rag_system,
                        os.path.basename(pdf_path)
                    )
                )
                tasks.append(task)
            
            # Wait for all text processing tasks to complete
            logger.info(f"Waiting for {len(tasks)} tasks in batch group {batch_num} to complete")
            try:
                await asyncio.gather(*tasks)
                logger.info(f"Completed all tasks in batch group {batch_num}")
            except Exception as e:
                logger.error(f"Error processing tasks in batch group {batch_num}: {str(e)}")
                raise
            
            # Update progress bar
            pages_processed = min((i + max_concurrent_batches) * batch_size, total_pages)
            pbar.update(pages_processed - pbar.n)
            logger.info(f"Processed {pages_processed} of {total_pages} pages")
            
            # Increased delay between batches for better resource management
            logger.info(f"Sleeping between batch groups")
            await asyncio.sleep(0.5)  # Increased from 0.25 to 0.5 seconds
        
        pbar.close()
        logger.info("Completed processing all pages")
        
        return {
            "success": True,
            "file": pdf_path,
            "pages_processed": total_pages
        }
        
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
        return {
            "success": False,
            "file": pdf_path,
            "error": str(e)
        }

async def process_large_document() -> Dict:
    try:
        rag_system = EnhancedPhysicalTherapyRAG()
        results = []
        
        # Process Nelson's Essentials of Pediatrics
        nelson_path = os.path.join(os.getcwd(), "books", "Nelson-essentials-of-pediatrics.pdf")
        if os.path.exists(nelson_path):
            logger.info(f"Starting to process Nelson's Essentials of Pediatrics: {nelson_path}")
            nelson_result = await process_pdf_in_batches(nelson_path, rag_system)
            results.append(nelson_result)
        else:
            logger.error(f"Book not found: {nelson_path}")
            results.append({
                "success": False,
                "file": nelson_path,
                "error": "File not found"
            })
        
        return {
            "success": all(r.get("success", False) for r in results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in process_large_document: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

async def main():
    logger.info("\n=== Starting Enhanced Document Processing (Large Document) ===")
    result = await process_large_document()
    
    # Log results summary
    if result.get("success"):
        logger.info(f"\nProcessing complete. Successfully processed: {result.get('file', 'Unknown file')} ({result.get('pages_processed', 0)} pages)")
    else:
        error_msg = result.get('error', 'Unknown error')
        file_name = result.get('file', 'Unknown file')
        logger.error(f"\nProcessing failed: {file_name} - {error_msg}")

if __name__ == "__main__":
    asyncio.run(main()) 