"""
Enterprise RAG Challenge - Optimized Implementation
This script processes PDF annual reports and answers questions with an efficient RAG pipeline.
By Krasina15. Made with crooked kind hands.
"""
import os
import gc
import json
import argparse
import logging
import time
import sys
from pathlib import Path
from typing import List, Dict, Any

# Import our optimized modules
from pdf_processor import PDFProcessor
from chunking_strategy import ChunkingStrategy
from vector_index import VectorIndexBuilder
from retriever import RAGRetriever
from answer_generator import AnswerGenerator
from pdf_tracking import PDFTracker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnterpriseRAGChallenge:
    """Main class integrating all components for the RAG challenge"""
    
    def __init__(self, 
                 pdf_dir: str, 
                 extract_tables: bool = True,
                 embedding_model: str = "text-embedding-3-small",
                 llm_model: str = "gpt-4o-mini",
                 chunk_size: int = 800,
                 chunk_overlap: int = 100,
                 top_k: int = 5,
                 batch_size: int = 10,
                 max_workers: int = 2,
                 vector_store_path: str = None,
                 skip_processed: bool = True):
        """Initialize the RAG challenge system"""
        self.pdf_dir = pdf_dir
        self.extract_tables = extract_tables
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.vector_store_path = vector_store_path
        self.skip_processed = skip_processed
        
        # Initialize components
        self.pdf_processor = PDFProcessor(extract_tables=extract_tables, batch_size=batch_size)
        self.chunking_strategy = ChunkingStrategy(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.vector_builder = VectorIndexBuilder(embedding_model=embedding_model, batch_size=batch_size)
        
        # Initialize PDF tracker if vector store path is provided
        if vector_store_path:
            tracking_file = os.path.join(os.path.dirname(vector_store_path), "processed_pdfs.json")
            self.pdf_tracker = PDFTracker(tracking_file)
        else:
            self.pdf_tracker = None
            
        # These will be initialized later
        self.pdf_data = None
        self.all_chunks = None
        self.vector_store = None
        self.retriever = None
        self.generator = None
    
    def process_pdfs(self):
        """Process all PDFs in the directory"""
        logger.info(f"Processing PDFs from {self.pdf_dir}")
        
        # Check if directory exists
        if not os.path.exists(self.pdf_dir):
            raise FileNotFoundError(f"PDF directory not found: {self.pdf_dir}")
        
        try:
            # Get all PDF files in the directory
            pdf_files = [f for f in os.listdir(self.pdf_dir) if f.lower().endswith('.pdf')]
            
            # If using tracking, filter out already processed PDFs
            if self.skip_processed and self.pdf_tracker:
                # Get the calculate_sha1 function from PDFProcessor
                calculate_sha1_fn = lambda filename: self.pdf_processor.calculate_sha1(
                    os.path.join(self.pdf_dir, filename))
                
                # Filter PDFs that need processing
                unprocessed_files = self.pdf_tracker.get_unprocessed_files(pdf_files, calculate_sha1_fn)
                
                if len(unprocessed_files) < len(pdf_files):
                    logger.info(f"Skipping {len(pdf_files) - len(unprocessed_files)} already processed PDFs")
                
                if not unprocessed_files:
                    logger.info("All PDFs already processed. Skipping processing step.")
                    self.pdf_data = {}
                    return
                
                # Process only unprocessed PDFs
                self.pdf_data = {}
                for pdf_file in unprocessed_files:
                    file_path = os.path.join(self.pdf_dir, pdf_file)
                    logger.info(f"Processing {pdf_file}")
                    sha1, sections = self.pdf_processor.process_pdf(file_path)
                    self.pdf_data[pdf_file] = {
                        "sha1": sha1,
                        "sections": sections,
                        "page_count": len(set(section.page for section in sections))
                    }
                    logger.info(f"Processed {pdf_file}: {len(sections)} sections, SHA1: {sha1}")
            else:
                # Process all PDFs
                self.pdf_data = self.pdf_processor.process_directory(
                    self.pdf_dir, 
                    max_workers=self.max_workers
                )
                
            logger.info(f"Processed {len(self.pdf_data)} PDF files")
        except Exception as e:
            logger.error(f"Error processing PDFs: {e}", exc_info=True)
            raise
    
    def create_chunks(self):
        """Create chunks from processed PDFs"""
        if not self.pdf_data and not (self.pdf_tracker and self.skip_processed):
            raise ValueError("PDF data not available. Run process_pdfs() first.")
        
        logger.info("Creating chunks from processed documents")
        self.all_chunks = []
        
        try:
            # Process new PDFs and create chunks
            for filename, data in self.pdf_data.items():
                chunks = self.chunking_strategy.create_chunks(
                    data["sections"], 
                    filename, 
                    data["sha1"]
                )
                self.all_chunks.extend(chunks)
                
                # Mark PDF as processed if we're tracking
                if self.pdf_tracker:
                    self.pdf_tracker.mark_pdf_as_processed(filename, data["sha1"], len(chunks))
                
                # Clear section data to save memory
                data["sections"] = None
                
                logger.info(f"Created {len(chunks)} chunks from {filename}")
            
            # Force garbage collection
            gc.collect()
            
            logger.info(f"Created {len(self.all_chunks)} new chunks")
        except Exception as e:
            logger.error(f"Error creating chunks: {e}", exc_info=True)
            raise
    
    def build_vector_index(self):
        """Build vector index from chunks"""
        # Allow building even if no new chunks when using existing vector store
        if not self.all_chunks and not (self.vector_store_path and os.path.exists(self.vector_store_path)):
            raise ValueError("Document chunks not available. Run create_chunks() first.")
        
        if self.all_chunks:
            logger.info(f"Building vector index with {len(self.all_chunks)} chunks")
        
        try:
            # Check if vector store exists and should be loaded
            if self.vector_store_path and os.path.exists(self.vector_store_path):
                logger.info(f"Loading existing vector store from {self.vector_store_path}")
                
                # Use the vector builder's load method for consistency
                self.vector_store = self.vector_builder.load_index(self.vector_store_path)
                
                if self.vector_store:
                    logger.info("Vector store loaded successfully")
                    
                    # Add new chunks if any
                    if self.all_chunks:
                        logger.info(f"Adding {len(self.all_chunks)} new chunks to existing vector store")
                        self.vector_store.add_documents(self.all_chunks)
                        
                        # Save updated vector store
                        if self.vector_store_path:
                            logger.info(f"Saving updated vector store to {self.vector_store_path}")
                            self.vector_store.save_local(self.vector_store_path)
                else:
                    logger.warning("Failed to load vector store. Building a new one.")
                    if self.all_chunks:
                        self.vector_store = self.vector_builder.build_index(
                            self.all_chunks, 
                            save_path=self.vector_store_path
                        )
                    else:
                        raise ValueError("No chunks available to build vector store")
            else:
                # Build new vector store from scratch
                if not self.all_chunks:
                    raise ValueError("No chunks available to build vector store")
                    
                self.vector_store = self.vector_builder.build_index(
                    self.all_chunks, 
                    save_path=self.vector_store_path
                )
            
            # Initialize retriever
            self.retriever = RAGRetriever(self.vector_store, top_k=self.top_k)
            
            # Free memory from chunks if no longer needed
            if self.all_chunks and self.vector_store_path:  # We've saved, so we can free memory
                self.all_chunks = None
                gc.collect()
                
        except Exception as e:
            logger.error(f"Error building vector index: {e}", exc_info=True)
            raise
    
    def answer_questions(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Answer a list of questions"""
        if not self.vector_store:
            raise ValueError("Vector store not available. Run build_vector_index() first.")
        
        # Initialize answer generator if not done yet
        if not self.generator:
            self.generator = AnswerGenerator(model_name=self.llm_model)
        
        logger.info(f"Answering {len(questions)} questions")
        answers = []
        
        # Process questions in batches to manage memory
        for i, question_obj in enumerate(questions):
            try:
                question_text = question_obj["text"]
                question_kind = question_obj["kind"]
                
                logger.info(f"Processing question {i+1}/{len(questions)}: {question_text}")
                
                # 1. Retrieve relevant context
                contexts = self.retriever.retrieve(question_text, question_kind)
                
                # 2. Generate answer
                answer = self.generator.answer_question(question_text, contexts, question_kind)
                
                # 3. Add to results
                answers.append(answer)
                
                # Log progress
                logger.info(f"Answer: {answer['value']}")
                logger.info(f"References: {len(answer['references'])}")
                
                # Cleanup after each question
                if i % 5 == 0:
                    gc.collect()
            except Exception as e:
                logger.error(f"Error answering question {i+1}: {e}", exc_info=True)
                # Add a fallback answer
                answers.append({
                    "value": "N/A" if question_kind != "names" else [],
                    "references": [],
                    "question_text": question_text,
                    "kind": question_kind
                })
        
        return answers
    
    def format_submission(self, answers: List[Dict[str, Any]], team_email: str, submission_name: str) -> Dict[str, Any]:
        """Format answers for submission"""
        formatted_answers = []
        
        for answer in answers:
            submission_answer = {
                "question_text": answer.get("question_text", ""),
                "kind": answer.get("kind", ""),
                "value": answer["value"],
                "references": [
                    {
                        "pdf_sha1": ref["pdf_sha1"],
                        "page_index": ref["page_index"]
                    } 
                    for ref in answer["references"]
                ]
            }
            
            formatted_answers.append(submission_answer)
        
        return {
            "team_email": team_email,
            "submission_name": submission_name,
            "answers": formatted_answers
        }
    
    def save_submission(self, submission: Dict[str, Any], output_file: str):
        """Save the submission to a JSON file"""
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                
            with open(output_file, 'w') as f:
                json.dump(submission, f, indent=2)
            
            logger.info(f"Submission saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving submission: {e}", exc_info=True)
            raise
    
    def process_questions(self, questions_file: str, team_email: str, submission_name: str, output_file: str):
        """Full pipeline to process questions and generate submission"""
        try:
            # Check if questions file exists
            if not os.path.exists(questions_file):
                raise FileNotFoundError(f"Questions file not found: {questions_file}")
                
            # Load questions
            with open(questions_file, 'r') as f:
                questions = json.load(f)
            
            logger.info(f"Loaded {len(questions)} questions from {questions_file}")
            
            # Answer questions
            answers = self.answer_questions(questions)
            
            # Format submission
            submission = self.format_submission(answers, team_email, submission_name)
            
            # Save submission
            self.save_submission(submission, output_file)
            
            return submission
        except Exception as e:
            logger.error(f"Error processing questions: {e}", exc_info=True)
            raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Enterprise RAG Challenge")
    
    # Required arguments
    parser.add_argument("--pdf_dir", required=True, help="Directory containing PDF files")
    
    # Processing options
    parser.add_argument("--extract_tables", action="store_true", default=True, 
                       help="Extract tables from PDFs")
    parser.add_argument("--skip_tables", dest="extract_tables", action="store_false",
                       help="Skip table extraction to save memory")
    parser.add_argument("--max_workers", type=int, default=2, 
                       help="Maximum number of parallel workers")
    parser.add_argument("--batch_size", type=int, default=10,
                       help="Batch size for processing")
    parser.add_argument("--skip_processed", action="store_true", default=True,
                       help="Skip processing PDFs that are already in the vector store")
    parser.add_argument("--force_reprocess", dest="skip_processed", action="store_false",
                       help="Force reprocessing of all PDFs even if already processed")
    
    # Model options
    parser.add_argument("--embedding_model", default="text-embedding-3-small",
                       help="Embedding model to use")
    parser.add_argument("--llm_model", default="gpt-4o-mini",
                       help="LLM model for answer generation")
    
    # Chunking options
    parser.add_argument("--chunk_size", type=int, default=800,
                       help="Chunk size for text splitting")
    parser.add_argument("--chunk_overlap", type=int, default=100,
                       help="Chunk overlap for text splitting")
    parser.add_argument("--top_k", type=int, default=5,
                       help="Number of chunks to retrieve per question")
    
    # Vector store options
    parser.add_argument("--vector_store_path", default=None,
                       help="Path to save/load vector store")
    
    # Processing mode options
    parser.add_argument("--process_only", action="store_true",
                       help="Only process PDFs and build vector store")
    parser.add_argument("--questions", help="JSON file containing questions")
    parser.add_argument("--team_email", help="Team email for submission")
    parser.add_argument("--submission_name", help="Name for the submission")
    parser.add_argument("--output", help="Output file for submission JSON")
    
    args = parser.parse_args()
    
    # Check question answering requirements if not process_only
    if not args.process_only:
        required_qa_args = ["questions", "team_email", "submission_name", "output"]
        missing_args = [arg for arg in required_qa_args if getattr(args, arg) is None]
        if missing_args:
            parser.error(f"--questions, --team_email, --submission_name, and --output are required "
                         f"when not in --process_only mode. Missing: {', '.join(missing_args)}")
    
    start_time = time.time()
    
    try:
        # Initialize RAG system
        rag = EnterpriseRAGChallenge(
            pdf_dir=args.pdf_dir,
            extract_tables=args.extract_tables,
            embedding_model=args.embedding_model,
            llm_model=args.llm_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            top_k=args.top_k,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            vector_store_path=args.vector_store_path,
            skip_processed=args.skip_processed
        )
        
        # Process PDFs and build vector store
        logger.info("Starting PDF processing")
        rag.process_pdfs()
        
        logger.info("Creating chunks")
        rag.create_chunks()
        
        logger.info("Building vector index")
        rag.build_vector_index()
        
        # Exit if only processing
        if args.process_only:
            elapsed = time.time() - start_time
            logger.info(f"Processing completed in {elapsed:.2f} seconds")
            return
        
        # Answer questions and generate submission
        logger.info("Processing questions")
        rag.process_questions(
            questions_file=args.questions,
            team_email=args.team_email,
            submission_name=args.submission_name,
            output_file=args.output
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Total execution time: {elapsed:.2f} seconds")
        
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()