import os
import json
import argparse
import logging
import time
import re
import hashlib
import gc
import traceback
from typing import List, Dict, Any, Optional, Tuple, Union, Literal, Generator, Iterator
from dataclasses import dataclass

import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
import psutil


# =========================
# Memory Monitoring Utility
# =========================
def log_memory_usage(label: str):
    """Log current memory usage with a label for tracking."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    logging.info(f"Memory usage at {label}: {memory_mb:.2f} MB")


# =========================
# Pydantic Models
# =========================
class SourceReference(BaseModel):
    pdf_sha1: str
    page_index: int


class Answer(BaseModel):
    question_text: Optional[str] = None
    kind: Optional[Literal["number", "name", "boolean", "names"]] = None
    value: Union[float, str, bool, List[str], Literal["N/A"]]
    references: List[SourceReference] = Field(default_factory=list)


class AnswerSubmission(BaseModel):
    answers: List[Answer]
    team_email: str
    submission_name: str


@dataclass
class EmbeddingJob:
    """Represents a batch of documents to be embedded."""
    docs: List[Document]
    file_sha1: str


# =========================
# PDF Loader & Chunker
# =========================
class PDFLoader:
    """
    Memory-efficient PDF loader that processes files one at a time and pages in batches.
    """

    def __init__(
            self,
            pdf_dir: str,
            chunk_size: int,
            chunk_overlap: int,
            store_year: bool = False,
            max_pages_per_file: Optional[int] = None  # Limit pages for very large PDFs
    ):
        self.pdf_dir = pdf_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.store_year = store_year
        self.max_pages_per_file = max_pages_per_file

        # Use more fine-grained separators for financial documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=[
                "\n\n\n",  # Triple line breaks often indicate section boundaries
                "\n\n",    # Double line breaks
                "\n",      # Single line breaks
                ". ",      # End of sentences
                ";\n",     # List items in financial docs often use semicolons
                ", ",      # Commas
                " ",       # Words
                ""         # Characters
            ]
        )

    def calculate_sha1(self, file_path: str) -> str:
        """Calculate SHA1 hash of a file."""
        sha1 = hashlib.sha1()
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                sha1.update(chunk)
        return sha1.hexdigest()

    def extract_year_from_text(self, text: str) -> Optional[int]:
        """
        Attempt to extract year from PDF text, looking for common patterns in annual reports.
        This is a more sophisticated approach than just using filename.
        """
        # Look for "Annual Report YYYY" or "YYYY Annual Report"
        matches = re.findall(r"Annual Report (?:for|of)?\s*(?:the year)?\s*(?:ended|ending)?\s*(?:in|on)?\s*(20\d{2})", text)
        if matches:
            return int(matches[0])
            
        # Look for "Fiscal Year YYYY" or "FY YYYY"
        matches = re.findall(r"(?:Fiscal Year|FY)\s+(20\d{2})", text)
        if matches:
            return int(matches[0])
            
        # Look for dates like "December 31, 2023"
        matches = re.findall(r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+(20\d{2})", text)
        if matches:
            return int(matches[0])
            
        return None

    def extract_company_name(self, text: str) -> Optional[str]:
        """
        Attempt to extract company name from the first few pages of text.
        """
        # Look for common patterns in annual reports
        matches = re.findall(r"([\w\s]+)(?:,? Inc\.?|,? Corporation|,? Ltd\.?|,? LLC|,? Limited|\bCorp\.?\b)(?:\s+and(?:\s+its)?\s+Subsidiaries)?", text)
        if matches:
            return matches[0].strip() + " " + re.search(r"(Inc\.?|Corporation|Ltd\.?|LLC|Limited|\bCorp\.?\b)", text).group(0)
            
        return None

    def yield_chunks(self) -> Generator[Document, None, None]:
        """
        Memory-efficient document processing that yields chunks one at a time.
        """
        pdf_files = [f for f in os.listdir(self.pdf_dir) if f.endswith(".pdf")]
        
        for pdf_file in pdf_files:
            file_path = os.path.join(self.pdf_dir, pdf_file)
            try:
                logging.info(f"Processing {pdf_file}")
                pdf_sha1 = self.calculate_sha1(file_path)
                metadata_extras = {}
                
                with open(file_path, 'rb') as fh:
                    reader = pypdf.PdfReader(fh)
                    num_pages = len(reader.pages)
                    
                    # First scan first few pages for year and company name
                    first_pages_text = ""
                    for i in range(min(5, num_pages)):
                        try:
                            first_pages_text += reader.pages[i].extract_text() + "\n"
                        except Exception as e:
                            logging.warning(f"Error reading page {i} for metadata: {e}")
                            
                    # Extract metadata from first pages
                    year = self.extract_year_from_text(first_pages_text)
                    if year:
                        metadata_extras["year"] = year
                        
                    company = self.extract_company_name(first_pages_text)
                    if company:
                        metadata_extras["company"] = company
                    
                    # Process all pages
                    pages_to_process = num_pages
                    if self.max_pages_per_file and num_pages > self.max_pages_per_file:
                        logging.warning(f"Limiting {pdf_file} to {self.max_pages_per_file} pages")
                        pages_to_process = self.max_pages_per_file
                        
                    for page_index in range(pages_to_process):
                        try:
                            page_obj = reader.pages[page_index]
                            raw_text = page_obj.extract_text()
                            if not raw_text or len(raw_text.strip()) < 10:  # Skip nearly empty pages
                                continue
                                
                            # Split page text into chunks
                            page_docs = self.text_splitter.create_documents([raw_text])
                            
                            for doc in page_docs:
                                # Add detailed metadata to each chunk
                                doc.metadata.update({
                                    "source": pdf_file,
                                    "sha1": pdf_sha1,
                                    "page": page_index,
                                    **metadata_extras
                                })
                                yield doc
                                
                        except Exception as e:
                            logging.error(f"Error processing page {page_index} of {pdf_file}: {e}")
                            continue
                            
                    # Explicitly clean up to help with memory
                    del reader
                    gc.collect()
                
            except Exception as e:
                logging.error(f"Error processing file {pdf_file}: {str(e)}")
                logging.error(traceback.format_exc())
                continue


# =========================
# Summarization Utility
# =========================
def summarize_text_chunks(
    text_chunks: List[str], 
    llm: ChatOpenAI,
    max_chars: int = 2000  # Reduced from 3000 to save memory
) -> str:
    """
    Summarize multiple text chunks into a single condensed text.
    This can reduce token usage in the final query.

    Args:
        text_chunks: list of strings from retrieved docs
        llm: an LLM object from langchain, e.g. ChatOpenAI
        max_chars: if text_chunks is huge, we skip some text

    Returns:
        A short summary as a string.
    """
    # Truncate or limit total text to save on tokens
    combined = []
    total_len = 0
    for chunk in text_chunks:
        if total_len + len(chunk) > max_chars:
            break
        combined.append(chunk)
        total_len += len(chunk)

    context = "\n".join(combined)
    if not context.strip():
        return "No relevant text found."

    # Improved prompt for summarization - more specific for financial data
    prompt = f"""You are an expert financial analyst summarizing key information. 
Focus on extracting factual financial data points, metrics, and explicit statements.

TEXT:
{context}

TASK: Create a concise summary that preserves:
1. All numerical values with their correct units
2. Company names with exact spelling
3. Years and dates precisely as stated  
4. Financial terminology without simplification
5. Facts rather than interpretations

Do not include any information not present in the source text.
SUMMARY:
"""

    resp = llm.invoke(prompt)
    summary = resp.content.strip()
    return summary


# =========================
# Main RAG Pipeline
# =========================
class EnterpriseRAGChallenge:
    def __init__(
        self,
        pdf_dir: str,
        llm_provider: str = "openai",
        api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",  # Default changed to a lighter model
        chunk_size: int = 600,  # Reduced from 800 for memory efficiency
        chunk_overlap: int = 80,  # Reduced from 100 for memory efficiency
        two_phase_k1: int = 15,  # Reduced from 20 for memory efficiency
        final_top_k: int = 5,
        model_name: str = "gpt-4",
        use_disk_annoy: bool = True,  # Default changed to True for memory efficiency 
        annoy_index_path: str = "my_annoy_index",
        store_year_in_metadata: bool = True,  # Default changed to True
        filter_year: Optional[int] = None,
        n_trees: int = 30,  # Increased from 10 for better accuracy
        batch_size: int = 50,  # New parameter for batched processing
        max_pages_per_file: Optional[int] = None,  # Limit very large PDFs
        temp_dir: str = "temp_embeddings"  # Directory for temporary embedding storage
    ):
        """
        Memory-optimized RAG pipeline for annual report analysis.
        """
        self.pdf_dir = pdf_dir
        self.llm_provider = llm_provider
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.two_phase_k1 = two_phase_k1
        self.final_top_k = final_top_k

        self.model_name = model_name
        self.use_disk_index = use_disk_annoy  # Renamed but kept parameter name for backward compatibility
        self.faiss_index_path = annoy_index_path # Renamed but kept parameter name for backward compatibility
        self.store_year_in_metadata = store_year_in_metadata
        self.filter_year = filter_year
        self.n_trees = n_trees
        self.batch_size = batch_size
        self.max_pages_per_file = max_pages_per_file
        self.temp_dir = temp_dir

        # Create temp directory if it doesn't exist
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

        # Prepare embeddings - prefer local models for memory efficiency
        if embedding_model == "text-embedding-ada-002":
            self.embeddings = OpenAIEmbeddings(
                model=self.embedding_model,
                openai_api_key=self.api_key
            )
        else:
            # HuggingFace embeddings are more memory-efficient for local use
            self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)

        self._initialize_llm()
        self.vectorstore = None
        
        # Store for documents and their metadata
        self.document_store = {}
        
        # Maintain a mapping of document chunks to their sources for reference tracking
        self.chunk_to_source_map = {}

    def _initialize_llm(self):
        """Initialize the LLM based on provider configuration."""
        if self.llm_provider == "openai":
            self.llm = ChatOpenAI(
                model_name=self.model_name,
                temperature=0,
                openai_api_key=self.api_key
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def build_vectorstore_in_batches(self):
        """
        Memory-efficient vector store building that processes documents in batches.
        With FAISS, we can add documents incrementally to the index.
        """
        log_memory_usage("Before building vectorstore")
        
        loader = PDFLoader(
            pdf_dir=self.pdf_dir,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            store_year=self.store_year_in_metadata,
            max_pages_per_file=self.max_pages_per_file
        )

        # Process in batches
        batch = []
        batch_count = 0
        doc_count = 0
        
        logging.info(f"Starting document processing in batches of {self.batch_size}")
        
        for doc in loader.yield_chunks():
            # Track document sources for later reference
            doc_id = f"{doc_count}"
            self.chunk_to_source_map[doc_id] = {
                "sha1": doc.metadata.get("sha1", ""),
                "page": doc.metadata.get("page", 0),
                "source": doc.metadata.get("source", ""),
                "content": doc.page_content[:100] + "..."  # Store a snippet for debugging
            }
            
            # Add document ID to metadata
            doc.metadata["doc_id"] = doc_id
            
            batch.append(doc)
            doc_count += 1
            
            # Process batch when it reaches batch_size
            if len(batch) >= self.batch_size:
                self._process_batch(batch)
                batch_count += 1
                batch = []
                
                # Periodically save index to disk if configured
                if self.use_disk_index and batch_count % 10 == 0:
                    self.save_faiss_index()
                
                # Force garbage collection to free memory
                gc.collect()
                
                if batch_count % 5 == 0:
                    log_memory_usage(f"After processing {batch_count} batches ({doc_count} docs)")
        
        # Process final batch if not empty
        if batch:
            self._process_batch(batch)
            
        logging.info(f"Completed processing {doc_count} documents in {batch_count + 1} batches")
        
        # Save final index to disk
        if self.use_disk_index:
            self.save_faiss_index()
        
        log_memory_usage("After building vectorstore")

    def _process_batch(self, batch: List[Document]):
        """Process a batch of documents, adding them incrementally to the FAISS index."""
        try:
            if not batch:
                return
                
            # Store documents for reference lookup
            for doc in batch:
                doc_id = doc.metadata.get("doc_id")
                if doc_id not in self.document_store:
                    self.document_store[doc_id] = doc
            
            # Add to existing vectorstore if we have one
            if self.vectorstore is not None:
                self.vectorstore.add_documents(batch)
            else:
                # Initialize the vectorstore with the first batch
                self.vectorstore = FAISS.from_documents(
                    documents=batch,
                    embedding=self.embeddings
                )
                
        except Exception as e:
            logging.error(f"Error processing batch: {e}")
            logging.error(traceback.format_exc())

    def save_faiss_index(self):
        """Save the FAISS index to disk if configured to do so."""
        if not self.use_disk_index or not self.vectorstore:
            return
            
        try:
            # Save FAISS index
            logging.info(f"Saving FAISS index to {self.faiss_index_path}")
            self.vectorstore.save_local(self.faiss_index_path)
            
            # Save metadata alongside index for document reference lookups
            metadata_path = os.path.join(self.faiss_index_path, "chunk_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(self.chunk_to_source_map, f)
                
            logging.info(f"Saved index metadata to {metadata_path}")
            
        except Exception as e:
            logging.error(f"Error saving FAISS index: {e}")
            logging.error(traceback.format_exc())
    
    def build_or_load_vectorstore(self):
        """Build or load the FAISS vector store from disk."""
        if self.use_disk_index and os.path.exists(self.faiss_index_path):
            logging.info("Loading FAISS index from disk...")
            try:
                # Load FAISS index
                self.vectorstore = FAISS.load_local(
                    self.faiss_index_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                
                # Load document metadata if available
                metadata_path = os.path.join(self.faiss_index_path, "chunk_metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        self.chunk_to_source_map = json.load(f)
                    logging.info(f"Loaded metadata for {len(self.chunk_to_source_map)} chunks")
                else:
                    logging.warning("No metadata file found alongside index")
                
                # Validate the loaded index
                if not self.vectorstore:
                    logging.warning("Failed to load valid index, rebuilding...")
                    self.build_vectorstore_in_batches()
            except Exception as e:
                logging.error(f"Error loading index: {e}")
                logging.error(traceback.format_exc())
                self.build_vectorstore_in_batches()
        else:
            logging.info("Building FAISS index from scratch...")
            self.build_vectorstore_in_batches()

    def filter_docs_by_metadata(self, docs: List[Document]) -> List[Document]:
        """Filter documents by metadata criteria."""
        if not docs:
            return []
            
        filtered_docs = docs
        
        # Filter by year if specified
        if self.filter_year is not None:
            filtered_docs = [d for d in filtered_docs if d.metadata.get("year") == self.filter_year]
            
        return filtered_docs

    def rerank_documents(self, docs: List[Document], query: str) -> List[Document]:
        """
        Rerank documents based on relevance to the query.
        This is a more sophisticated reranking than just taking the top K by order.
        """
        if not docs:
            return []
            
        # For a proper implementation, we'd use a cross-encoder model
        # But for memory constraints, we'll use a simple keyword matching approach
        query_keywords = set(query.lower().split())
        scored_docs = []
        
        for doc in docs:
            # Simple keyword matching score
            doc_text = doc.page_content.lower()
            score = sum(1 for kw in query_keywords if kw in doc_text)
            
            # Boost score for exact phrase matches
            if query.lower() in doc_text:
                score += 5
                
            # Add distance to score with some weighting
            scored_docs.append((doc, score))
            
        # Sort by score descending
        sorted_docs = [doc for doc, score in sorted(scored_docs, key=lambda x: x[1], reverse=True)]
        return sorted_docs

    def retrieve_context_with_references(self, question: str) -> Tuple[str, List[SourceReference]]:
        """
        Retrieve and summarize relevant context while tracking source references.
        Returns the summarized context and a list of source references.
        """
        # Ensure vector store is built
        if not self.vectorstore:
            self.build_or_load_vectorstore()
            
        # Create retriever with larger k for initial retrieval
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.two_phase_k1})
        
        try:
            # Retrieve initial documents
            docs_stage1 = retriever.get_relevant_documents(question)
            log_memory_usage("After initial retrieval")
            
            # Filter by metadata
            docs_stage1 = self.filter_docs_by_metadata(docs_stage1)
            
            # Rerank documents based on relevance to question
            reranked_docs = self.rerank_documents(docs_stage1, question)
            
            # Take final top K
            docs_final = reranked_docs[:self.final_top_k]
            
            # Extract source references
            references = []
            for doc in docs_final:
                # First try using doc_id
                doc_id = doc.metadata.get("doc_id")
                if doc_id and doc_id in self.chunk_to_source_map:
                    source_info = self.chunk_to_source_map[doc_id]
                    references.append(
                        SourceReference(
                            pdf_sha1=source_info["sha1"],
                            page_index=source_info["page"]
                        )
                    )
                else:
                    # Fallback to metadata directly (for FAISS, the metadata should still be in the docs)
                    sha1 = doc.metadata.get("sha1")
                    page = doc.metadata.get("page", 0)
                    if sha1:
                        references.append(
                            SourceReference(
                                pdf_sha1=sha1,
                                page_index=page
                            )
                        )
            
            # Prepare text chunks for summarization
            text_chunks = []
            for doc in docs_final:
                chunk_text = f"(Document: {doc.metadata.get('source','?')}, Page: {doc.metadata.get('page','?')}) {doc.page_content}"
                text_chunks.append(chunk_text)
                
            # Summarize the final content
            summary = summarize_text_chunks(text_chunks, self.llm)
            
            return summary, references
            
        except Exception as e:
            logging.error(f"Error in retrieval: {e}")
            logging.error(traceback.format_exc())
            return "Error retrieving context.", []

    def answer_single_question(
        self,
        question_text: str,
        question_kind: str
    ) -> Tuple[Any, List[SourceReference]]:
        """
        Retrieve context, answer the question, and track references.
        """
        # Get context and references
        unified_context, references = self.retrieve_context_with_references(question_text)
        
        # Construct prompt based on question kind
        final_prompt = self._construct_prompt(question_text, unified_context, question_kind)
        
        try:
            # Get answer from LLM
            llm_result = self.llm.invoke(final_prompt)
            raw_answer = llm_result.content.strip()
            
            # Parse and validate answer based on question kind
            final_answer = self._post_process_answer(raw_answer, question_kind)
            
            return final_answer, references
            
        except Exception as e:
            logging.error(f"Error generating answer: {e}")
            # Return N/A with any references found
            if question_kind == "names":
                return [], references
            else:
                return "N/A", references

    def _construct_prompt(self, question_text: str, context: str, question_kind: str) -> str:
        """Create a prompt tailored to the question type."""
        # Base instructions
        instructions = (
            "You are a precise financial analyst extracting specific information from annual reports. "
            "Use ONLY the information in the provided context to answer the question. "
            "If the exact information is not in the context, respond with 'N/A'. "
            "DO NOT make up or infer data not explicitly stated in the context.\n\n"
        )
        
        # Add question-specific formatting instructions based on kind
        if question_kind == "number":
            instructions += (
                "FORMAT INSTRUCTIONS: Your answer must be a NUMBER ONLY without any text, symbols, "
                "or currency notations. For example: 42.5 or 1000 or 0.75\n"
                "If the number is present but with different units, convert it to match the question's units.\n"
            )
        elif question_kind == "boolean":
            instructions += (
                "FORMAT INSTRUCTIONS: Your answer must be either 'yes' or 'no' only.\n"
                "Answer 'N/A' if there's not enough information to determine with certainty.\n"
            )
        elif question_kind == "name":
            instructions += (
                "FORMAT INSTRUCTIONS: Your answer must be a NAME ONLY, without additional description "
                "or explanation. For example: 'John Smith' or 'Acme Corporation'.\n"
            )
        elif question_kind == "names":
            instructions += (
                "FORMAT INSTRUCTIONS: Your answer must be a list of names separated by commas. "
                "Include only the names, not descriptions or explanations. "
                "Return an empty list [] if no names are found.\n"
            )
            
        # Construct the final prompt
        return f"{instructions}CONTEXT:\n{context}\n\nQUESTION: {question_text}\nANSWER:"

    def _post_process_answer(self, raw_answer: str, question_kind: str):
        """
        Process the LLM's response into the correct format for the question type.
        This includes handling edge cases and ensuring consistent output.
        """
        raw_lower = raw_answer.lower().strip()
        
        # Handle any variations of "not applicable" with standardized N/A
        if any(phrase in raw_lower for phrase in ["n/a", "not applicable", "insufficient info", "no information", "cannot determine", "not enough"]):
            if question_kind == "names":
                return []
            return "N/A"
            
        if question_kind == "boolean":
            # Handle boolean questions
            if any(affirmative in raw_lower for affirmative in ["yes", "true", "correct", "affirmative"]):
                return "yes"
            elif any(negative in raw_lower for negative in ["no", "false", "incorrect", "negative"]):
                return "no"
            else:
                return "N/A"
                
        elif question_kind == "number":
            # Handle numeric questions
            # Look for numbers in the response, prioritizing those with decimal points and proper formatting
            # More sophisticated regex to extract numeric values
            matches = re.findall(r"(-?\d+(?:\.\d+)?)", raw_lower)
            if matches:
                # Filter and prioritize numbers that look like they're the main answer
                if len(matches) == 1:
                    return matches[0]
                    
                # If multiple matches, try to find the one most likely to be the answer
                # For example, prefer a number that follows "is" or appears at the start/end
                for pattern in [r"is\s+(-?\d+(?:\.\d+)?)", r"equals\s+(-?\d+(?:\.\d+)?)", r"^(-?\d+(?:\.\d+)?)", r"(-?\d+(?:\.\d+)?)\s*$"]:
                    specific_match = re.search(pattern, raw_lower)
                    if specific_match:
                        return specific_match.group(1)
                        
                # Default to first match if no better option
                return matches[0]
            return "N/A"
            
        elif question_kind == "names":
            # Handle list of names questions
            if "n/a" in raw_lower or "[]" in raw_lower or "none" in raw_lower:
                return []
                
            # Try to parse a list from the response
            if "[" in raw_answer and "]" in raw_answer:
                # Extract content between brackets
                list_content = re.search(r"\[(.*?)\]", raw_answer).group(1)
                names = [name.strip().strip('"\'') for name in list_content.split(",") if name.strip()]
                return names
                
            # Handle comma-separated names
            names = [name.strip() for name in re.split(r",|\band\b|\n", raw_answer) if name.strip()]
            return names
            
        else:  # "name" or default
            # Handle single name questions - take just the name without explanation
            if "n/a" in raw_lower:
                return "N/A"
                
            # Try to extract just the name part
            lines = raw_answer.split("\n")
            if len(lines) > 0:
                # Take the first non-empty line as the answer
                for line in lines:
                    if line.strip():
                        # Further clean up any explanations
                        name_part = re.split(r"[:\-–]", line)[0].strip()
                        return name_part
                        
            # Fallback to the whole answer
            return raw_answer

    def answer_questions(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process all questions and return answers with references.
        This method handles the overall pipeline.
        """
        if not self.vectorstore:
            self.build_or_load_vectorstore()

        answers = []
        for i, q in enumerate(questions):
            logging.info(f"Processing question {i+1}/{len(questions)}: {q['text'][:50]}...")
            q_text = q["text"]
            q_kind = q.get("kind", "text")
            
            try:
                final_ans, refs = self.answer_single_question(q_text, q_kind)
                answers.append({
                    "value": final_ans,
                    "references": [ref.dict() for ref in refs]
                })
                
                # Log partial progress
                if (i + 1) % 5 == 0:
                    log_memory_usage(f"After processing {i+1} questions")
                    
            except Exception as e:
                logging.error(f"Error processing question {i+1}: {str(e)}")
                logging.error(traceback.format_exc())
                # Provide fallback answer in case of errors
                answers.append({
                    "value": "N/A" if q_kind != "names" else [],
                    "references": []
                })
                
        return answers

    def create_submission(
        self,
        questions: List[Dict[str, Any]],
        team_email: str,
        submission_name: str
    ) -> Dict[str, Any]:
        """
        Create the final submission with answers and references.
        """
        log_memory_usage("Before answering questions")
        raw_answers = self.answer_questions(questions)
        log_memory_usage("After answering questions")
        
        final_answers = []
        for q, ans in zip(questions, raw_answers):
            answer_obj = {
                "question_text": q["text"],
                "kind": q["kind"],
                "value": ans["value"],
                "references": ans["references"]
            }
            final_answers.append(answer_obj)
            
        submission = {
            "team_email": team_email,
            "submission_name": submission_name,
            "answers": final_answers
        }
        
        # Validate submission against schema
        try:
            AnswerSubmission(**submission)
        except Exception as e:
            logging.error(f"Validation error in submission: {str(e)}")
            
        return submission


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", required=True, help="Directory containing PDF files")
    parser.add_argument("--questions", required=True, help="JSON file with questions")
    parser.add_argument("--output", required=True, help="Output file for submission")
    parser.add_argument("--team_email", required=True, help="Team email for submission")
    parser.add_argument("--submission_name", required=True, help="Submission name")

    # RAG pipeline config
    parser.add_argument("--llm_provider", default="openai", choices=["openai"])
    parser.add_argument("--api_key", default=None, help="API key for LLM provider")
    parser.add_argument("--embedding_model", default="all-MiniLM-L6-v2", help="Model for embeddings")
    parser.add_argument("--chunk_size", type=int, default=600, help="Characters per chunk")
    parser.add_argument("--chunk_overlap", type=int, default=80, help="Overlap between chunks")
    parser.add_argument("--two_phase_k1", type=int, default=15, help="First-phase retrieval size")
    parser.add_argument("--final_top_k", type=int, default=5, help="Final top K after reranking")
    parser.add_argument("--model_name", default="gpt-3.5-turbo", help="LLM model name")
    parser.add_argument("--use_disk_annoy", action="store_true", help="Use disk-based Annoy index")
    parser.add_argument("--annoy_index_path", default="my_annoy_index", help="Path for Annoy index")
    parser.add_argument("--store_year_in_metadata", action="store_true", help="Extract year from text")
    parser.add_argument("--filter_year", type=int, default=None, help="Filter by year")
    parser.add_argument("--n_trees", type=int, default=30, help="Annoy index n_trees param")
    parser.add_argument("--batch_size", type=int, default=50, help="Documents per batch")
    parser.add_argument("--max_pages_per_file", type=int, default=None, help="Limit pages for large PDFs")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()

    # Configure logging
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=getattr(logging, args.log_level), format=log_format)

    log_memory_usage("At start")
    
    try:
        with open(args.questions, "r") as f:
            questions_data = json.load(f)
    except Exception as e:
        logging.error(f"Error loading questions: {str(e)}")
        return

    pipeline = EnterpriseRAGChallenge(
        pdf_dir=args.pdf_dir,
        llm_provider=args.llm_provider,
        api_key=args.api_key,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        two_phase_k1=args.two_phase_k1,
        final_top_k=args.final_top_k,
        model_name=args.model_name,
        use_disk_annoy=args.use_disk_annoy,  # Using the original parameter name for compatibility
        annoy_index_path=args.annoy_index_path,  # Using the original parameter name for compatibility
        store_year_in_metadata=args.store_year_in_metadata,
        filter_year=args.filter_year,
        n_trees=args.n_trees,  # Not used with FAISS but kept for compatibility
        batch_size=args.batch_size,
        max_pages_per_file=args.max_pages_per_file
    )

    try:
        submission = pipeline.create_submission(
            questions_data,
            team_email=args.team_email,
            submission_name=args.submission_name
        )

        with open(args.output, "w") as out_f:
            json.dump(submission, out_f, indent=2)

        print(f"Submission saved to {args.output}")
        log_memory_usage("At end")
        
    except Exception as e:
        logging.error(f"Error creating submission: {str(e)}")
        logging.error(traceback.format_exc())


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"Done! Elapsed time: {elapsed:.2f}s")