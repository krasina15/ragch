import os
import json
import argparse
import logging
import time
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Union, Literal, Type, TypeVar, cast

import pandas as pd
from tqdm import tqdm
import requests
import pypdf
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel, Field


# Define schema models based on the screenshots and challenge requirements
class PinType(BaseModel):
    pin_id: Optional[Union[int, str]] = Field(None, description="Pin number or name")
    type: str = Field(..., description="Pin type")


class IsolationTestVoltage(BaseModel):
    duration_sec: Optional[int] = None
    unit: Literal["VDC", "VAC", "Unknown"]
    voltage: int


class SourceReference(BaseModel):
    pdf_sha1: str = Field(..., description="SHA1 hash of the PDF file")
    page_index: int = Field(..., description="Zero-based physical page number in the PDF file")


class Answer(BaseModel):
    question_text: Optional[str] = Field(None, description="Text of the question")
    kind: Optional[Literal["number", "name", "boolean", "names"]] = Field(None, description="Kind of the question")
    value: Union[float, str, bool, List[str], Literal["N/A"]] = Field(..., description="Answer to the question, according to the question schema")
    references: List[SourceReference] = Field([], description="References to the source material in the PDF file")


class AnswerSubmission(BaseModel):
    answers: List[Answer] = Field(..., description="List of answers to the questions")
    team_email: str = Field(..., description="Email that your team used to register for the challenge")
    submission_name: str = Field(..., description="Unique name of the submission (e.g. experiment name)")


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnterpriseRAGChallenge:
    def __init__(self, 
                 pdf_dir: str, 
                 llm_provider: str = "openai",
                 api_key: Optional[str] = None,
                 embedding_model: str = "text-embedding-3-large",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 top_k_retrieval: int = 7,
                 model_name: str = "gpt-4o"):
        """
        Initialize the Enterprise RAG Challenge solver with improved parameters.
        
        Args:
            pdf_dir: Directory containing PDF files
            llm_provider: LLM provider ("openai" or "anthropic")
            api_key: API key for the LLM provider
            embedding_model: Model to use for embeddings
            chunk_size: Size of text chunks for embeddings
            chunk_overlap: Overlap between chunks
            top_k_retrieval: Number of chunks to retrieve for each question
            model_name: LLM model to use for answering questions
        """
        self.pdf_dir = pdf_dir
        self.llm_provider = llm_provider
        self.api_key = api_key or self._get_api_key(llm_provider)
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k_retrieval = top_k_retrieval
        self.model_name = model_name
        
        # Initialize storage for documents
        self.documents = {}
        self.document_sha1 = {}
        self.vectorstore = None
        
        # Initialize embeddings
        if "text-embedding" in embedding_model:
            self.embeddings = OpenAIEmbeddings(model=embedding_model)
        else:
            self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Initialize LLM based on provider
        self._initialize_llm()
    
    def _get_api_key(self, provider: str) -> str:
        """Get API key from environment variables based on provider."""
        if provider == "openai":
            return os.environ.get("OPENAI_API_KEY")
        elif provider == "anthropic":
            return os.environ.get("ANTHROPIC_API_KEY")
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def _initialize_llm(self):
        """Initialize the LLM based on the provider."""
        if self.llm_provider == "openai":
            self.llm = ChatOpenAI(
                model_name=self.model_name,
                temperature=0,
                api_key=self.api_key
            )
        elif self.llm_provider == "anthropic":
            self.llm = Anthropic(
                model="claude-3-opus-20240229",
                temperature=0,
                api_key=self.api_key
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
    def calculate_sha1(self, file_path: str) -> str:
        """Calculate SHA1 hash for a file."""
        sha1 = hashlib.sha1()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                sha1.update(chunk)
        return sha1.hexdigest()
    
    def load_pdfs(self):
        """Load all PDFs from the directory and calculate their SHA1 hashes."""
        logger.info(f"Loading PDFs from {self.pdf_dir}")
        
        for filename in tqdm(os.listdir(self.pdf_dir), desc="Loading PDFs"):
            if filename.endswith(".pdf"):
                file_path = os.path.join(self.pdf_dir, filename)
                sha1 = self.calculate_sha1(file_path)
                
                # Extract text from PDF with page tracking
                pages_text = self._extract_text_from_pdf_with_pages(file_path)
                
                # Store document and its hash
                self.documents[filename] = pages_text
                self.document_sha1[filename] = sha1
                
                logger.info(f"Loaded {filename} with SHA1: {sha1}")
        
        logger.info(f"Loaded {len(self.documents)} PDF documents")
    
    def _extract_text_from_pdf_with_pages(self, file_path: str) -> Dict[int, str]:
        """Extract text from a PDF file with page numbers as a dictionary."""
        pages_text = {}
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                page_count = len(pdf_reader.pages)
                
                for page_num in range(page_count):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        pages_text[page_num] = page_text.strip()
            
            return pages_text
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            return {}
    
    def create_vector_store(self):
        """Create a vector store from the loaded documents with improved chunking strategy."""
        logger.info("Creating vector store...")
        
        # Prepare documents for vectorization
        all_docs = []
        for filename, pages_text in self.documents.items():
            sha1 = self.document_sha1[filename]
            
            for page_num, text in pages_text.items():
                # Add header with metadata for each page
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "sha1": sha1,
                        "page": page_num,
                        # Include first 100 chars as metadata to help with retrieval
                        "preview": text[:100] if len(text) > 100 else text
                    }
                )
                all_docs.append(doc)
        
        # Split texts with optimized parameters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )
        all_splits = text_splitter.split_documents(all_docs)
        
        # Add chunk index to metadata
        for i, doc in enumerate(all_splits):
            doc.metadata["chunk_id"] = i
        
        # Create vector store
        self.vectorstore = FAISS.from_documents(all_splits, self.embeddings)
        logger.info(f"Vector store created with {len(all_splits)} document chunks")
    
    def answer_questions(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Answer a list of questions using the improved RAG system."""
        logger.info(f"Answering {len(questions)} questions...")
        
        answers = []
        
        for i, question_obj in enumerate(tqdm(questions, desc="Processing questions")):
            question_text = question_obj["text"]
            question_kind = question_obj["kind"]
            
            logger.info(f"Processing question {i+1}/{len(questions)}: {question_text}")
            
            # Try up to 3 times with different retrievals if needed
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    # Gradually increase k for retrieval in subsequent attempts
                    k_retrieval = self.top_k_retrieval + (attempt * 2)
                    answer, references = self._answer_single_question(
                        question_text, 
                        question_kind,
                        k=k_retrieval
                    )
                    break
                except Exception as e:
                    logger.warning(f"Attempt {attempt+1} failed: {e}")
                    if attempt == max_attempts - 1:
                        # If all attempts fail, return N/A
                        logger.error(f"All attempts failed for question: {question_text}")
                        answer = "N/A" if question_kind != "names" else []
                        references = []
            
            answers.append({
                "value": answer,
                "references": references
            })
        
        return answers
    
    def _answer_single_question(self, question: str, question_kind: str, k: int = None) -> Tuple[Any, List[Dict[str, Any]]]:
        """Answer a single question using enhanced RAG approach."""
        if k is None:
            k = self.top_k_retrieval
            
        # Create search query with question type hints
        search_query = self._enhance_query_for_search(question, question_kind)
        
        # Retrieve relevant documents
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        relevant_docs = retriever.get_relevant_documents(search_query)
        
        # Sort documents by relevance and page number for more coherent context
        relevant_docs.sort(key=lambda x: (x.metadata.get("page", 0)))
        
        # Create context from relevant documents
        context = self._format_context_from_docs(relevant_docs)
        
        # Prepare the prompt based on question type and enhance with anti-hallucination instructions
        prompt_template = self._get_enhanced_prompt_template(question_kind)
        
        # Execute query
        response = prompt_template.format(
            context=context,
            query=question
        )
        
        # Use the LLM directly for more control
        if self.llm_provider == "openai":
            result = self.llm.invoke(response)
            answer = result.content
        else:
            result = self.llm.invoke(response)
            answer = result
            
        # Format the answer based on the question kind
        formatted_answer = self._format_answer(answer, question_kind)
        
        # Extract references
        references = self._extract_references(relevant_docs)
        
        # Validate the answer
        self._validate_answer(formatted_answer, question_kind)
        
        return formatted_answer, references
    
    def _enhance_query_for_search(self, question: str, question_kind: str) -> str:
        """Enhance the query to improve retrieval by adding type-specific keywords."""
        # Extract main entities and key terms from the question
        entities = re.findall(r'"([^"]*)"', question)
        entities_str = " ".join(entities)
        
        # Add type-specific search terms
        if question_kind == "number":
            # For numerical questions, focus on finding statistics, figures, amounts
            hints = "financial figures statistics amounts numbers data"
        elif question_kind == "boolean":
            # For yes/no questions, focus on finding facts and statements
            hints = "facts information statements"
        elif question_kind in ["name", "names"]:
            # For name questions, focus on finding entities
            hints = "names entities people positions executives titles"
        else:
            hints = ""
            
        # Combine original question with enhanced terms
        enhanced_query = f"{question} {entities_str} {hints}"
        return enhanced_query
    
    def _format_context_from_docs(self, docs: List[Document]) -> str:
        """Format retrieved documents into a context string with improved structure."""
        if not docs:
            return "No relevant information found in the provided documents."
            
        context_parts = []
        
        # Group documents by source and page
        doc_groups = {}
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            sha1 = doc.metadata.get("sha1", "Unknown")
            page = doc.metadata.get("page", 0)
            
            key = (source, sha1, page)
            if key not in doc_groups:
                doc_groups[key] = []
                
            doc_groups[key].append(doc)
        
        # Format each group
        for (source, sha1, page), group_docs in doc_groups.items():
            content = "\n".join([doc.page_content for doc in group_docs])
            
            context_parts.append(
                f"SOURCE: {source}\n"
                f"DOCUMENT_ID: {sha1}\n"
                f"PAGE: {page}\n"
                f"CONTENT:\n{content}\n"
                f"{'=' * 50}\n"
            )
        
        return "\n".join(context_parts)
    
    def _get_enhanced_prompt_template(self, question_kind: str) -> PromptTemplate:
        """Get enhanced prompt templates with anti-hallucination instructions and specific formatting guidance."""
        
        # Common anti-hallucination instructions
        anti_hallucination = """
IMPORTANT: Only answer based on the information provided in the context above. If the information needed to answer the question is not present in the context, respond with 'N/A'. Do not make up or infer information that is not explicitly stated. If only partial information is available, provide only that partial information without speculation.

For each piece of information you include in your answer, identify which source document and page it comes from.
"""
        
        # Common prompt template for all question types
        template_start = """You are a precise financial document analyst tasked with answering questions about company information from annual reports.

CONTEXT:
{context}

QUESTION:
{query}
"""

        # Type-specific instructions
        if question_kind == "number":
            template_specific = """
INSTRUCTIONS:
1. Answer with ONLY a numeric value, with no additional text, words, or symbols.
2. For currency values, include ONLY the number (no currency symbols).
3. Use decimal points for fractional values (e.g., 12.5).
4. Do not use commas as thousand separators.
5. If the exact number isn't specified in the context, answer with 'N/A'.
6. Do not round or approximate unless the source text specifically does so.
"""
        elif question_kind == "boolean":
            template_specific = """
INSTRUCTIONS:
1. Answer ONLY with 'yes' or 'no'.
2. If the information cannot be determined from the context, answer with 'N/A'.
3. For comparing two companies, only answer 'yes' or 'no' if both companies have the relevant information.
4. Remember that absence of evidence is not evidence of absence - only answer 'no' if there is explicit contradictory information.
"""
        elif question_kind == "name":
            template_specific = """
INSTRUCTIONS:
1. Answer with ONLY the exact name requested.
2. Provide the full, properly formatted name without additional commentary.
3. If the information is not in the context, answer with 'N/A'.
4. Do not make assumptions about names based on titles or partial information.
"""
        elif question_kind == "names":
            template_specific = """
INSTRUCTIONS:
1. Answer with a list of names, separated by commas if there are multiple.
2. Provide complete, properly formatted names.
3. If there are no names that match the criteria, or if the information is not in the context, answer with 'N/A'.
4. Include ALL relevant names that match the criteria in the question.
5. Return the list in chronological or importance order if that information is available.
"""
        else:
            template_specific = """
INSTRUCTIONS:
1. Answer the question directly and concisely based only on the context provided.
2. If the information is not available in the context, answer with 'N/A'.
"""

        template_end = """
ANSWER FORMAT:
Your answer should contain only the requested information in the exact format specified above.

ANSWER:
"""

        full_template = template_start + template_specific + anti_hallucination + template_end
        return PromptTemplate(template=full_template, input_variables=["context", "query"])
    
    def _format_answer(self, answer: str, question_kind: str) -> Any:
        """Format the answer based on the question kind with improved parsing."""
        # Clean up the answer (remove quotes, extra spaces, etc.)
        answer = answer.strip().strip('"\'').strip()
        
        # Handle N/A answers
        na_patterns = ["n/a", "not available", "no information", "not applicable",
                      "cannot be determined", "insufficient information"]
        
        if any(pattern in answer.lower() for pattern in na_patterns):
            if question_kind == "names":
                return []
            return "N/A"
        
        if question_kind == "number":
            # Try to extract a number from the answer
            try:
                # First check for currency values with symbols
                currency_match = re.search(r'([\$€£¥])\s*([0-9,]+(\.[0-9]+)?)', answer)
                if currency_match:
                    number_str = currency_match.group(2).replace(',', '')
                    return float(number_str)
                
                # Then look for plain numbers
                number_match = re.search(r'([0-9,]+(\.[0-9]+)?)', answer)
                if number_match:
                    number_str = number_match.group(1).replace(',', '')
                    return float(number_str)
                
                # If no match or can't convert to float, return original
                return answer if answer else "N/A"
            except Exception:
                return answer if answer else "N/A"
            
        elif question_kind == "boolean":
            # Convert to boolean response
            answer_lower = answer.lower()
            if any(affirmative in answer_lower for affirmative in ["yes", "true", "correct"]):
                return "yes"
            elif any(negative in answer_lower for negative in ["no", "false", "incorrect"]):
                return "no"
            return answer if answer else "N/A"
            
        elif question_kind == "names":
            # Convert to a list of names
            if not answer or answer.lower() == "n/a":
                return []
                
            # Handle different separators
            if "," in answer:
                names = [name.strip() for name in answer.split(",")]
            elif " and " in answer:
                names = [name.strip() for name in answer.split(" and ")]
            elif ";" in answer:
                names = [name.strip() for name in answer.split(";")]
            elif "\n" in answer:
                names = [name.strip() for name in answer.split("\n") if name.strip()]
            else:
                names = [answer]
                
            # Filter out any empty strings
            names = [name for name in names if name]
            return names
            
        # For name type, return as is
        return answer if answer else "N/A"
    
    def _validate_answer(self, answer: Any, question_kind: str) -> None:
        """Validate that the answer matches the expected type."""
        if answer == "N/A":
            return
            
        if question_kind == "number":
            if not isinstance(answer, (int, float)) and not (isinstance(answer, str) and answer == "N/A"):
                logger.warning(f"Expected number but got: {answer}")
                
        elif question_kind == "boolean":
            if not isinstance(answer, str) or answer.lower() not in ["yes", "no", "n/a"]:
                logger.warning(f"Expected boolean but got: {answer}")
                
        elif question_kind == "names":
            if not isinstance(answer, list):
                logger.warning(f"Expected list of names but got: {answer}")
                
        elif question_kind == "name":
            if not isinstance(answer, str):
                logger.warning(f"Expected name string but got: {answer}")
    
    def _extract_references(self, docs: List[Document]) -> List[Dict[str, Any]]:
        """Extract references from relevant documents with improved deduplication."""
        references = []
        
        for doc in docs:
            sha1 = doc.metadata.get("sha1", "")
            page_num = doc.metadata.get("page", 0)
            
            # Only add reference if it has a valid SHA1 and page
            if sha1 and isinstance(page_num, int):
                references.append({
                    "pdf_sha1": sha1,
                    "page_index": page_num
                })
        
        # Deduplicate references
        unique_refs = []
        seen = set()
        
        for ref in references:
            key = (ref["pdf_sha1"], ref["page_index"])
            if key not in seen:
                unique_refs.append(ref)
                seen.add(key)
        
        return unique_refs
    
    def create_submission(self, questions: List[Dict[str, Any]], team_email: str, submission_name: str) -> Dict[str, Any]:
        """Create a submission for the RAG challenge with enhanced validation."""
        logger.info("Creating submission...")
        
        # Load PDFs if not already done
        if not self.documents:
            self.load_pdfs()
        
        # Create vector store if not already done
        if not self.vectorstore:
            self.create_vector_store()
        
        # Generate answers
        raw_answers = self.answer_questions(questions)
        
        # Add question text and kind to each answer
        answers = []
        for i, ans in enumerate(raw_answers):
            if i < len(questions):
                ans["question_text"] = questions[i]["text"]
                ans["kind"] = questions[i]["kind"]
            answers.append(ans)
        
        # Create submission object
        submission = {
            "team_email": team_email,
            "submission_name": submission_name,
            "answers": answers
        }
        
        # Validate submission
        try:
            AnswerSubmission.model_validate(submission)
            logger.info("Submission validation successful")
        except Exception as e:
            logger.error(f"Submission validation failed: {e}")
            
        return submission
    
    def save_submission(self, submission: Dict[str, Any], output_file: str):
        """Save the submission to a JSON file."""
        with open(output_file, 'w') as f:
            json.dump(submission, f, indent=2)
        
        logger.info(f"Submission saved to {output_file}")
    
    def submit_to_api(self, submission: Dict[str, Any], api_url: str):
        """Submit the answers to the competition API."""
        logger.info(f"Submitting to API: {api_url}")
        
        try:
            # Create a temporary file
            temp_file = "temp_submission.json"
            with open(temp_file, 'w') as f:
                json.dump(submission, f)
            
            # Submit using requests
            files = {
                "file": ("submission.json", open(temp_file, "rb"), "application/json")
            }
            headers = {"accept": "application/json"}
            
            response = requests.post(api_url, headers=headers, files=files)
            
            # Delete temporary file
            os.remove(temp_file)
            
            if response.status_code == 200:
                logger.info("Submission successful!")
                logger.info(f"Response: {response.json()}")
                return response.json()
            else:
                logger.error(f"Submission failed with status code {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error submitting to API: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description="Enterprise RAG Challenge Solver")
    parser.add_argument("--pdf_dir", required=True, help="Directory containing PDF files")
    parser.add_argument("--questions", required=True, help="JSON file containing questions")
    parser.add_argument("--output", required=True, help="Output file for submission JSON")
    parser.add_argument("--llm_provider", default="openai", choices=["openai", "anthropic"], help="LLM provider to use")
    parser.add_argument("--api_key", help="API key for the LLM provider")
    parser.add_argument("--team_email", required=True, help="Team email for submission")
    parser.add_argument("--submission_name", required=True, help="Submission name")
    parser.add_argument("--api_url", help="API URL for submission (optional)")
    parser.add_argument("--embedding_model", default="text-embedding-3-large", help="Embedding model to use")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Chunk size for text splitting")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Chunk overlap for text splitting")
    parser.add_argument("--top_k", type=int, default=7, help="Number of chunks to retrieve for each question")
    parser.add_argument("--model_name", default="gpt-4o", help="LLM model name")
    
    args = parser.parse_args()
    
    # Load questions
    with open(args.questions, 'r') as f:
        questions = json.load(f)
    
    # Initialize and run the challenge solver
    rag_challenge = EnterpriseRAGChallenge(
        pdf_dir=args.pdf_dir,
        llm_provider=args.llm_provider,
        api_key=args.api_key,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k_retrieval=args.top_k,
        model_name=args.model_name
    )
    
    # Create submission
    submission = rag_challenge.create_submission(
        questions=questions,
        team_email=args.team_email,
        submission_name=args.submission_name
    )
    
    # Save submission
    rag_challenge.save_submission(submission, args.output)
    
    # Submit to API if URL provided
    if args.api_url:
        rag_challenge.submit_to_api(submission, args.api_url)


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    logger.info(f"Total execution time: {elapsed_time:.2f} seconds")