"""
Chunking Strategy module for Enterprise RAG Challenge
Specialized text chunking for financial documents and annual reports
By Krasina15. Made with crooked kind hands
"""
import re
import logging
from typing import List, Dict, Any, Optional

# LangChain imports
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import DocumentSection from our module
try:
    from pdf_processor import DocumentSection
except ImportError:
    # Define a simplified version for standalone use that matches
    # the structure expected by other modules
    from dataclasses import dataclass
    
    @dataclass
    class DocumentSection:
        """Container for document sections with metadata"""
        title: str
        level: int
        content: str
        page: int
        is_table: bool = False
        table_data: Any = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChunkingStrategy:
    """Text chunking optimized for financial documents"""
    
    def __init__(self, chunk_size=800, chunk_overlap=100):
        """
        Initialize chunking strategy
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Specialized separators for financial documents
        self.separators = [
            # Document structure separators (strongest first)
            "\n\n\n",  # Triple line breaks
            "\n\n",    # Double line breaks
            ".\n",     # End of sentence with newline
            "\n",      # Single line break
            ". ",      # End of sentence
            
            # Financial document section headers
            "Financial Statements",
            "Balance Sheet",
            "Income Statement", 
            "Statement of Cash Flows",
            "Notes to the Consolidated Financial Statements",
            "Management's Discussion and Analysis",
            "Risk Factors",
            "Executive Compensation",
            "Corporate Governance",
            
            # Weaker separators
            ", ",      # Commas
            ": ",      # Colons
            " ",       # Spaces
            ""         # Character-level fallback
        ]
        
        # Create recursive text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators
        )
        
        # Financial keyword patterns for semantic chunking
        self.financial_patterns = [
            r"(\d{4}\s*Financial\s*Results)",
            r"((?:Q[1234]|Quarter\s*[1234])\s*\d{4})",
            r"((?:FY|Fiscal\s*Year)\s*\d{4})",
            r"((?:January|February|March|April|May|June|July|August|September|October|November|December)\s*\d{1,2},\s*\d{4})",
            r"(Table\s*\d+[\.:]\s*[A-Z])",
            r"(Note\s*\d+[\.:]\s*[A-Z])",
        ]
    
    def create_chunks(self, sections: List[DocumentSection], filename: str, sha1: str) -> List[Document]:
        """
        Create optimally chunked documents from sections
        
        Args:
            sections: List of document sections
            filename: Source filename
            sha1: SHA1 hash of the document
            
        Returns:
            List of LangChain Document objects with metadata
        """
        chunks = []
        
        if not sections:
            logger.warning(f"No sections provided for {filename}")
            return chunks
            
        try:
            for section in sections:
                # Special handling for tables - keep them whole
                if section.is_table:
                    # Don't split tables - preserve their structure
                    doc = Document(
                        page_content=section.content,
                        metadata={
                            "source": filename,
                            "sha1": sha1,
                            "page": section.page,
                            "title": section.title,
                            "is_table": True,
                            "section_level": section.level,
                            "document_type": "annual_report"
                        }
                    )
                    chunks.append(doc)
                else:
                    # Apply semantic chunking to text sections
                    text_chunks = self._create_semantic_chunks(section)
                    
                    # Convert to LangChain Documents
                    for i, chunk_text in enumerate(text_chunks):
                        # Skip empty chunks
                        if not chunk_text.strip():
                            continue
                        
                        doc = Document(
                            page_content=chunk_text,
                            metadata={
                                "source": filename,
                                "sha1": sha1,
                                "page": section.page,
                                "title": section.title,
                                "is_table": False,
                                "section_level": section.level,
                                "chunk_index": i,
                                "document_type": "annual_report"
                            }
                        )
                        chunks.append(doc)
        except Exception as e:
            logger.error(f"Error creating chunks for {filename}: {e}")
            # Continue processing remaining sections instead of failing completely
            
        return chunks
    
    def _create_semantic_chunks(self, section: DocumentSection) -> List[str]:
        """
        Create semantically meaningful chunks from a text section
        
        Args:
            section: Document section to split
            
        Returns:
            List of text chunks
        """
        text = section.content
        
        # Check for financial tables in text format (not caught as tables)
        if self._looks_like_financial_table(text):
            # Keep small tables as a single chunk 
            if len(text) <= self.chunk_size * 1.5:
                return [text]
        
        # Apply financial pattern splitting
        if section.level <= 2:  # Only for main sections
            presplit_chunks = self._presplit_on_patterns(text)
            
            # If we got meaningful presplits, split each one with the text splitter
            if len(presplit_chunks) > 1:
                final_chunks = []
                for presplit in presplit_chunks:
                    # Skip very small presplits or empty ones
                    if len(presplit.strip()) < 50:
                        continue
                    final_chunks.extend(self.text_splitter.split_text(presplit))
                return final_chunks
        
        # Default case: standard recursive splitting
        return self.text_splitter.split_text(text)
    
    def _presplit_on_patterns(self, text: str) -> List[str]:
        """
        Split text on financial patterns before the main splitting
        
        Args:
            text: Text to split
            
        Returns:
            List of pre-split chunks
        """
        chunks = [text]
        
        # Apply each pattern
        for pattern in self.financial_patterns:
            new_chunks = []
            for chunk in chunks:
                # Find all matches
                matches = list(re.finditer(pattern, chunk))
                if not matches:
                    new_chunks.append(chunk)
                    continue
                
                # Split on matches
                last_end = 0
                for match in matches:
                    if match.start() > last_end:
                        new_chunks.append(chunk[last_end:match.start()])
                    new_chunks.append(chunk[match.start():match.end() + 200])  # Include context after pattern
                    last_end = match.end()
                
                # Add remainder
                if last_end < len(chunk):
                    new_chunks.append(chunk[last_end:])
            
            chunks = new_chunks
        
        return chunks
    
    def _looks_like_financial_table(self, text: str) -> bool:
        """
        Check if text looks like a financial table
        
        Args:
            text: Text to check
            
        Returns:
            Boolean indicating if text likely contains a financial table
        """
        # Check for common financial table indicators
        financial_terms = ['revenue', 'income', 'profit', 'loss', 'expense',
                          'asset', 'liability', 'equity', 'cash', 'total']
        
        # Count lines with financial terms and numbers
        lines = text.lower().split('\n')
        financial_lines = 0
        number_lines = 0
        
        for line in lines:
            if any(term in line for term in financial_terms):
                financial_lines += 1
            
            # Count number patterns
            if re.search(r'[\d,.]+%?', line):
                number_lines += 1
        
        # Check density of financial content
        if len(lines) > 5:  # Need enough lines to be a table
            if financial_lines / len(lines) > 0.3 and number_lines / len(lines) > 0.5:
                return True
        
        return False