"""
PDF Processor module for Enterprise RAG Challenge
Handles efficient extraction of text and tables from PDF files
By Krasina15. Made with crooked kind hands
"""
import os
import re
import gc
import base64
import tempfile
import hashlib
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing PyMuPDF
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    logger.warning("PyMuPDF not available. Install with 'pip install pymupdf'")

# OpenAI imports
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    logger.warning("OpenAI not available. Install with 'pip install openai'")


@dataclass
class DocumentSection:
    """Container for document sections with metadata"""
    title: str
    level: int
    content: str
    page: int
    is_table: bool = False
    table_data: Any = None


class PDFProcessor:
    """Efficient PDF processing with memory optimization"""
    
    def __init__(self, extract_tables=True, batch_size=5, vision_model="gpt-4o-mini"):
        """
        Initialize PDF processor
        
        Args:
            extract_tables: Whether to extract tables (requires OpenAI API access)
            batch_size: Batch size for processing multiple PDFs
            vision_model: OpenAI vision model to use for table extraction
        """
        self.extract_tables = extract_tables and HAS_OPENAI
        self.batch_size = batch_size
        self.vision_model = vision_model
        
        # Check dependencies
        if not HAS_PYMUPDF:
            logger.error("PyMuPDF is required for PDF processing")
        
        if extract_tables and not HAS_OPENAI:
            logger.warning("OpenAI not available. Install with 'pip install openai'")
            self.extract_tables = False
            
        # NOTE: We don't initialize OpenAI client here anymore
        # to avoid pickling issues with multiprocessing
        # Instead, a new client is created when needed in _extract_tables_with_vision
    
    def calculate_sha1(self, file_path: str) -> str:
        """Calculate SHA1 hash for a file"""
        try:
            sha1 = hashlib.sha1()
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192):
                    sha1.update(chunk)
            return sha1.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating SHA1 for {file_path}: {e}")
            # Return a placeholder value rather than failing
            return "error_calculating_sha1"
    
    def process_pdf(self, file_path: str) -> Tuple[str, List[DocumentSection]]:
        """
        Process a single PDF file
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (sha1_hash, list_of_sections)
        """
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return "file_not_found", []
            
        # Calculate SHA1 for the file
        sha1 = self.calculate_sha1(file_path)
        
        try:
            # Extract content from PDF
            sections = self._extract_content(file_path)
            return sha1, sections
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return sha1, []
    
    def _extract_content(self, file_path: str) -> List[DocumentSection]:
        """
        Extract structured content from PDF
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of DocumentSection objects
        """
        if not HAS_PYMUPDF:
            logger.error("PyMuPDF not available")
            return []
        
        sections = []
        doc = None
        
        try:
            # Open PDF with PyMuPDF
            doc = fitz.open(file_path)
            
            # Extract table of contents if available
            toc = doc.get_toc()
            section_markers = {}
            if toc:
                for level, title, page in toc:
                    section_markers[page-1] = (level, title)  # Convert to 0-based
            
            # Extract metadata
            metadata = {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "keywords": doc.metadata.get("keywords", ""),
                "page_count": doc.page_count
            }
            
            # Process each page
            for page_num in range(doc.page_count):
                # Check if this page starts a new section
                title = f"Page {page_num + 1}"
                level = 1
                
                if page_num in section_markers:
                    level, title = section_markers[page_num]
                
                # Get page
                page = doc[page_num]
                
                # Extract text
                text = page.get_text("text")
                
                # Skip empty pages
                if not text.strip():
                    continue
                
                # Add text section
                sections.append(DocumentSection(
                    title=title,
                    level=level,
                    content=text,
                    page=page_num,
                    is_table=False
                ))
                
                # Extract tables if enabled
                if self.extract_tables and self._has_potential_table(text):
                    # Get page as image for OpenAI Vision API
                    table_sections = self._extract_tables_with_vision(file_path, page_num, page)
                    sections.extend(table_sections)
                
                # Clean up memory every few pages
                if page_num % 10 == 0:
                    gc.collect()
            
            # Add file metadata as a special section
            metadata_text = "\n".join([f"{k}: {v}" for k, v in metadata.items() if v])
            if metadata_text:
                sections.append(DocumentSection(
                    title="Document Metadata",
                    level=1,
                    content=metadata_text,
                    page=0,  # Always associate with first page
                    is_table=False
                ))
            
            return sections
            
        except Exception as e:
            logger.error(f"Error in _extract_content for {file_path}: {e}")
            return []
            
        finally:
            # Ensure document is closed even if an exception occurs
            if doc:
                try:
                    doc.close()
                except Exception:
                    pass
    
    def _has_potential_table(self, text: str) -> bool:
        """
        Detect if page might contain tables based on text patterns
        
        Args:
            text: Page text content
            
        Returns:
            Boolean indicating if page likely contains tables
        """
        # Count lines with high percentage of numbers or common separators
        lines = text.split('\n')
        total_lines = len([line for line in lines if line.strip()])
        if total_lines == 0:
            return False
            
        # Count potential table rows
        potential_table_rows = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for common table patterns
            if any(pattern in line for pattern in ['|', '\t']):
                potential_table_rows += 1
                continue
                
            # Check for numeric patterns
            digits = sum(c.isdigit() for c in line)
            digits_percent = digits / len(line) if len(line) > 0 else 0
            
            if digits_percent > 0.3:
                potential_table_rows += 1
        
        # Return true if enough potential table rows found
        table_row_percent = potential_table_rows / total_lines if total_lines > 0 else 0
        return potential_table_rows >= 3 and table_row_percent > 0.15
    
    def _extract_tables_with_vision(self, file_path: str, page_num: int, page) -> List[DocumentSection]:
        """
        Extract tables from a specific page using OpenAI Vision API
        
        Args:
            file_path: Path to the PDF file
            page_num: Page number (0-indexed)
            page: PyMuPDF page object
            
        Returns:
            List of DocumentSection objects with tables
        """
        if not HAS_OPENAI:
            return []
        
        table_sections = []
        temp_path = None
        
        try:
            # Create an OpenAI client in this process
            # (Create a new client each time to avoid pickling issues)
            openai_client = OpenAI()
            
            # Render page to a temporary image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
            
            # Use a temporary file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                temp_path = temp_file.name
                pix.save(temp_path)
            
            # Encode image to base64
            with open(temp_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Call OpenAI Vision API with proper error handling and rate limit consideration
            try:
                response = openai_client.chat.completions.create(
                    model=self.vision_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a table extraction expert. Extract all tables from the image and format them as properly aligned text tables. Only extract tables, not other content."
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Extract all tables from this page of a financial document. Format each table as a plain text table with columns properly aligned. If there are no tables, respond with 'NO_TABLES_FOUND'."},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                            ]
                        }
                    ],
                    max_tokens=1500
                )
                
                # Process response
                extracted_text = response.choices[0].message.content
                
                # Check if no tables were found
                if "NO_TABLES_FOUND" in extracted_text:
                    return []
                
                # Split into separate tables if multiple
                table_texts = []
                
                # Try to identify individual tables in the response
                potential_tables = re.split(r'\n\s*Table\s+\d+[:\.\)]*\s*\n', extracted_text)
                if len(potential_tables) > 1:
                    # Remove the intro text if present
                    potential_tables = potential_tables[1:] if not _looks_like_table(potential_tables[0]) else potential_tables
                    table_texts = potential_tables
                else:
                    # Just one table or no clear table markers
                    table_texts = [extracted_text]
                
                # Create a section for each table
                for i, table_text in enumerate(table_texts):
                    if not _looks_like_table(table_text):
                        continue
                        
                    # Create structured section
                    table_section = DocumentSection(
                        title=f"Table {i+1} on page {page_num + 1}",
                        level=3,
                        content=table_text.strip(),
                        page=page_num,
                        is_table=True
                    )
                    table_sections.append(table_section)
                
            except Exception as e:
                logger.warning(f"OpenAI API error on page {page_num}: {e}")
                # Continue processing without this table
            
        except Exception as e:
            logger.warning(f"Table extraction error on page {page_num}: {e}")
        
        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
        
        return table_sections
    
    def process_directory(self, pdf_dir: str, max_workers: int = 2) -> Dict[str, Dict]:
        """
        Process all PDFs in a directory with parallelization
        
        Args:
            pdf_dir: Directory containing PDF files
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary with results for each PDF
        """
        # Find all PDF files
        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
        total_files = len(pdf_files)
        
        if total_files == 0:
            logger.warning(f"No PDF files found in {pdf_dir}")
            return {}
            
        logger.info(f"Processing {total_files} PDF files with {max_workers} workers")
        
        results = {}
        
        # Use ProcessPoolExecutor for CPU-bound PDF processing
        if max_workers > 1 and not self.extract_tables:
            # Only use multiprocessing if not extracting tables with OpenAI 
            # (avoids pickling issues with the OpenAI client)
            # Process PDFs in batches to control memory usage
            for batch_start in range(0, total_files, self.batch_size):
                batch_end = min(batch_start + self.batch_size, total_files)
                batch = pdf_files[batch_start:batch_end]
                
                logger.info(f"Processing batch {batch_start//self.batch_size + 1}: {len(batch)} files")
                
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    # Submit batch for parallel processing
                    future_to_pdf = {
                        executor.submit(self.process_pdf, os.path.join(pdf_dir, pdf)): pdf 
                        for pdf in batch
                    }
                    
                    # Collect results
                    for future in as_completed(future_to_pdf):
                        pdf = future_to_pdf[future]
                        try:
                            sha1, sections = future.result()
                            results[pdf] = {
                                "sha1": sha1,
                                "sections": sections,
                                "page_count": len(set(section.page for section in sections))
                            }
                            logger.info(f"Processed {pdf}: {len(sections)} sections, SHA1: {sha1}")
                        except Exception as e:
                            logger.error(f"Error with {pdf}: {e}")
                
                # Clean up after batch is complete
                gc.collect()
        else:
            # Single process mode (always use this when extracting tables with OpenAI)
            for batch_start in range(0, total_files, self.batch_size):
                batch_end = min(batch_start + self.batch_size, total_files)
                batch = pdf_files[batch_start:batch_end]
                
                logger.info(f"Processing batch {batch_start//self.batch_size + 1}: {len(batch)} files (single process mode)")
                
                for pdf in batch:
                    file_path = os.path.join(pdf_dir, pdf)
                    try:
                        logger.info(f"Processing {pdf}")
                        sha1, sections = self.process_pdf(file_path)
                        results[pdf] = {
                            "sha1": sha1,
                            "sections": sections,
                            "page_count": len(set(section.page for section in sections))
                        }
                        logger.info(f"Processed {pdf}: {len(sections)} sections, SHA1: {sha1}")
                    except Exception as e:
                        logger.error(f"Error processing {pdf}: {e}")
                
                # Clean up after batch
                gc.collect()
        
        return results


def _looks_like_table(text: str) -> bool:
    """Helper function to determine if text looks like a table"""
    # Check for common table characteristics
    lines = text.strip().split('\n')
    if len(lines) < 3:  # Too short to be a table
        return False
        
    # Check for column alignment
    whitespace_patterns = []
    for line in lines[:5]:  # Check first few lines
        # Record positions of whitespace clusters
        positions = [i for i, char in enumerate(line) if char.isspace()]
        
        # Find clusters of whitespace (potential column separators)
        clusters = []
        if positions:
            cluster_start = positions[0]
            for i in range(1, len(positions)):
                if positions[i] > positions[i-1] + 1:  # Gap in sequence
                    if positions[i-1] - cluster_start > 1:  # Cluster must be at least 2 spaces
                        clusters.append((cluster_start, positions[i-1]))
                    cluster_start = positions[i]
            
            # Add last cluster
            if positions[-1] - cluster_start > 1:
                clusters.append((cluster_start, positions[-1]))
                
        whitespace_patterns.append(clusters)
    
    # Check for consistent column separators
    if len(whitespace_patterns) > 1:
        # Count similar patterns
        similar_patterns = 0
        for i in range(1, len(whitespace_patterns)):
            if len(whitespace_patterns[i]) == len(whitespace_patterns[0]):
                similar_patterns += 1
        
        # If most patterns are similar, likely a table
        if similar_patterns / len(whitespace_patterns) > 0.5:
            return True
    
    # Check for other table indicators
    dashes = sum(1 for line in lines if re.search(r'^[- |+]+$', line))
    if dashes > 0:
        return True
        
    # Check for number density
    number_lines = sum(1 for line in lines if re.search(r'\d+', line))
    if number_lines / len(lines) > 0.7:
        return True
    
    return False


if __name__ == "__main__":
    # Simple test if run as script
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pdf_processor.py <pdf_file_or_directory>")
        sys.exit(1)
    
    path = sys.argv[1]
    processor = PDFProcessor(extract_tables=True)
    
    if os.path.isdir(path):
        results = processor.process_directory(path, max_workers=2)
        print(f"Processed {len(results)} PDFs")
    else:
        sha1, sections = processor.process_pdf(path)
        print(f"SHA1: {sha1}")
        print(f"Found {len(sections)} sections")
        
        # Print section titles
        for i, section in enumerate(sections):
            print(f"{i+1}. {section.title} (Page {section.page+1}, {'Table' if section.is_table else 'Text'})")