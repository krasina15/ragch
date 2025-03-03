"""
PDF Processing Tracker module for Enterprise RAG Challenge
Tracks processed PDFs to avoid duplicate processing
By Krasina15. Made with crooked kind hands
"""
import os
import json
import logging
from typing import Dict, List, Set, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFTracker:
    """
    Tracks processed PDFs to avoid duplicate processing
    Uses SHA1 hash to identify PDFs and tracks their processing status
    """
    
    def __init__(self, tracking_file_path: str):
        """
        Initialize PDF tracker
        
        Args:
            tracking_file_path: Path to the tracking JSON file
        """
        self.tracking_file_path = tracking_file_path
        self.tracking_data = self._load_tracking_data()
    
    def _load_tracking_data(self) -> Dict:
        """
        Load tracking data from file
        
        Returns:
            Dictionary with tracking data
        """
        if os.path.exists(self.tracking_file_path):
            try:
                with open(self.tracking_file_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error loading tracking data: {e}. Creating new tracking file.")
                return {"processed_pdfs": {}, "version": "1.0"}
        else:
            return {"processed_pdfs": {}, "version": "1.0"}
    
    def save_tracking_data(self):
        """Save tracking data to file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.tracking_file_path), exist_ok=True)
            
            with open(self.tracking_file_path, 'w') as f:
                json.dump(self.tracking_data, f, indent=2)
                
            logger.info(f"Tracking data saved to {self.tracking_file_path}")
        except IOError as e:
            logger.error(f"Error saving tracking data: {e}")
    
    def is_pdf_processed(self, filename: str, sha1: str) -> bool:
        """
        Check if a PDF has already been processed
        
        Args:
            filename: PDF filename
            sha1: SHA1 hash of the PDF
            
        Returns:
            Boolean indicating if the PDF has been processed
        """
        processed_pdfs = self.tracking_data.get("processed_pdfs", {})
        
        # Check if the filename exists and the SHA1 matches
        if filename in processed_pdfs and processed_pdfs[filename]["sha1"] == sha1:
            return True
        
        return False
    
    def mark_pdf_as_processed(self, filename: str, sha1: str, chunk_count: int):
        """
        Mark a PDF as processed
        
        Args:
            filename: PDF filename
            sha1: SHA1 hash of the PDF
            chunk_count: Number of chunks created from this PDF
        """
        if "processed_pdfs" not in self.tracking_data:
            self.tracking_data["processed_pdfs"] = {}
            
        self.tracking_data["processed_pdfs"][filename] = {
            "sha1": sha1,
            "chunk_count": chunk_count,
            "timestamp": str(os.path.getmtime(self.tracking_file_path)) if os.path.exists(self.tracking_file_path) else None
        }
        
        # Save after each update
        self.save_tracking_data()
    
    def get_processed_files(self) -> List[str]:
        """
        Get list of processed PDF filenames
        
        Returns:
            List of processed PDF filenames
        """
        return list(self.tracking_data.get("processed_pdfs", {}).keys())
    
    def get_unprocessed_files(self, all_files: List[str], calculate_sha1_fn) -> List[str]:
        """
        Get list of unprocessed PDF filenames
        
        Args:
            all_files: List of all PDF filenames
            calculate_sha1_fn: Function to calculate SHA1 hash for a file
            
        Returns:
            List of unprocessed PDF filenames
        """
        processed_pdfs = self.tracking_data.get("processed_pdfs", {})
        unprocessed_files = []
        processed_count = 0
        changed_count = 0
        
        for filename in all_files:
            if filename not in processed_pdfs:
                unprocessed_files.append(filename)
                logger.info(f"PDF {filename} has not been processed before")
            else:
                # Check if the file has changed (different SHA1)
                current_sha1 = calculate_sha1_fn(filename)
                if processed_pdfs[filename]["sha1"] != current_sha1:
                    unprocessed_files.append(filename)
                    changed_count += 1
                    logger.info(f"PDF {filename} has changed since last processing (SHA1 different)")
                else:
                    processed_count += 1
                    logger.debug(f"PDF {filename} already processed with matching SHA1")
        
        if processed_count > 0:
            logger.info(f"Found {processed_count} already processed PDFs with matching SHA1")
        if changed_count > 0:
            logger.info(f"Found {changed_count} previously processed PDFs with changed content")
        
        return unprocessed_files
    
    def get_stats(self) -> Dict:
        """
        Get statistics about processed PDFs
        
        Returns:
            Dictionary with statistics
        """
        processed_pdfs = self.tracking_data.get("processed_pdfs", {})
        total_chunks = sum(data.get("chunk_count", 0) for data in processed_pdfs.values())
        
        return {
            "total_files": len(processed_pdfs),
            "total_chunks": total_chunks,
            "last_updated": max([data.get("timestamp") for data in processed_pdfs.values()]) if processed_pdfs else None
        }