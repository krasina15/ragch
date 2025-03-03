"""
Vector Index Builder module for Enterprise RAG Challenge
Creates memory-efficient vector stores for document retrieval
By Krasina15. Made with crooked kind hands
"""
import os
import gc
import time
import pickle
import logging
from typing import List, Dict, Any, Optional

# LangChain imports
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorIndexBuilder:
    """Memory-efficient vector index builder"""
    
    def __init__(self, embedding_model="text-embedding-3-small", batch_size=10):
        """
        Initialize vector index builder
        
        Args:
            embedding_model: Name of embedding model to use
            batch_size: Batch size for processing
        """
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        
        # Initialize embeddings with batch processing for efficiency
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            chunk_size=batch_size  # Process embeddings in batches
        )
    
    def build_index(self, chunks: List[Document], save_path: Optional[str] = None) -> FAISS:
        """
        Build a FAISS index incrementally from chunks
        
        Args:
            chunks: List of Document objects to index
            save_path: Optional path to save the index
            
        Returns:
            FAISS vector store
        """
        if not chunks:
            raise ValueError("No chunks provided for indexing")
        
        total_chunks = len(chunks)
        logger.info(f"Building vector index with {total_chunks} chunks in batches of {self.batch_size}")
        
        start_time = time.time()
        
        # Start with first batch
        first_batch = chunks[:self.batch_size]
        vector_store = FAISS.from_documents(first_batch, self.embeddings)
        
        # Process remaining chunks in batches
        for i in range(self.batch_size, total_chunks, self.batch_size):
            batch_end = min(i + self.batch_size, total_chunks)
            current_batch = chunks[i:batch_end]
            
            batch_num = i // self.batch_size + 1
            total_batches = (total_chunks + self.batch_size - 1) // self.batch_size
            logger.info(f"Adding batch {batch_num}/{total_batches} ({len(current_batch)} chunks)")
            
            # Add batch to vector store
            vector_store.add_documents(current_batch)
            
            # Log progress periodically
            if batch_num % 5 == 0 or batch_num == total_batches:
                elapsed = time.time() - start_time
                chunks_per_sec = i / elapsed if elapsed > 0 else 0
                logger.info(f"Progress: {i}/{total_chunks} chunks, "
                           f"{elapsed:.1f} seconds ({chunks_per_sec:.1f} chunks/sec)")
            
            # Clean up batch references to save memory
            del current_batch
            if i % (self.batch_size * 5) == 0:  # Garbage collect periodically
                gc.collect()
        
        # Final stats
        elapsed = time.time() - start_time
        logger.info(f"Completed indexing {total_chunks} chunks in {elapsed:.1f} seconds "
                   f"({total_chunks/elapsed:.1f} chunks/sec)")
        
        # Save index if path provided
        if save_path:
            self._save_index(vector_store, save_path, total_chunks)
        
        return vector_store
    
    def _save_index(self, vector_store: FAISS, save_path: str, total_chunks: int) -> None:
        """
        Save the vector store and metadata
        
        Args:
            vector_store: FAISS vector store to save
            save_path: Path to save the index
            total_chunks: Total number of chunks indexed
        """
        try:
            # Create directory if it doesn't exist
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            
            # Save the vector store
            vector_store.save_local(save_path)
            
            # Save metadata to help with loading
            # Handle path with or without extension for metadata
            base_path = os.path.splitext(save_path)[0]
            metadata_path = f"{base_path}_metadata.pkl"
            
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'embedding_model': self.embedding_model,
                    'chunk_size': self.batch_size,
                    'total_chunks': total_chunks,
                    'created_at': time.time()
                }, f)
                
            logger.info(f"Vector index saved to {save_path}")
            logger.info(f"Metadata saved to {metadata_path}")
            
        except Exception as e:
            logger.error(f"Error saving vector index: {e}")
            raise
    
    def load_index(self, load_path: str) -> Optional[FAISS]:
        """
        Load a saved FAISS index
        
        Args:
            load_path: Path to load the index from
            
        Returns:
            FAISS vector store or None if loading fails
        """
        try:
            # Check if the directory and index files exist
            if not os.path.exists(load_path):
                logger.error(f"Vector index path does not exist: {load_path}")
                return None
                
            # Get base path for metadata (handle paths with or without extension)
            base_path = os.path.splitext(load_path)[0]
            metadata_path = f"{base_path}_metadata.pkl"
            
            # Load metadata to verify compatibility, here I am almost confused how it works
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                # Check if the embedding model matches
                if metadata.get('embedding_model') != self.embedding_model:
                    logger.warning(f"Embedding model mismatch: {metadata.get('embedding_model')} "
                                  f"vs {self.embedding_model}")
                    logger.warning("This may cause dimension mismatch errors. Consider rebuilding the index.")
                    # We'll still try to load, but have warned the user
            else:
                logger.warning(f"Metadata file not found at {metadata_path}")
            
            # Load the vector store
            logger.info(f"Loading vector index from {load_path}")
            vector_store = FAISS.load_local(load_path, self.embeddings, allow_dangerous_deserialization=True)
            
            # Log info about the loaded index
            if metadata:
                logger.info(f"Loaded index with {metadata.get('total_chunks', 'unknown')} chunks, "
                           f"created with {metadata.get('embedding_model', 'unknown')}")
            
            return vector_store
            
        except Exception as e:
            logger.error(f"Error loading vector index: {e}")
            raise
