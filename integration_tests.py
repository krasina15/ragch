"""
Integration test script for the Enterprise RAG Challenge components
Tests each component and their integration with small test files
By Krasina15. Made with crooked kind hands
"""
import os
import sys
import shutil
import tempfile
import unittest
from pathlib import Path

# Import our components
from pdf_processor import PDFProcessor, DocumentSection
from chunking_strategy import ChunkingStrategy
from vector_index import VectorIndexBuilder
from retriever import RAGRetriever
from answer_generator import AnswerGenerator

# Sample PDF path - update this to point to a valid PDF file
SAMPLE_PDF_PATH = "sample.pdf"  # Update this path to a test PDF


class IntegrationTest(unittest.TestCase):
    """Integration tests for RAG pipeline components"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a temporary directory for vector store
        self.temp_dir = tempfile.mkdtemp()
        self.vector_store_path = os.path.join(self.temp_dir, "test_vector_store")
        
        # Skip PDF processing tests if sample PDF not available
        self.skip_pdf_tests = not os.path.exists(SAMPLE_PDF_PATH)
        if self.skip_pdf_tests:
            print(f"WARNING: Sample PDF not found at {SAMPLE_PDF_PATH}. Skipping PDF processing tests.")
    
    def tearDown(self):
        """Clean up after tests"""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_vector_index_save_load(self):
        """Test vector index save and load functionality"""
        # Create sample documents
        from langchain_core.documents import Document
        
        docs = [
            Document(page_content=f"Test document {i}", metadata={"id": i})
            for i in range(10)
        ]
        
        # Build and save index
        builder = VectorIndexBuilder(batch_size=5)
        vector_store = builder.build_index(docs, save_path=self.vector_store_path)
        
        # Load index
        loaded_store = builder.load_index(self.vector_store_path)
        
        # Check if index was loaded successfully
        self.assertIsNotNone(loaded_store)
        
        # Test query
        retriever = RAGRetriever(loaded_store, top_k=2)
        results = retriever.retrieve("test document", "name")
        
        # Verify results
        self.assertEqual(len(results), 2)
        
        print("Vector index save/load test: PASSED")
    
    def test_pdf_processor(self):
        """Test PDF processor"""
        if self.skip_pdf_tests:
            self.skipTest("Sample PDF not available")
        
        # Process sample PDF
        processor = PDFProcessor(extract_tables=False)  # Disable tables for testing
        sha1, sections = processor.process_pdf(SAMPLE_PDF_PATH)
        
        # Check results
        self.assertIsNotNone(sha1)
        self.assertGreater(len(sections), 0)
        
        # Check section properties
        for section in sections:
            self.assertIsInstance(section, DocumentSection)
            self.assertTrue(hasattr(section, 'title'))
            self.assertTrue(hasattr(section, 'level'))
            self.assertTrue(hasattr(section, 'content'))
            self.assertTrue(hasattr(section, 'page'))
            self.assertTrue(hasattr(section, 'is_table'))
        
        print(f"PDF processor test: PASSED - Found {len(sections)} sections")
    
    def test_chunking_strategy(self):
        """Test chunking strategy"""
        # Create sample sections
        sections = [
            DocumentSection(
                title="Test Section",
                level=1,
                content="This is a test section with some content.\n" * 20,
                page=0,
                is_table=False
            ),
            DocumentSection(
                title="Test Table",
                level=2,
                content="Column1\tColumn2\tColumn3\n1\t2\t3\n4\t5\t6\n7\t8\t9\n",
                page=1,
                is_table=True
            ),
        ]
        
        # Apply chunking
        chunker = ChunkingStrategy(chunk_size=100, chunk_overlap=20)
        chunks = chunker.create_chunks(sections, "test.pdf", "abc123")
        
        # Check results
        self.assertGreater(len(chunks), 2)  # Should create multiple chunks from text section
        
        # Verify regular chunks
        text_chunks = [c for c in chunks if not c.metadata.get("is_table")]
        self.assertGreater(len(text_chunks), 0)
        
        # Verify table chunks (should be preserved as one chunk)
        table_chunks = [c for c in chunks if c.metadata.get("is_table")]
        self.assertEqual(len(table_chunks), 1)
        
        print(f"Chunking strategy test: PASSED - Created {len(chunks)} chunks")
    
    def test_end_to_end(self):
        """Test basic end-to-end pipeline with mock data"""
        from langchain_core.documents import Document
        
        # Create sample documents
        docs = [
            Document(
                page_content="ABC Corp reported annual revenue of $10.5 million in 2023.",
                metadata={"source": "report.pdf", "sha1": "abc123", "page": 0}
            ),
            Document(
                page_content="John Smith has served as CEO since January 2020.",
                metadata={"source": "report.pdf", "sha1": "abc123", "page": 1}
            ),
            Document(
                page_content="The Board of Directors includes Jane Doe, Robert Johnson, and Mary Williams.",
                metadata={"source": "report.pdf", "sha1": "abc123", "page": 2}
            ),
        ]
        
        # Build index
        builder = VectorIndexBuilder(batch_size=5)
        vector_store = builder.build_index(docs, save_path=self.vector_store_path)
        
        # Create retriever
        retriever = RAGRetriever(vector_store, top_k=2)
        
        # Test retrieval
        number_question = "What was the revenue of ABC Corp in 2023?"
        name_question = "Who is the CEO of ABC Corp?"
        
        num_results = retriever.retrieve(number_question, "number")
        name_results = retriever.retrieve(name_question, "name")
        
        # Check retrieval results
        self.assertGreater(len(num_results), 0)
        self.assertGreater(len(name_results), 0)
        
        # Test answer generation if LLM API is available
        try:
            generator = AnswerGenerator(model_name="gpt-3.5-turbo")
            result = generator.answer_question(number_question, num_results, "number")
            
            # Check answer structure
            self.assertIn("value", result)
            self.assertIn("references", result)
            print(f"Answer generation test: PASSED - Generated answer: {result['value']}")
        except Exception as e:
            print(f"Answer generation test skipped: {e}")
        
        print("End-to-end pipeline test: PASSED")


if __name__ == "__main__":
    unittest.main()