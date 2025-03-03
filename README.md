# Enterprise RAG Challenge

A robust, memory-optimized Retrieval-Augmented Generation (RAG) system designed to process corporate annual reports and answer specific questions with precise references. 

This code is my personal attempt in the [Enterprise Rag Challange](https://rag.timetoact.at/).


## Overview

This system can efficiently process PDF annual reports, extract text and tables, create embeddings, and answer questions with accurate references to source documents. Key features include memory optimization for handling large documents, table extraction, and question-specific response formatting.

## Architecture

The system consists of several modular components:

- **PDF Processor**: Extracts text and tables from PDF documents
- **Chunking Strategy**: Splits text into semantically meaningful chunks optimized for financial documents
- **Vector Index Builder**: Creates memory-efficient vector embeddings for document retrieval
- **RAG Retriever**: Retrieves and re-ranks relevant document chunks based on the question
- **Answer Generator**: Produces precise answers from the retrieved context with validation
- **PDF Tracker**: Avoids redundant processing of previously analyzed documents


## Key Features


### Vector Embeddings & Similarity Search

- **OpenAI Embeddings API**: We use text-embedding-3-small to convert text chunks into high-dimensional vectors, capturing semantic meaning in a 1536-dimensional space.
- **FAISS** (Facebook AI Similarity Search): Provides efficient vector storage and similarity search capabilities with optimizations for nearest-neighbor lookups. We use batch processing to handle large document collections efficiently.
- **Incremental Indexing**: Our vector store supports adding new documents without rebuilding the entire index, enabling efficient updates to the knowledge base.

### Document Processing

- **PyMuPDF** (fitz): Extracts text and structural information from PDF documents while preserving document hierarchy.
- **Table Extraction**: Uses OpenAI's Vision API (GPT-4o-mini) to identify and extract tabular data that traditional text extraction would misinterpret.
- **SHA1 Tracking**: Uses cryptographic hashing to identify which documents have already been processed, avoiding redundant work.

### Chunking Strategies

- **RecursiveCharacterTextSplitter**: A specialized implementation that respects semantic boundaries in financial documents.
- **Financial Document-Aware Splitting**: Custom separators and patterns optimized for financial statements, including section headers, tables, and numerical data.
- **Semantic Preservation**: Special handling to keep tables intact while breaking down text content into manageable chunks.
- **Overlap Management**: Configurable chunk overlap to maintain context between adjacent chunks.

### Retrieval Enhancement

- **Financial Term Boosting**: Domain-specific weighting of financial terminology in queries and document scoring.
- **Query Expansion**: Enhances user queries with relevant financial terms and entity recognition.
- **Question-Type Aware Retrieval**: Different retrieval strategies based on whether questions seek numerical data, names, or boolean answers.
- **Multi-stage Ranking**: Initially retrieves a larger candidate set, then applies custom re-ranking based on multiple relevance signals.

### Answer Generation

- **OpenAI GPT Models**: Uses gpt-4o (configurable) for answer generation from retrieved context.
- **Question-Specific Prompting**: Specialized prompts for different question types (numerical, boolean, name, names).
- **Structured Output Format**: Generates answers in a consistent, submission-ready JSON format.
- **Answer Validation**: Verifies answers against the retrieved context to prevent hallucination.

### Memory Optimization

- **Batch Processing**: Processes documents and generates embeddings in batches to control memory usage.
- **Strategic Garbage Collection**: Implements garbage collection at key points in the pipeline to free memory.
- **Reference Management**: Efficiently handles document references to minimize duplication.

### LangChain Integration

- **Modular Components**: Leverages LangChain's abstractions for embeddings, vector stores, document loading, and text splitting.
- **Retriever Implementation**: Extends LangChain's retriever interfaces with domain-specific enhancements.
- **Chain-of-Thought Prompting**: Implements structured prompting techniques to guide the LLM's reasoning process.

### Domain-Specific Optimizations

- **Financial Entity Recognition**: Enhanced recognition of company names, executive roles, and financial metrics.
- **Table Structure Preservation**: Special handling for financial tables, which often contain critical numerical data.
- **Currency and Unit Normalization**: Standardized handling of monetary values and financial metrics.
- **Confidence Scoring**: Each answer includes an internal confidence score based on how explicitly it appears in the source documents.

### Pipeline Architecture

- **Component Isolation**: Each step in the RAG pipeline is implemented as a separate module with well-defined interfaces.
- **Error Resilience**: Comprehensive error handling to ensure the pipeline continues even if individual documents fail to process.
- **Configurable Parameters**: Extensive command-line options to tune the system for different document collections and question types.
- **Parallel Processing**: Multi-worker processing for CPU-bound operations to improve throughput.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd enterprise-rag-challenge
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
   
  but I hope `pip install pymupdf openai langchain langchain_openai langchain_community faiss-cpu camelot-py[cv] opencv-python` should be enough.

3. Set up your OpenAI API key:
   ```bash
   export OPENAI_API_KEY=your-api-key-here
   ```
   
   Oh yeah, there was secondone implementation for Claude Sonnet, but something went wrong :(
   

## Usage

### Processing PDFs and Building Vector Store

To process PDFs and build a vector store:

```bash
python main.py --pdf_dir=pdfs --max_workers=2 --process_only --vector_store_path=./vector_store
```

This command will:
1. Process all PDFs in the `pdfs` directory
2. Extract text and tables (if enabled)
3. Create optimized chunks
4. Build and save a vector index to `./vector_store`

### Answering Questions

To process questions and generate answers:

```bash
python main.py --pdf_dir=pdfs --vector_store_path=./vector_store --questions=questions.json --team_email=your@email.com --submission_name="Your Team Name" --output=answers.json
```

This will:
1. Load the vector store (or create it if it doesn't exist)
2. Answer questions from the JSON file
3. Save the answers in the required submission format

### Incremental Updates

The system automatically tracks processed PDFs to avoid redundant processing:

```bash
python main.py --pdf_dir=pdfs --vector_store_path=./vector_store --process_only
```

When run multiple times, only new or modified PDFs will be processed.

## Command-Line Arguments

| Argument | Description |
|----------|-------------|
| `--pdf_dir` | Directory containing PDF files |
| `--process_only` | Only process PDFs and build vector store |
| `--vector_store_path` | Path to save/load vector store |
| `--questions` | JSON file containing questions |
| `--team_email` | Team email for submission |
| `--submission_name` | Name for the submission |
| `--output` | Output file for submission JSON |
| `--extract_tables` | Extract tables from PDFs (default: True) |
| `--skip_tables` | Skip table extraction to save memory |
| `--max_workers` | Maximum number of parallel workers (default: 2) |
| `--batch_size` | Batch size for processing (default: 10) |
| `--embedding_model` | Embedding model to use (default: text-embedding-3-small) |
| `--llm_model` | LLM model for answer generation (default: gpt-4o) |
| `--chunk_size` | Chunk size for text splitting (default: 800) |
| `--chunk_overlap` | Chunk overlap for text splitting (default: 100) |
| `--top_k` | Number of chunks to retrieve per question (default: 5) |
| `--skip_processed` | Skip processing PDFs that are already in the vector store (default: True) |
| `--force_reprocess` | Force reprocessing of all PDFs even if already processed |
| `--unicorn_amplifier` | Maybe I didn't have time to finish something or its already broken |


## File Structure

- `main.py`: Main entry point and pipeline orchestration
- `pdf_processor.py`: PDF text and table extraction 
- `chunking_strategy.py`: Text chunking optimized for financial documents
- `vector_index.py`: Vector embedding creation and storage
- `retriever.py`: Enhanced retrieval with financial domain awareness
- `answer_generator.py`: Answer generation and validation
- `pdf_tracking.py`: Tracks processed PDFs to avoid duplicate work

## Question Format

The questions should be provided in a JSON file with the following structure:

```json
[
  {
    "text": "What was the revenue of Company X in 2023?",
    "kind": "number"
  },
  {
    "text": "Who is the CEO of Company Y?",
    "kind": "names"
  }
]
```

Question kinds include:
- `number`: Numerical questions
- `names`: Multiple names questions
- `boolean`: Yes/no questions

## Output Format

Answers are generated in the following format:

```json
{
  "team_email": "your@email.com",
  "submission_name": "Your Team Name",
  "answers": [
    {
      "question_text": "What was the revenue of Company X in 2023?",
      "kind": "number",
      "value": 10500000,
      "references": [
        {
          "pdf_sha1": "abc123...",
          "page_index": 42
        }
      ]
    }
  ]
}
```


## Dependencies

- `langchain`, `langchain_openai`, `langchain_community`: Foundation for RAG pipeline
- `pymupdf`: PDF text extraction
- `openai`: LLM-based table extraction and answer generation
- `faiss-cpu`: Vector storage and similarity search

## Troubleshooting

- **FAISS Loading Error**: If you encounter a pickle deserialization warning, the system is designed to handle this securely.
- **Memory Issues**: For very large PDFs, use `--skip_tables` to reduce memory usage.
- **API Rate Limits**: When processing many documents with table extraction, you might hit OpenAI API rate limits. Add delays or process in smaller batches.

## Tips for Best Results

1. **Optimize Chunk Size**: 800-1000 tokens often works well for financial documents
2. **Adjust Top K**: Increase for complex questions requiring more context
3. **Use Process_Only**: Run with `--process_only` first to build the vector store, then run queries
4. **Track Memory Usage**: Monitor memory consumption and adjust batch size as needed

## Disclaimer

Use at your own risk. Warning - it is possible to get results from the code (but that's not for sure, lol)

Good Luck!
