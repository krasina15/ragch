"""
RAG Retriever module for Enterprise RAG Challenge
Enhanced retrieval with re-ranking for financial documents
By Krasina15. Made with crooked kind hands
"""
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

# LangChain imports
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGRetriever:
    """Enhanced retrieval with financial-aware re-ranking"""
    
    def __init__(self, vector_store, top_k=5):
        """
        Initialize retriever
        
        Args:
            vector_store: Vector store to retrieve from
            top_k: Number of documents to retrieve
        """
        self.vector_store = vector_store
        self.top_k = top_k
        
        # Financial terms to boost in retrieval
        self.financial_terms = {
            # Financial statements
            'revenue': 1.2, 'income': 1.2, 'profit': 1.2, 'loss': 1.2, 'margin': 1.2,
            'asset': 1.1, 'liability': 1.1, 'equity': 1.1, 'debt': 1.1, 'cash': 1.1,
            'earnings': 1.2, 'ebitda': 1.3, 'eps': 1.3, 'dividend': 1.2,
            'balance sheet': 1.3, 'income statement': 1.3, 'cash flow': 1.3,
            
            # Personnel
            'ceo': 1.3, 'cfo': 1.3, 'board': 1.2, 'director': 1.2, 'executive': 1.2,
            'chairman': 1.3, 'president': 1.2, 'officer': 1.1, 'management': 1.1,
            
            # Business metrics
            'growth': 1.1, 'increase': 1.0, 'decrease': 1.0, 'acquisition': 1.2,
            'merger': 1.2, 'subsidiary': 1.1, 'segment': 1.1, 'market share': 1.2,
            'forecast': 1.1, 'outlook': 1.1, 'guidance': 1.2, 'strategy': 1.1,
            
            # Time periods
            'quarter': 1.1, 'annual': 1.1, 'fiscal year': 1.2, 'year-end': 1.1,
            'q1': 1.2, 'q2': 1.2, 'q3': 1.2, 'q4': 1.2, 'fy': 1.2
        }
    
    def retrieve(self, question: str, question_kind: str) -> List[Document]:
        """
        Retrieve and rerank documents for a question
        
        Args:
            question: Question text
            question_kind: Type of question (number, name, boolean, names)
            
        Returns:
            List of relevant documents
        """
        # 1. Enhance query with financial terms
        enhanced_query = self._enhance_query(question, question_kind)
        logger.info(f"Enhanced query: {enhanced_query}")
        
        # 2. Retrieve more documents than needed for reranking
        retriever = self.vector_store.as_retriever(search_kwargs={"k": self.top_k * 2})
        # Using invoke() instead of deprecated get_relevant_documents()
        candidates = retriever.invoke(enhanced_query)
        
        # 3. Apply financial-aware reranking
        reranked_docs = self._rerank_documents(candidates, question, question_kind)
        
        # 4. Return top k after reranking
        return reranked_docs[:self.top_k]

    def _enhance_query(self, question: str, question_kind: str) -> str:
        """
        Enhance query with relevant terms based on question type
        
        Args:
            question: Original question
            question_kind: Type of question
            
        Returns:
            Enhanced query string
        """
        # Extract quoted entities (e.g. company names)
        entities = re.findall(r'"([^"]*)"', question)
        entities_str = " ".join(entities)
        
        # Extract key financial terms from question
        financial_terms_in_question = [
            term for term in self.financial_terms 
            if term.lower() in question.lower()
        ]
        
        # Add type-specific search terms
        type_hints = ""
        if question_kind == "number":
            type_hints = "financial figures statistics amounts numbers data consolidated statement"
        elif question_kind == "boolean":
            type_hints = "confirm verify reported disclosed mentioned"
        elif question_kind in ["name", "names"]:
            type_hints = "names people positions executives titles leadership management"
        
        # Industry-specific metric mapping
        industry_metrics = {
            "cloud storage capacity": "data center storage TB gigabytes capacity cloud",
            "patents": "intellectual property patent R&D innovation proprietary",
            "employees": "headcount staff workforce personnel employees",
            "executive compensation": "salary bonus compensation package executive officer",
            "total assets": "balance sheet assets book value total assets"
            # Add more industry-specific terms based on question analysis
        }
        
        # Extract industry metrics from question
        added_terms = []
        for metric, terms in industry_metrics.items():
            if metric.lower() in question.lower():
                added_terms.append(terms)
        
        # Combine all terms, prioritizing entities and financial terms
        enhanced_query = f"{question} {entities_str} {' '.join(financial_terms_in_question)} {' '.join(added_terms)} {type_hints}"
        
        return enhanced_query
    
    def _rerank_documents(self, docs: List[Document], question: str, question_kind: str) -> List[Document]:
        """
        Rerank documents based on relevance to question
        
        Args:
            docs: Candidate documents
            question: Original question
            question_kind: Type of question
            
        Returns:
            Reranked documents
        """
        # Extract entities and key terms for matching
        question_lower = question.lower()
        entities = re.findall(r'"([^"]*)"', question)
        
        # Score each document
        scored_docs: List[Tuple[Document, float]] = []
        
        for doc in docs:
            # Start with base score
            score = 1.0
            content_lower = doc.page_content.lower()
            
            # 1. Score based on entity mentions (companies, people, etc.)
            for entity in entities:
                entity_lower = entity.lower()
                if entity_lower in content_lower:
                    # Higher boost for exact matches
                    if re.search(r'\b' + re.escape(entity_lower) + r'\b', content_lower):
                        score += 2.0
                    else:
                        score += 1.0
            
            # 2. Score based on financial terms
            for term, boost in self.financial_terms.items():
                if term.lower() in content_lower and term.lower() in question_lower:
                    score += boost
            
            # 3. Type-specific scoring
            if question_kind == "number":
                # For number questions, prioritize tables and numeric content
                if doc.metadata.get("is_table", False):
                    score += 2.0
                
                # Count numbers in the document
                num_count = len(re.findall(r'\b\d+(?:,\d+)*(?:\.\d+)?(?:\s*%|\s*million|\s*billion)?\b', content_lower))
                score += min(num_count * 0.1, 1.0)  # Cap the boost
                
                # Check for currency symbols
                if re.search(r'[$€£¥]', doc.page_content):
                    score += 0.5
                
            elif question_kind in ["name", "names"]:
                # For name questions, look for sections about people
                title = doc.metadata.get("title", "").lower()
                
                # Boost executive/management sections
                exec_terms = ["management", "executive", "leadership", "board", "director", "officer"]
                if any(term in title for term in exec_terms):
                    score += 1.5
                elif any(term in content_lower for term in exec_terms):
                    score += 0.8
                
                # Look for name patterns (e.g., "John Smith")
                name_patterns = len(re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', doc.page_content))
                score += min(name_patterns * 0.1, 1.0)
                
            elif question_kind == "boolean":
                # For boolean questions, look for affirmative/negative statements
                affirm_terms = ["yes", "confirmed", "does", "is", "has", "did", "will"]
                negate_terms = ["no", "not", "never", "neither", "nor", "doesn't", "isn't", "won't"]
                
                # Check for presence of yes/no language
                if any(term in content_lower for term in affirm_terms + negate_terms):
                    score += 0.5
            
            # 4. Page-based scoring: first pages often contain key info
            page_num = doc.metadata.get("page", 0)
            if page_num < 5:  # Boost first few pages
                score += max(0, (5 - page_num) * 0.1)
            
            # Add scored document
            scored_docs.append((doc, score))
        
        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Debug logging for top scores
        for i, (doc, score) in enumerate(scored_docs[:5]):
            page = doc.metadata.get("page", 0)
            is_table = doc.metadata.get("is_table", False)
            logger.debug(f"Rank {i+1}: Score {score:.2f}, Page {page+1}, {'TABLE' if is_table else 'TEXT'}")
        
        # Return documents only
        return [doc for doc, _ in scored_docs]