"""
Answer Generator module for Enterprise RAG Challenge.
Generates precise answers from retrieved context with validation.
By Krasina15. Made with crooked kind hands.
"""
import re
import json
import logging
from typing import List, Dict, Any, Optional, Union

# LangChain imports
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Generate and validate answers for financial questions"""
    
    def __init__(self, model_name="gpt-4o", temperature=0):
        """
        Initialize answer generator
        
        Args:
            model_name: Name of LLM model to use
            temperature: Temperature for generation (0 for deterministic)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    
    def answer_question(self, question: str, contexts: List[Document], question_kind: str) -> Dict[str, Any]:
        """
        Generate an answer with validation
        
        Args:
            question: Question text
            contexts: Retrieved context documents
            question_kind: Type of question
            
        Returns:
            Dictionary with answer and metadata
        """
        # Format context for the LLM
        formatted_context = self._format_context(contexts)
        
        # Create prompt based on question type
        prompt = self._create_prompt(question, formatted_context, question_kind)
        
        # Get LLM response
        try:
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse structured response
            answer_data = self._parse_response(response_text)
            
            # Extract and validate the answer
            raw_answer = answer_data.get("answer", "")
            confidence = float(answer_data.get("confidence", 0.5))
            
            # Format answer based on question kind
            formatted_answer = self._format_answer(raw_answer, question_kind)
            
            # Extract references from source docs
            references = self._extract_references(contexts, formatted_answer, question_kind)
            
            # Perform answer validation
            if formatted_answer != "N/A":
                validation_score = self._validate_answer(formatted_answer, formatted_context, question_kind)
                
                # If answer can't be validated in context, force N/A
                if validation_score < 0.2 and confidence < 0.7:
                    logger.warning(f"Answer '{formatted_answer}' not validated in context. Forcing N/A.")
                    formatted_answer = "N/A"
                    references = []
            
            # Build result
            return {
                "value": formatted_answer,
                "references": references,
                "confidence": confidence,
                "question_text": question,
                "kind": question_kind,
                "validation_notes": answer_data.get("validation", "")
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            # Fallback to N/A on error
            return {
                "value": "N/A" if question_kind != "names" else [],
                "references": [],
                "confidence": 0.0,
                "question_text": question,
                "kind": question_kind
            }
    
    def _format_context(self, docs: List[Document]) -> str:
        """
        Format retrieved documents into context string
        
        Args:
            docs: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown")
            sha1 = doc.metadata.get("sha1", "Unknown")
            page = doc.metadata.get("page", 0)
            is_table = doc.metadata.get("is_table", False)
            title = doc.metadata.get("title", "Unknown section")
            
            # Format with clear section markers
            formatted_section = (
                f"[DOCUMENT {i+1}]\n"
                f"SOURCE: {source}\n"
                f"SHA1: {sha1}\n"
                f"PAGE: {page+1}\n"  # Convert to 1-indexed for LLM
                f"SECTION: {title}\n"
                f"FORMAT: {'TABLE' if is_table else 'TEXT'}\n\n"
                f"CONTENT:\n{doc.page_content}\n\n"
                f"{'-' * 50}\n"
            )
            
            context_parts.append(formatted_section)
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, question: str, context: str, question_kind: str) -> str:
        """
        Create a prompt for the specific question type
        
        Args:
            question: Question text
            context: Formatted context
            question_kind: Type of question
            
        Returns:
            Prompt for the LLM
        """
        # Common instructions for all question types
        common_instructions = f"""You are a financial analyst extracting precise information from annual reports.
Answer the question based ONLY on the context provided. Do not use prior knowledge.

CONTEXT:
{context}

QUESTION:
{question}

IMPORTANT ANTI-HALLUCINATION INSTRUCTIONS:
1. ONLY answer based on the information provided in the context above.
2. If the information needed to answer the question is not present in the context, respond with 'N/A'.
3. Do not make up or infer information that is not explicitly stated in the context.
4. If only partial information is available, provide only that partial information and mark your confidence lower.
5. You must cite specific sections of the context that support your answer, referencing the SOURCE, SHA1, and PAGE.
6. For each key piece of information in your answer, provide a confidence level (0.0-1.0) based on how explicitly it is stated in the context.
7. Remember: absence of evidence is not evidence of absence - lack of mention in the context means 'N/A', not a negative answer.
"""

        # Add type-specific instructions
        if question_kind == "number":
            type_instructions = """
INSTRUCTIONS FOR NUMERICAL QUESTIONS:
1. Answer with ONLY a numeric value, with no additional text, words, or symbols.
2. For currency values, include ONLY the number (no currency symbols).
3. Use decimal points for fractional values (e.g., 12.5).
4. Do not use commas as thousand separators.
5. If the exact number isn't specified in the context, answer with 'N/A'.
6. Do not round or approximate unless the source text specifically does so.
7. Carefully check for the most recent or relevant time period specified in the question.
8. If multiple numbers could answer the question, use the most specific one that matches the question context.
"""
        elif question_kind == "boolean":
            type_instructions = """
INSTRUCTIONS FOR YES/NO QUESTIONS:
1. Answer ONLY with 'yes' or 'no'.
2. If the information cannot be determined from the context, answer with 'N/A'.
3. For comparing two companies, only answer 'yes' or 'no' if both companies have the relevant information.
4. Remember that absence of evidence is not evidence of absence - only answer 'no' if there is explicit contradictory information.
5. If the answer is technically correct but potentially misleading without context, still provide just 'yes' or 'no' but lower your confidence score.
"""
        elif question_kind == "name":
            type_instructions = """
INSTRUCTIONS FOR NAME QUESTIONS:
1. Answer with ONLY the exact name requested.
2. Provide the full, properly formatted name without additional commentary.
3. If the information is not in the context, answer with 'N/A'.
4. Do not make assumptions about names based on titles or partial information.
5. For questions about specific roles (CEO, CFO, etc.), verify that the person actually holds that exact role.
6. If multiple names could potentially answer the question, select the most accurate one based on the specific criteria in the question.
"""
        elif question_kind == "names":
            type_instructions = """
INSTRUCTIONS FOR MULTIPLE NAMES QUESTIONS:
1. Answer with a list of names, separated by commas if there are multiple.
2. Provide complete, properly formatted names.
3. If there are no names that match the criteria, or if the information is not in the context, answer with 'N/A'.
4. Include ALL relevant names that match the criteria in the question.
5. Return the list in chronological or importance order if that information is available.
6. Verify that each person actually meets the criteria specified in the question.
"""
        else:
            type_instructions = """
INSTRUCTIONS:
1. Answer the question directly and concisely based only on the context provided.
2. If the information is not available in the context, answer with 'N/A'.
"""

        # Response format instructions
        format_instructions = """
ANSWER FORMAT:
Your response must be in this JSON format:
{
  "answer": "YOUR_ANSWER_HERE", 
  "confidence": 0.0-1.0,
  "validation": "Brief notes on the verification steps you performed",
  "citations": ["Brief quotes from the context that support your answer"]
}

RESPONSE:
"""

        # Combine instructions
        return common_instructions + type_instructions + format_instructions
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse structured response from LLM into components
        
        Args:
            response: LLM response text
            
        Returns:
            Parsed response as dictionary
        """
        try:
            # Clean up response for JSON parsing
            clean_response = response.strip()
            
            # Handle cases where LLM added markdown code block
            if clean_response.startswith('```json'):
                clean_response = clean_response.replace('```json', '', 1)
            if clean_response.endswith('```'):
                clean_response = clean_response[:-3]
            
            clean_response = clean_response.strip()
            
            # Parse JSON
            parsed_response = json.loads(clean_response)
            return parsed_response
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse response as JSON: {e}")
            
            # Fallback parsing for non-JSON responses
            try:
                # Extract answer using regex
                answer_match = re.search(r'"answer"\s*:\s*"([^"]*)"', response)
                confidence_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', response)
                validation_match = re.search(r'"validation"\s*:\s*"([^"]*)"', response)
                
                result = {
                    "answer": answer_match.group(1) if answer_match else response,
                    "confidence": float(confidence_match.group(1)) if confidence_match else 0.5,
                    "validation": validation_match.group(1) if validation_match else ""
                }
                
                return result
            except Exception as ex:
                logger.error(f"Error in fallback parsing: {ex}")
                # Last resort
                return {"answer": "N/A", "confidence": 0.0, "validation": ""}
    
    def _format_answer(self, answer: str, question_kind: str) -> Any:
        """
        Format the answer based on the question kind
        
        Args:
            answer: Raw answer text
            question_kind: Type of question
            
        Returns:
            Formatted answer
        """
        # Clean up the answer
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
                
            # Filter out any empty strings or "N/A" values
            names = [name for name in names if name and name.lower() != "n/a"]
            
            # Remove brackets or quotation marks
            #names = [re.sub(r'^\s*[\[\("\']|[\]\)"\'], '', name).strip() for name in names]
            names = [re.sub(r'^\s*[\[\("\']|[\]\)"\']$', '', name).strip() for name in names]

            
            return names
            
        # For name type, return as is
        return answer if answer else "N/A"


    # In answer_generator.py, add specialized handling for leadership questions:
    def _extract_leadership_positions(self, context: str) -> list:
        """Extract leadership position changes from context"""
        leadership_positions = []
        
        # Patterns for leadership changes
        patterns = [
            r"appointed\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\s+as\s+([A-Za-z\s]+)",
            r"([A-Z][a-z]+\s+[A-Z][a-z]+)\s+joined\s+as\s+([A-Za-z\s]+)",
            r"([A-Z][a-z]+\s+[A-Z][a-z]+),\s+(?:the\s+)?([A-Za-z\s]+),\s+(?:resigned|left|departed)",
            r"named\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:as\s+)?(?:the\s+)?(?:new\s+)?([A-Za-z\s]+)"
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, context)
            for match in matches:
                try:
                    person = match.group(1)
                    position = match.group(2)
                    
                    # Filter for executive positions
                    if any(term in position.lower() for term in 
                        ["ceo", "cfo", "coo", "president", "director", "officer", "executive"]):
                        leadership_positions.append(position.strip())
                except IndexError:
                    continue
        
        return leadership_positions

    # In answer_generator.py, enhance boolean answer validation:
    def _validate_boolean_answer(self, answer: str, question: str, context: str) -> dict:
        """Validate boolean answers with extra checks for false negatives"""
        
        # Extract what we're looking for
        topics = {
            "share buyback": ["buyback", "repurchase", "buy back"],
            "mergers or acquisitions": ["merger", "acquisition", "acquire"],
            "dividend policy": ["dividend", "payout", "distribution"],
            "product launches": ["launch", "introduce", "new product"],
            "esg initiatives": ["esg", "environmental", "sustainable"]
        }
        
        # Determine which topic we're looking for
        target_topic = None
        for topic, keywords in topics.items():
            if any(keyword in question.lower() for keyword in keywords):
                target_topic = topic
                break
        
        # If we found a topic, look for evidence in the context
        if target_topic:
            keywords = topics[target_topic]
            evidence = any(keyword in context.lower() for keyword in keywords)
            
            # For boolean questions, we default to False when no evidence is found
            if not evidence:
                return {"value": "False", "confidence": 0.9}
            
            # If evidence is found, return the original answer with high confidence
            return {"value": answer, "confidence": 0.9}
        
        # Default return
        return {"value": answer, "confidence": 0.7}


    def _validate_answer(self, answer: Any, context: str, question_kind: str) -> float:
        """
        Validate answer against context
        
        Args:
            answer: Formatted answer
            context: Full context string
            question_kind: Type of question
            
        Returns:
            Validation score (0.0-1.0)
        """
        # Convert answer to string form for validation
        if isinstance(answer, (int, float)):
            answer_str = str(answer).lower()
        elif isinstance(answer, list):
            # For list of names, check if ANY name is in the context
            if not answer:
                return 1.0  # Empty list is valid
            
            # Check each name
            name_scores = []
            for name in answer:
                name_str = str(name).lower()
                score = 1.0 if name_str in context.lower() else 0.0
                name_scores.append(score)
            
            # Return average score for all names
            return sum(name_scores) / len(name_scores) if name_scores else 0.0
        else:
            answer_str = str(answer).lower()
        
        # If answer is N/A, it's always valid (conservative approach)
        if answer_str == "n/a":
            return 1.0
        
        # For numeric answers, be more strict about exact matches
        if question_kind == "number":
            # Need to find the exact number in context
            # Handle thousands separators and variations
            clean_answer = answer_str.replace(",", "").replace(" ", "")
            
            # Look for exact matches of the number
            if re.search(r'\b' + re.escape(clean_answer) + r'\b', context.lower()):
                return 1.0
                
            # Check for variations with currency symbols or units
            for prefix in ['€', '£', '¥', 'EUR', 'GBP', 'JPY']:
                if re.search(r'\b' + re.escape(prefix) + r'\s*' + re.escape(clean_answer), context.lower()):
                    return 1.0
                    
            for suffix in ['million', 'billion', 'percent', '%']:
                if re.search(r'\b' + re.escape(clean_answer) + r'\s*' + re.escape(suffix), context.lower()):
                    return 1.0
            
            # Didn't find exact match
            return 0.0
            
        # For name questions, check if name appears in context
        elif question_kind == "name":
            return 1.0 if answer_str in context.lower() else 0.0
            
        # For boolean, harder to validate directly
        elif question_kind == "boolean":
            # This requires more sophisticated analysis
            # For simplicity, don't invalidate boolean answers
            return 0.5  # Medium confidence
        
        # Default: check if answer text is in context
        return 1.0 if answer_str in context.lower() else 0.0


    # In answer_generator.py, add currency handling:
    def _extract_currency_value(self, text: str, target_currency: str) -> float:
        """Extract monetary values in the specified currency"""
        # Currency symbol mapping
        currency_symbols = {
            'USD': r'\$|USD',
            'EUR': r'€|EUR',
            'GBP': r'£|GBP',
            'AUD': r'A\$|AUD'
        }
        
        # Look for values in the specified currency
        symbol_pattern = currency_symbols.get(target_currency, r'\$')
        
        # Pattern to match currency value with scaling (million, billion)
        pattern = fr'{symbol_pattern}\s*([\d,]+\.?\d*)(?:\s*(million|m|billion|b))?'
        
        matches = re.findall(pattern, text, re.IGNORECASE)
        if not matches:
            return None
        
        # Process the first match
        value_str, scale = matches[0]
        value = float(value_str.replace(',', ''))
        
        # Apply scaling
        if scale.lower() in ('million', 'm'):
            value *= 1_000_000
        elif scale.lower() in ('billion', 'b'):
            value *= 1_000_000_000
        
        return value




    def _extract_references(self, docs: List[Document], answer: Any, question_kind: str) -> List[Dict[str, Any]]:
        """
        Extract references to source documents
        
        Args:
            docs: Context documents
            answer: Formatted answer
            question_kind: Type of question
            
        Returns:
            List of reference dictionaries
        """
        references = []
        
        # Skip if answer is N/A or empty
        if answer == "N/A" or (isinstance(answer, list) and len(answer) == 0):
            return references
            
        # Convert answer to search form for matching
        if isinstance(answer, (int, float)):
            answer_str = str(answer).lower()
            # For numbers, also create variants with thousand separators
            search_variants = [
                answer_str,
                f"{answer:,}".lower(),  # With commas
                f"{answer:.2f}".lower() if isinstance(answer, float) else answer_str
            ]
        elif isinstance(answer, list):
            # For list of names, search for each name
            search_variants = []
            for name in answer:
                search_variants.append(str(name).lower())
        else:
            answer_str = str(answer).lower()
            search_variants = [answer_str]
        
        # Go through documents and find matches
        for doc in docs:
            sha1 = doc.metadata.get("sha1", "")
            page_num = doc.metadata.get("page", 0)
            content_lower = doc.page_content.lower()
            
            # Skip if no SHA1 (needed for reference)
            if not sha1:
                continue
                
            # Check for answer text in document
            matches_found = False
            
            # For numbers, look for any variant
            if question_kind == "number":
                for variant in search_variants:
                    if variant in content_lower or any(term in content_lower for term in search_variants):
                        matches_found = True
                        break
            # For names, look for each name
            elif question_kind == "names" and isinstance(answer, list):
                # If any name is found, add reference
                for name in answer:
                    if str(name).lower() in content_lower:
                        matches_found = True
                        break
            # For other question types
            else:
                matches_found = any(variant in content_lower for variant in search_variants)
            
            # Add reference if there's a match
            if matches_found:
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


if __name__ == "__main__":
    # Simple test if run as script
    from langchain_core.documents import Document
    
    # Create test documents
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
    
    # Create generator
    generator = AnswerGenerator(model_name="gpt-3.5-turbo")  # Use faster model for testing
    
    # Test questions
    number_question = "What was the revenue of ABC Corp in 2023?"
    name_question = "Who is the CEO of ABC Corp?"
    names_question = "Who are the members of the Board of Directors?"
    boolean_question = "Does ABC Corp have more than $5 million in revenue?"
    
    # Test each question
    print("\nTesting number question...")
    result = generator.answer_question(number_question, docs, "number")
    print(f"Answer: {result['value']}")
    print(f"References: {result['references']}")
    
    print("\nTesting name question...")
    result = generator.answer_question(name_question, docs, "name")
    print(f"Answer: {result['value']}")
    print(f"References: {result['references']}")
    
    print("\nTesting names question...")
    result = generator.answer_question(names_question, docs, "names")
    print(f"Answer: {result['value']}")
    print(f"References: {result['references']}")
    
    print("\nTesting boolean question...")
    result = generator.answer_question(boolean_question, docs, "boolean")
    print(f"Answer: {result['value']}")
    print(f"References: {result['references']}")
