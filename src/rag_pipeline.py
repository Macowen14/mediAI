"""
Main RAG pipeline orchestrator for medical AI system.

This module coordinates the entire RAG (Retrieval-Augmented Generation) workflow:
1. Theme detection
2. Vector search
3. Context evaluation
4. Response generation
5. Output formatting

This is the core pipeline that ties together all utilities.
"""

import logging
from typing import Optional, List

try:
    from .logger import LoggerSetup
    from .enums import QuestionTheme, ResponseSource
    from .models import MedicalAnswer, RAGContext, VectorSearchResult
    from .model_utils import ModelManager, ThemeDetector, ResponseGenerator, PromptBuilder
    from .vector_utils import VectorSearch, VectorStore
except ImportError:
    from logger import LoggerSetup
    from enums import QuestionTheme, ResponseSource
    from models import MedicalAnswer, RAGContext, VectorSearchResult
    from model_utils import ModelManager, ThemeDetector, ResponseGenerator, PromptBuilder
    from vector_utils import VectorSearch, VectorStore
from langchain_pinecone import PineconeVectorStore

logger = LoggerSetup.setup_logger(__name__)


class MedicalRAGPipeline:
    """
    Main pipeline for medical question answering with RAG.
    
    Workflow:
    1. Detect question theme
    2. Search vector database for relevant documents
    3. Evaluate if context is sufficient
    4. Generate answer with or without context
    5. Return structured response
    """

    def __init__(self, vectorstore: PineconeVectorStore):
        """
        Initialize RAG pipeline.
        
        Args:
            vectorstore: PineconeVectorStore instance
        """
        self.vectorstore = vectorstore
        self.model_manager = ModelManager()
        self.theme_detector = ThemeDetector(self.model_manager)
        self.response_generator = ResponseGenerator(self.model_manager)
        
        logger.info("MedicalRAGPipeline initialized")

    def process_question(
        self,
        question: str,
        search_k: int = 3,
        relevance_threshold: float = 0.5
    ) -> MedicalAnswer:
        """
        Process a medical question through the full RAG pipeline.
        
        Args:
            question: Medical question from user
            search_k: Number of documents to retrieve
            relevance_threshold: Minimum relevance score for context
            
        Returns:
            MedicalAnswer with comprehensive response and metadata
        """
        logger.info(f"Processing question: {question[:100]}...")
        
        try:
            # Step 1: Detect question theme
            logger.info("Step 1: Detecting question theme...")
            theme_response = self.theme_detector.detect_theme(question)
            theme = theme_response.detected_theme
            theme_confidence = theme_response.confidence
            
            logger.info(f"  Theme: {theme} (confidence: {theme_confidence})")
            
            # Step 2: Search vector database
            logger.info("Step 2: Searching vector database...")
            search_results = VectorSearch.search_similar_documents(
                self.vectorstore,
                question,
                k=search_k
            )
            
            logger.info(f"  Found {len(search_results)} documents")
            
            # Step 3: Evaluate context sufficiency
            logger.info("Step 3: Evaluating context sufficiency...")
            has_sufficient = VectorSearch.has_sufficient_context(
                search_results,
                relevance_threshold
            )
            
            logger.info(f"  Sufficient context: {has_sufficient}")
            
            # Build context string
            context_summary = PromptBuilder.build_context_summary(search_results)
            
            # Step 4: Generate answer
            logger.info("Step 4: Generating answer...")
            answer_text = self.response_generator.generate_answer(
                question=question,
                theme=theme,
                context=context_summary,
                has_vector_context=has_sufficient
            )
            
            logger.info("  Answer generated successfully")
            
            # Step 5: Build structured response
            logger.info("Step 5: Building structured response...")
            
            # Determine response source
            if has_sufficient:
                source_type = ResponseSource.HYBRID.value
            else:
                source_type = ResponseSource.MODEL_TRAINING.value
            
            # Extract sources
            sources = [result.source for result in search_results]
            
            # Calculate confidence
            confidence = (theme_confidence + (0.8 if has_sufficient else 0.4)) / 2
            
            # Build caveats
            caveats = self._build_caveats(theme, has_sufficient)
            
            medical_answer = MedicalAnswer(
                answer=answer_text,
                question=question,
                theme=theme,
                sources=sources,
                source_type=source_type,
                confidence_score=min(confidence, 1.0),
                has_vector_context=has_sufficient,
                context_summary=context_summary if has_sufficient else None,
                caveats=caveats
            )
            
            logger.info("Response structured successfully")
            return medical_answer
            
        except Exception as e:
            logger.error(f"Error processing question: {e}", exc_info=True)
            raise

    def _build_caveats(self, theme: str, has_context: bool) -> str:
        """
        Build appropriate caveats based on theme and context.
        
        Args:
            theme: Question theme
            has_context: Whether context was found
            
        Returns:
            Caveat string for the response
        """
        caveats_parts = []
        
        # General disclaimer
        caveats_parts.append("This information is for educational purposes only and should not replace professional medical advice.")
        
        # Theme-specific caveats
        if theme == QuestionTheme.DIAGNOSIS.value:
            caveats_parts.append("This information cannot be used for self-diagnosis. Please consult a healthcare professional.")
        elif theme == QuestionTheme.TREATMENT.value:
            caveats_parts.append("Do not use this information to make treatment decisions without consulting a qualified healthcare provider.")
        elif theme == QuestionTheme.PHARMACOLOGY.value:
            caveats_parts.append("Always consult with a pharmacist or doctor before taking any medications.")
        elif theme == QuestionTheme.SYMPTOMS.value:
            caveats_parts.append("Seek immediate medical attention if experiencing severe symptoms.")
        
        # Context caveats
        if not has_context:
            caveats_parts.append("This answer is based on general medical knowledge, not curated medical literature.")
        
        return " ".join(caveats_parts)

    def batch_process_questions(
        self,
        questions: List[str],
        search_k: int = 3
    ) -> List[MedicalAnswer]:
        """
        Process multiple questions in batch.
        
        Args:
            questions: List of questions to process
            search_k: Number of documents to retrieve per question
            
        Returns:
            List of MedicalAnswer objects
            
        TODO: Implement parallel processing for better performance
        TODO: Add caching for identical or similar questions
        TODO: Add batch result aggregation and analysis
        """
        logger.info(f"Processing batch of {len(questions)} questions")
        
        results = []
        for i, question in enumerate(questions, 1):
            logger.info(f"Processing question {i}/{len(questions)}")
            try:
                answer = self.process_question(question, search_k=search_k)
                results.append(answer)
            except Exception as e:
                logger.error(f"Error processing question {i}: {e}")
                # Continue with next question even if one fails
                continue
        
        logger.info(f"Batch processing complete. Processed {len(results)}/{len(questions)} questions")
        return results


# TODO: Implement advanced features:
# - Query expansion using LLM
# - Re-ranking of retrieved documents
# - Multi-turn conversation support
# - User feedback integration for continuous improvement
# - A/B testing different prompt templates
# - Response caching with TTL
# - Rate limiting and usage tracking
# - Feedback loop for model refinement
# - Support for multiple knowledge bases
# - Document update detection and re-indexing
