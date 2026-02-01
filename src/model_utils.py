"""
Model management and inference utilities.

This module handles interactions with various Ollama models for different tasks
including theme detection, embedding, and response generation.
"""

import logging
import json
from typing import Optional, Dict, Any
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

try:
    from .logger import LoggerSetup
    from .enums import ModelType, QuestionTheme
    from .models import ThemeDetectionResponse, MedicalAnswer
    from .prompts import PromptTemplates
except ImportError:
    from logger import LoggerSetup
    from enums import ModelType, QuestionTheme
    from models import ThemeDetectionResponse, MedicalAnswer
    from prompts import PromptTemplates

logger = LoggerSetup.setup_logger(__name__)


class ModelManager:
    """Manage different model instances for various tasks."""

    def __init__(self):
        """Initialize model manager with different model types."""
        self.theme_detector: Optional[ChatOllama] = None
        self.main_generator: Optional[ChatOllama] = None
        self.embeddings: Optional[OllamaEmbeddings] = None
        
        logger.info("ModelManager initialized")

    def get_theme_detector(self) -> ChatOllama:
        """
        Get or create the theme detection model.
        
        Returns:
            ChatOllama instance for theme detection
        """
        if self.theme_detector is None:
            logger.info(f"Initializing theme detector model: {ModelType.THEME_DETECTOR.value}")
            self.theme_detector = ChatOllama(
                model=ModelType.THEME_DETECTOR.value,
                temperature=0.1,  # Low temperature for consistent results
                format="json"
            )
        return self.theme_detector

    def get_main_generator(self, temperature: float = 0.3) -> ChatOllama:
        """
        Get or create the main generation model.
        
        Args:
            temperature: Temperature for generation (0.0-1.0)
            
        Returns:
            ChatOllama instance for main generation
        """
        if self.main_generator is None:
            logger.info(f"Initializing main generator model: {ModelType.MAIN_GENERATOR.value}")
            self.main_generator = ChatOllama(
                model=ModelType.MAIN_GENERATOR.value,
                temperature=temperature,
                format="json"
            )
        return self.main_generator

    def get_embeddings(self) -> OllamaEmbeddings:
        """
        Get or create embeddings model.
        
        Returns:
            OllamaEmbeddings instance
        """
        if self.embeddings is None:
            logger.info(f"Initializing embeddings model: {ModelType.EMBEDDING.value}")
            self.embeddings = OllamaEmbeddings(model=ModelType.EMBEDDING.value)
        return self.embeddings


class ThemeDetector:
    """Detect and classify question themes."""

    def __init__(self, model_manager: ModelManager):
        """
        Initialize theme detector.
        
        Args:
            model_manager: ModelManager instance
        """
        self.model_manager = model_manager
        self.model = model_manager.get_theme_detector()
        logger.info("ThemeDetector initialized")

    def detect_theme(self, question: str) -> ThemeDetectionResponse:
        """
        Detect the theme of a medical question.
        
        Args:
            question: Medical question to analyze
            
        Returns:
            ThemeDetectionResponse with detected theme and confidence
            
        TODO: Add few-shot examples for better detection
        TODO: Add fine-tuning on medical question datasets
        TODO: Add fallback mechanisms for ambiguous questions
        """
        logger.info(f"Detecting theme for question: {question[:100]}...")
        
        prompt = PromptTemplates.get_theme_detection_prompt()
        
        try:
            response = self.model.invoke(f"{prompt}\n\nQuestion: {question}")
            logger.debug(f"Raw response: {response}")
            
            # Parse response - handle both string and structured output
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Try to extract JSON from response
            try:
                # Find JSON in response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    result = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse theme detection response, using fallback: {e}")
                result = {
                    "detected_theme": QuestionTheme.GENERAL.value,
                    "confidence": 0.5,
                    "reasoning": "Theme detection failed, defaulting to general"
                }
            
            theme_response = ThemeDetectionResponse(
                question=question,
                detected_theme=result.get('detected_theme', QuestionTheme.GENERAL.value),
                confidence=result.get('confidence', 0.5),
                reasoning=result.get('reasoning', 'Theme detection completed')
            )
            
            logger.info(f"Detected theme: {theme_response.detected_theme} (confidence: {theme_response.confidence})")
            return theme_response
            
        except Exception as e:
            logger.error(f"Error detecting theme: {e}")
            # Return default theme on error
            return ThemeDetectionResponse(
                question=question,
                detected_theme=QuestionTheme.GENERAL.value,
                confidence=0.0,
                reasoning=f"Error in detection: {str(e)}"
            )


class ResponseGenerator:
    """Generate medical answers using retrieved context and models."""

    def __init__(self, model_manager: ModelManager):
        """
        Initialize response generator.
        
        Args:
            model_manager: ModelManager instance
        """
        self.model_manager = model_manager
        self.model = model_manager.get_main_generator()
        logger.info("ResponseGenerator initialized")

    def generate_answer(
        self,
        question: str,
        theme: str,
        context: str,
        has_vector_context: bool = False
    ) -> str:
        """
        Generate a medical answer based on question, theme, and context.
        
        Args:
            question: User's medical question
            theme: Detected question theme
            context: Retrieved context from vector database
            has_vector_context: Whether context comes from vector DB
            
        Returns:
            Generated answer string
            
        TODO: Add streaming response support
        TODO: Add response validation and fact-checking
        TODO: Add output truncation for very long answers
        """
        logger.info(f"Generating answer for theme: {theme}")
        
        # Get theme-specific system prompt
        system_prompt = PromptTemplates.get_system_prompt(theme)
        
        # Build user message
        if context:
            user_message = f"""Question: {question}

Context from Medical Knowledge Base:
{context}

Please provide a comprehensive answer using the context above. Cite specific sources when using information from the knowledge base."""
        else:
            user_message = f"""Question: {question}

No specific context was found in the medical knowledge base for this question. 
Please provide an answer based on general medical knowledge, and clearly indicate that this is from your training data rather than curated sources."""
        
        try:
            # Create prompt template
            system_template = SystemMessagePromptTemplate.from_template(system_prompt)
            human_template = HumanMessagePromptTemplate.from_template(user_message)
            prompt = ChatPromptTemplate.from_messages([system_template, human_template])
            
            # Generate response
            chain = prompt | self.model
            response = chain.invoke({})
            
            # Extract text content
            answer_text = response.content if hasattr(response, 'content') else str(response)
            
            logger.info(f"Answer generated successfully (length: {len(answer_text)} chars)")
            return answer_text
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise

    def extract_json_response(self, response_text: str) -> Dict[str, Any]:
        """
        Extract JSON from model response.
        
        Args:
            response_text: Raw response text from model
            
        Returns:
            Parsed JSON dictionary
        """
        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
        except Exception as e:
            logger.error(f"Error extracting JSON: {e}")
            raise


class PromptBuilder:
    """Build and manage prompts for different model tasks."""

    @staticmethod
    def build_qa_prompt(
        question: str,
        context: str,
        theme: str = "general"
    ) -> str:
        """
        Build a Q&A prompt with context.
        
        Args:
            question: User question
            context: Retrieved context
            theme: Question theme
            
        Returns:
            Formatted prompt string
        """
        template = PromptTemplates.get_user_prompt_template()
        
        formatted_context = context if context else "No relevant context found in knowledge base."
        
        return template.format(question=question, context=formatted_context)

    @staticmethod
    def build_context_summary(documents: list, max_length: int = 500) -> str:
        """
        Build a summary of retrieved documents.
        
        Args:
            documents: Retrieved document objects
            max_length: Maximum summary length
            
        Returns:
            Formatted context string
            
        TODO: Implement intelligent summarization
        TODO: Add document importance ranking
        """
        if not documents:
            return ""
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.get('source', 'Unknown') if isinstance(doc, dict) else getattr(doc, 'metadata', {}).get('source', 'Unknown')
            content = doc.get('content', str(doc)) if isinstance(doc, dict) else getattr(doc, 'page_content', str(doc))
            
            # Truncate individual document content
            if len(content) > 200:
                content = content[:200] + "..."
            
            context_parts.append(f"[Source {i}: {source}]\n{content}")
        
        full_context = "\n\n".join(context_parts)
        
        # Truncate if too long
        if len(full_context) > max_length:
            full_context = full_context[:max_length] + "..."
        
        return full_context
