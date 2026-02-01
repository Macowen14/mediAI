"""
Enum definitions for medical RAG system.

This module contains enums used throughout the medical AI system
to standardize and categorize different aspects of the application.
"""

from enum import Enum


class QuestionTheme(str, Enum):
    """
    Enum for categorizing medical questions into different themes.
    
    This helps determine the appropriate response strategy and prompt template
    to use based on the nature of the question.
    """
    
    ANATOMY = "anatomy"  # Questions about body structure
    PHYSIOLOGY = "physiology"  # Questions about how body systems work
    PATHOLOGY = "pathology"  # Questions about disease and conditions
    PHARMACOLOGY = "pharmacology"  # Questions about drugs and medications
    SYMPTOMS = "symptoms"  # Questions about symptoms and clinical signs
    DIAGNOSIS = "diagnosis"  # Questions about diagnostic procedures
    TREATMENT = "treatment"  # Questions about treatment options
    PREVENTION = "prevention"  # Questions about disease prevention
    LIFESTYLE = "lifestyle"  # Questions about healthy lifestyle
    GENERAL = "general"  # General medical questions that don't fit other categories


class ModelType(str, Enum):
    """
    Enum for different model types used in the system.
    
    Different models are used for different tasks:
    - Theme detection (small, fast model)
    - Main generation (larger, more capable model)
    - Embedding (specialized for text encoding)
    """
    
    THEME_DETECTOR = "ministral-3:8"  # Small, fast model for theme detection
    MAIN_GENERATOR = "mistral-large-3:675b-cloud"  # Large model for main responses
    EMBEDDING = "nomic-embed-text:latest"  # Specialized embedding model
    # TODO: Add support for gemma3:27b-cloud for faster general queries
    # TODO: Add support for mistral-large-3:675b-cloud for high-quality responses
    # TODO: Add vision model (qwen3-vl:235b-cloud) for image analysis


class ResponseSource(str, Enum):
    """
    Enum for indicating the source of information in a response.
    
    Helps users understand whether the answer comes from the knowledge base,
    vector database, or the model's own training data.
    """
    
    VECTOR_DB = "vector_db"  # Answer sourced from vector database
    KNOWLEDGE_BASE = "knowledge_base"  # Answer sourced from knowledge base
    MODEL_TRAINING = "model_training"  # Answer from model's training data
    HYBRID = "hybrid"  # Answer combining multiple sources
