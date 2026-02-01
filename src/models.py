"""
Pydantic models for structured outputs and type safety.

This module defines all Pydantic models used throughout the medical RAG system
to ensure type safety, validation, and structured data handling.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class MedicalAnswer(BaseModel):
    """
    Structured response from the medical AI system.
    
    Contains the main answer along with metadata about sources,
    confidence, and theme information.
    """
    
    answer: str = Field(
        ...,
        description="The comprehensive answer to the user's medical question"
    )
    question: str = Field(
        ...,
        description="The original question asked by the user"
    )
    theme: str = Field(
        default="general",
        description="The detected theme/category of the question"
    )
    sources: List[str] = Field(
        default_factory=list,
        description="List of source documents used to generate the answer"
    )
    source_type: str = Field(
        default="hybrid",
        description="Whether answer is from vector_db, knowledge_base, model_training, or hybrid"
    )
    confidence_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence score from 0 to 1 indicating reliability of the answer"
    )
    has_vector_context: bool = Field(
        default=False,
        description="Whether the answer was augmented with vector database context"
    )
    context_summary: Optional[str] = Field(
        default=None,
        description="Summary of the context retrieved from vector database"
    )
    caveats: Optional[str] = Field(
        default=None,
        description="Important disclaimers or limitations of the answer"
    )


class ThemeDetectionRequest(BaseModel):
    """Request model for theme detection."""
    
    question: str = Field(
        ...,
        description="The medical question to analyze for theme"
    )


class ThemeDetectionResponse(BaseModel):
    """Response model for theme detection."""
    
    question: str = Field(..., description="The original question")
    detected_theme: str = Field(..., description="The detected theme from QuestionTheme enum")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the theme detection"
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of why this theme was detected"
    )


class VectorSearchResult(BaseModel):
    """Result from vector database search."""
    
    content: str = Field(..., description="The content of the document chunk")
    source: str = Field(..., description="Source file or reference")
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance score from similarity search"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the document"
    )


class RAGContext(BaseModel):
    """Context passed to the RAG pipeline."""
    
    question: str = Field(..., description="User's question")
    theme: str = Field(..., description="Detected question theme")
    retrieved_documents: List[VectorSearchResult] = Field(
        default_factory=list,
        description="Documents retrieved from vector DB"
    )
    has_sufficient_context: bool = Field(
        default=False,
        description="Whether retrieved documents provide sufficient context"
    )


class ImageVector(BaseModel):
    """
    Model for storing image metadata and vector information.
    
    TODO: Implement image vector storage and retrieval
    TODO: Support multiple image formats (PNG, JPG, DICOM)
    TODO: Add medical image specific metadata (modality, body part, etc.)
    """
    
    image_id: str = Field(..., description="Unique identifier for the image")
    image_path: str = Field(..., description="Path to the image file")
    vector: List[float] = Field(..., description="Vector embedding of the image")
    description: Optional[str] = Field(
        default=None,
        description="Description of the image content"
    )
    image_type: Optional[str] = Field(
        default=None,
        description="Type of medical image (e.g., X-ray, MRI, CT)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional image metadata"
    )


class ToolCall(BaseModel):
    """
    Model for representing tool calls/actions the agent should take.
    
    TODO: Implement internet search tool
    TODO: Implement medical database lookup tool
    TODO: Implement image analysis tool
    TODO: Implement literature search tool
    """
    
    tool_name: str = Field(
        ...,
        description="Name of the tool to call (e.g., 'internet_search', 'image_analysis')"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters to pass to the tool"
    )
    reason: str = Field(
        ...,
        description="Reason why this tool should be called"
    )
