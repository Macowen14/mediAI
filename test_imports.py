#!/usr/bin/env python3
"""Quick test to verify all imports work correctly."""

import sys
sys.path.insert(0, 'src')

try:
    from enums import QuestionTheme, ModelType
    print("✓ enums imported")
    from models import MedicalAnswer
    print("✓ models imported")
    from prompts import PromptTemplates
    print("✓ prompts imported")
    from logger import LoggerSetup
    print("✓ logger imported")
    from vector_utils import DocumentLoader, VectorStore
    print("✓ vector_utils imported")
    from model_utils import ModelManager, ThemeDetector
    print("✓ model_utils imported")
    from rag_pipeline import MedicalRAGPipeline
    print("✓ rag_pipeline imported")
    
    print("\n✅ All imports successful!")
    
    # Show available themes
    print("\nAvailable QuestionThemes:")
    for theme in QuestionTheme:
        print(f"  - {theme.value}")
        
    print("\nAvailable Models:")
    print(f"  - Theme Detector: {ModelType.THEME_DETECTOR.value}")
    print(f"  - Main Generator: {ModelType.MAIN_GENERATOR.value}")
    print(f"  - Embedding: {ModelType.EMBEDDING.value}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
