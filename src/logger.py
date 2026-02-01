"""
Logging configuration and utilities for the medical RAG system.

This module provides centralized logging setup to track all system operations,
errors, and important events throughout the application lifecycle.
"""

import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from typing import Optional


class LoggerSetup:
    """Setup and manage logging for the medical RAG system."""

    @staticmethod
    def setup_logger(
        name: str,
        log_dir: str = "./logs",
        level: int = logging.INFO
    ) -> logging.Logger:
        """
        Setup a logger with both file and console handlers.
        
        Args:
            name: Logger name (typically __name__)
            log_dir: Directory for log files
            level: Logging level (default: INFO)
            
        Returns:
            Configured logger instance
        """
        # Create logs directory if it doesn't exist
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        
        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler (rotating)
        log_file = log_path / f"mediai_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10485760,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        if not logger.handlers:  # Avoid duplicate handlers
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger


# TODO: Add metrics tracking for:
# - Query response times
# - Vector database hit rates
# - Model performance metrics
# - User satisfaction scores
# - Query themes distribution
# - Vector database usage statistics

# TODO: Add structured logging for:
# - Query performance profiling
# - Error categorization and analysis
# - Model inference times
# - Cache hit/miss ratios
# - API call tracking (e.g., Pinecone, Ollama)
