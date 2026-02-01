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
        level: int = logging.INFO,
        retention_days: int = 30
    ) -> logging.Logger:
        """
        Setup a logger with both file and console handlers.
        
        Creates daily log files with automatic rotation at midnight.
        Old logs are kept for the specified retention period.
        
        Args:
            name: Logger name (typically __name__)
            log_dir: Directory for log files
            level: Logging level (default: INFO)
            retention_days: Number of days to keep old logs (default: 30)
            
        Returns:
            Configured logger instance
        """
        # Create logs directory if it doesn't exist
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        
        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Remove existing handlers to avoid duplicates (important in Jupyter notebooks)
        logger.handlers.clear()
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # File handler with daily rotation
        # Creates a new log file each day at midnight
        log_file = log_path / "mediai.log"
        file_handler = logging.handlers.TimedRotatingFileHandler(
            log_file,
            when='midnight',
            interval=1,
            backupCount=retention_days,
            encoding='utf-8'
        )
        # Add suffix with date format for rotated files
        file_handler.suffix = "%Y%m%d"
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        
        # Console handler (show all INFO and above for visibility)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)  # Match logger level for console output
        console_handler.setFormatter(console_formatter)
        
        # Add handlers to logger  
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    @staticmethod
    def cleanup_old_logs(log_dir: str = "./logs", retention_days: int = 30) -> int:
        """
        Manually cleanup log files older than retention period.
        
        Args:
            log_dir: Directory containing log files
            retention_days: Number of days to keep logs
            
        Returns:
            Number of files deleted
        """
        from datetime import datetime, timedelta
        
        log_path = Path(log_dir)
        if not log_path.exists():
            return 0
        
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        deleted_count = 0
        
        for log_file in log_path.glob("mediai*.log*"):
            if log_file.stat().st_mtime < cutoff_date.timestamp():
                log_file.unlink()
                deleted_count += 1
        
        return deleted_count


# LOGGING FEATURES:
# ✅ Daily log file rotation at midnight
# ✅ Automatic retention policy (default 30 days)
# ✅ Timestamped log file names (mediai.log.YYYYMMDD)
# ✅ Current day logs to mediai.log
# ✅ Separate formatters for file and console output
# ✅ Manual cleanup utility for old logs

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
