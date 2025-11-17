"""
Logging utility for structured logging across the project.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import json


class StructuredLogger:
    """Structured logger with file and console output."""
    
    def __init__(
        self,
        name: str,
        log_dir: Optional[Path] = None,
        level: int = logging.INFO,
        console_output: bool = True
    ):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            log_dir: Directory to save log files
            level: Logging level
            console_output: Whether to output to console
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers = []  # Clear existing handlers
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        
        # File handler
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = log_dir / f"{name}_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
    
    def info(self, message: str, **kwargs):
        """Log info message with optional structured data."""
        self.logger.info(self._format_message(message, kwargs))
    
    def warning(self, message: str, **kwargs):
        """Log warning message with optional structured data."""
        self.logger.warning(self._format_message(message, kwargs))
    
    def error(self, message: str, **kwargs):
        """Log error message with optional structured data."""
        self.logger.error(self._format_message(message, kwargs))
    
    def debug(self, message: str, **kwargs):
        """Log debug message with optional structured data."""
        self.logger.debug(self._format_message(message, kwargs))
    
    def critical(self, message: str, **kwargs):
        """Log critical message with optional structured data."""
        self.logger.critical(self._format_message(message, kwargs))
    
    @staticmethod
    def _format_message(message: str, data: dict) -> str:
        """Format message with structured data."""
        if data:
            return f"{message} | {json.dumps(data)}"
        return message


def get_logger(
    name: str,
    log_dir: Optional[Path] = None,
    level: int = logging.INFO
) -> StructuredLogger:
    """
    Get or create a logger instance.
    
    Args:
        name: Logger name
        log_dir: Directory to save log files
        level: Logging level
    
    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name, log_dir, level)
