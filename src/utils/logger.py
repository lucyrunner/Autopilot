"""
Structured Logging System

Provides JSON-formatted logging for better observability in production.
Structured logs are easier to parse, search, and analyze with tools like
ELK Stack (Elasticsearch, Logstash, Kibana) or Splunk.

Why JSON logging?
- Each log is a JSON object with consistent fields
- Easy to filter and aggregate (e.g., "show all ERROR logs from worker-2")
- Machine-readable for automated alerting
"""

import logging
import json
import sys
from datetime import datetime
from typing import Dict, Any
from config.settings import get_settings


class JSONFormatter(logging.Formatter):
    """
    Custom log formatter that outputs structured JSON logs.
    
    Each log message becomes a JSON object with standard fields:
    - timestamp: When the log was created (ISO format)
    - level: Log severity (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - logger: Which component logged this (e.g., "lane_detector")
    - message: The actual log message
    - module: Python module name
    - function: Function that created the log
    - line: Line number in source code
    - exception: Stack trace (if an exception was logged)
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Convert a log record into a JSON string.
        
        Args:
            record: LogRecord object created by the logging system
            
        Returns:
            str: JSON-formatted log message
        """
        # Build the base log data dictionary
        log_data = {
            # ISO 8601 timestamp (universal format, timezone-aware)
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            
            # Log level as string (INFO, ERROR, etc.)
            'level': record.levelname,
            
            # Logger name (usually module name)
            'logger': record.name,
            
            # The actual message
            'message': record.getMessage(),
            
            # Source code location (helps with debugging)
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception information if this log was created during exception handling
        # Example: logger.exception("Failed to process frame")
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add any extra fields passed via logger.info("msg", extra={'key': 'value'})
        # This is useful for adding context like request_id, user_id, etc.
        if hasattr(record, 'extra'):
            log_data.update(record.extra)
        
        # Convert to JSON string
        # ensure_ascii=False: Allow unicode characters
        return json.dumps(log_data, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """
    Human-readable text formatter for development.
    
    Easier to read during local development than JSON.
    Format: "2024-01-15 10:30:45 - lane_detector - INFO - Lane detected"
    """
    
    def __init__(self):
        # Define the log format string
        # %(asctime)s: Timestamp
        # %(name)s: Logger name
        # %(levelname)s: Log level
        # %(message)s: Log message
        super().__init__(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.
    
    This is the main function to use for getting a logger in any module.
    It automatically configures the logger based on application settings.
    
    Args:
        name: Logger name (typically __name__ to use module name)
        
    Returns:
        logging.Logger: Configured logger instance
        
    Example:
        # In lane_detector.py
        from src.utils.logger import get_logger
        
        logger = get_logger(__name__)
        logger.info("Processing frame", extra={'frame_id': 123, 'fps': 30})
        
        # Output (JSON format):
        # {
        #   "timestamp": "2024-01-15T10:30:45.123Z",
        #   "level": "INFO",
        #   "logger": "src.perception.lane_detector",
        #   "message": "Processing frame",
        #   "frame_id": 123,
        #   "fps": 30
        # }
    """
    # Load application settings
    settings = get_settings()
    
    # Get or create logger with given name
    logger = logging.getLogger(name)
    
    # Set log level from configuration
    # This determines minimum severity to log
    # DEBUG < INFO < WARNING < ERROR < CRITICAL
    logger.setLevel(settings.logging.level)
    
    # Avoid adding duplicate handlers if logger already configured
    # This prevents logs from appearing multiple times
    if logger.handlers:
        return logger
    
    # Create handler (where logs go: stdout, file, etc.)
    if settings.logging.output == "stdout":
        # Log to console (standard output)
        handler = logging.StreamHandler(sys.stdout)
    else:
        # Log to file
        # For production, consider using RotatingFileHandler to prevent
        # log files from growing indefinitely
        handler = logging.FileHandler(settings.logging.output)
    
    # Choose formatter based on configuration
    if settings.logging.format == 'json':
        # Structured JSON logs (production)
        formatter = JSONFormatter()
    else:
        # Human-readable text logs (development)
        formatter = TextFormatter()
    
    # Attach formatter to handler
    handler.setFormatter(formatter)
    
    # Attach handler to logger
    logger.addHandler(handler)
    
    # Prevent logs from propagating to parent loggers
    # This avoids duplicate log messages
    logger.propagate = False
    
    return logger


def log_performance(logger: logging.Logger, operation: str, duration_ms: float, 
                   metadata: Dict[str, Any] = None):
    """
    Convenience function for logging performance metrics.
    
    This creates a consistent format for performance logs, making them
    easier to aggregate and analyze.
    
    Args:
        logger: Logger instance to use
        operation: Name of the operation (e.g., "lane_detection")
        duration_ms: How long it took in milliseconds
        metadata: Additional context (e.g., {"frame_id": 123})
        
    Example:
        logger = get_logger(__name__)
        start = time.time()
        process_frame()
        duration = (time.time() - start) * 1000
        log_performance(logger, "process_frame", duration, {"frame_id": 123})
    """
    # Build log message
    extra = {
        'operation': operation,
        'duration_ms': round(duration_ms, 2),
        'metric_type': 'performance'  # Tag for filtering performance logs
    }
    
    # Add any additional metadata
    if metadata:
        extra.update(metadata)
    
    # Log at INFO level with structured data
    logger.info(
        f"Operation '{operation}' completed in {duration_ms:.2f}ms",
        extra=extra
    )


def log_error(logger: logging.Logger, error: Exception, context: Dict[str, Any] = None):
    """
    Convenience function for logging errors with context.
    
    Args:
        logger: Logger instance to use
        error: The exception that occurred
        context: Additional context about where/why error occurred
        
    Example:
        try:
            process_frame(frame)
        except Exception as e:
            log_error(logger, e, {"frame_id": 123, "step": "lane_detection"})
    """
    # Build structured error log
    extra = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'metric_type': 'error'
    }
    
    # Add context if provided
    if context:
        extra.update(context)
    
    # Log at ERROR level with full stack trace
    # exc_info=True tells logger to include the full exception traceback
    logger.error(
        f"Error occurred: {type(error).__name__}: {str(error)}",
        exc_info=True,
        extra=extra
    )


# Example usage demonstration (commented out for production)
"""
if __name__ == "__main__":
    # Get a logger
    logger = get_logger(__name__)
    
    # Basic logging
    logger.debug("This is a debug message")
    logger.info("System started successfully")
    logger.warning("Configuration value missing, using default")
    logger.error("Failed to connect to Redis")
    
    # Logging with extra context
    logger.info("Frame processed", extra={
        'frame_id': 42,
        'fps': 30.5,
        'lane_confidence': 0.87
    })
    
    # Performance logging
    import time
    start = time.time()
    time.sleep(0.1)  # Simulate work
    log_performance(logger, "test_operation", (time.time() - start) * 1000)
    
    # Error logging
    try:
        1 / 0
    except Exception as e:
        log_error(logger, e, {"operation": "division", "numerator": 1})
"""
