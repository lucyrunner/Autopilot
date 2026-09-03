"""
Configuration Management System for Autopilot Vision

This module provides centralized configuration using Pydantic Settings.
All configuration values can be overridden via environment variables.

Design Pattern: Singleton (only one Settings instance exists)
Why: Ensures consistent configuration across entire application
"""

from pydantic_settings import BaseSettings
from typing import Optional, List
from functools import lru_cache


class ModelConfig(BaseSettings):
    """
    Configuration for machine learning models.
    
    These settings control the behavior of lane detection and sign detection models.
    Tuning these parameters affects accuracy/speed trade-offs.
    """
    
    # Lane Detection Parameters
    # -------------------------
    # Canny edge detection thresholds (lower = more sensitive to edges)
    lane_canny_low: int = 50
    lane_canny_high: int = 150
    
    # Hough Transform parameters
    # Higher threshold = fewer but stronger line detections
    lane_hough_threshold: int = 50
    lane_min_line_length: int = 100  # Minimum pixels for valid lane line
    lane_max_line_gap: int = 50      # Maximum gap to connect line segments
    
    # Sign Detection Parameters
    # -------------------------
    # Path to the ONNX model file (optimized for deployment)
    sign_model_path: str = "models/sign_detector/yolov8s.onnx"
    
    # Confidence threshold: detections below this are filtered out
    # Lower = more detections (but more false positives)
    # Higher = fewer detections (but may miss signs)
    sign_confidence_threshold: float = 0.5
    
    # IoU threshold for Non-Maximum Suppression (removes overlapping boxes)
    # Lower = more aggressive filtering of overlaps
    sign_iou_threshold: float = 0.4
    
    # Input image size for sign detection model (must match training)
    sign_input_size: int = 640
    
    # Temporal Smoothing Parameters
    # -----------------------------
    # Alpha for exponential moving average (higher = more weight on new data)
    # Range: 0.0 (ignore new data) to 1.0 (ignore history)
    temporal_alpha: float = 0.2
    
    # Number of frames to keep in history for temporal smoothing
    temporal_memory_frames: int = 10
    
    class Config:
        # Environment variables must be prefixed with MODEL_
        # Example: MODEL_SIGN_CONFIDENCE_THRESHOLD=0.7
        env_prefix = "MODEL_"


class APIConfig(BaseSettings):
    """
    Configuration for FastAPI server.
    
    Controls network settings, CORS, and rate limiting for the API.
    """
    
    # Network settings
    host: str = "0.0.0.0"  # 0.0.0.0 means accept connections from any IP
    port: int = 8000
    
    # Number of worker processes (for production, set to CPU cores)
    # In development with reload=True, this must be 1
    workers: int = 4
    
    # Auto-reload on code changes (development only)
    reload: bool = False
    
    # CORS (Cross-Origin Resource Sharing) settings
    # List of origins allowed to call the API
    # Example: ["http://localhost:3000", "https://myapp.com"]
    cors_origins: List[str] = ["http://localhost:3000"]
    
    # Rate limiting (prevent abuse)
    rate_limit_requests: int = 100  # Max requests
    rate_limit_window: int = 60     # Per time window (seconds)
    
    class Config:
        env_prefix = "API_"


class RedisConfig(BaseSettings):
    """
    Configuration for Redis message queue.
    
    Redis is used for asynchronous inference queue to handle multiple
    concurrent requests without blocking.
    """
    
    # Redis connection settings
    host: str = "localhost"
    port: int = 6379
    db: int = 0  # Redis database number (0-15)
    password: Optional[str] = None  # Set if Redis requires authentication
    
    # Queue names (different queues for different purposes)
    inference_queue: str = "inference_requests"
    results_queue: str = "inference_results"
    
    # TTL (Time To Live) settings - how long data persists
    request_ttl: int = 300  # 5 minutes - old requests are cleaned up
    result_ttl: int = 600   # 10 minutes - results cached for retrieval
    
    class Config:
        env_prefix = "REDIS_"


class LoggingConfig(BaseSettings):
    """
    Configuration for application logging.
    
    Structured logging helps with debugging and monitoring in production.
    """
    
    # Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    # DEBUG: Everything (very verbose)
    # INFO: General information about system operation
    # WARNING: Something unexpected happened
    # ERROR: Serious problem, function failed
    # CRITICAL: System may be unable to continue
    level: str = "INFO"
    
    # Log format: "json" for structured logging or "text" for human-readable
    # JSON is better for log aggregation tools (ELK, Splunk)
    format: str = "json"
    
    # Output destination: "stdout" or file path
    output: str = "stdout"
    
    # Structured logging fields (added to every log message)
    service_name: str = "autopilot-vision"
    environment: str = "development"  # development, staging, production
    
    class Config:
        env_prefix = "LOG_"


class Settings(BaseSettings):
    """
    Master settings class combining all configuration sections.
    
    This is the single source of truth for all application configuration.
    Access pattern: settings.model.lane_canny_low
    """
    
    # Sub-configurations
    model: ModelConfig = ModelConfig()
    api: APIConfig = APIConfig()
    redis: RedisConfig = RedisConfig()
    logging: LoggingConfig = LoggingConfig()
    
    # Application metadata
    app_name: str = "Autopilot Vision System"
    version: str = "1.0.0"
    debug: bool = False
    
    # Hardware configuration
    # Options: "cpu", "cuda" (NVIDIA GPU), "mps" (Apple Silicon)
    device: str = "cuda"
    
    class Config:
        # Load from .env file if present
        env_file = ".env"
        env_file_encoding = "utf-8"


# Singleton instance using LRU cache
# @lru_cache ensures Settings is only instantiated once
# This prevents re-reading config files multiple times
@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings (singleton pattern).
    
    Returns:
        Settings: Single shared instance of application settings
        
    Example:
        from config.settings import get_settings
        settings = get_settings()
        print(settings.api.port)  # Access configuration values
    """
    return Settings()
