"""
Redis-Based Inference Queue

Implements asynchronous inference using Redis as a message queue.
This allows horizontal scaling - add more workers to process more requests.

Architecture:
Client → API → Redis Queue → Worker Pool (scalable) → Redis Results → Client

Benefits:
- Horizontal scalability: Add workers without changing code
- Decoupling: API doesn't wait for inference
- Resilience: Requests survive worker crashes
- Load balancing: Redis distributes fairly
"""

import redis
import json
import uuid
import time
import pickle
import base64
from typing import Optional, Dict, Any
from config.settings import get_settings
from src.utils.logger import get_logger


class RedisInferenceQueue:
    """
    Redis-based async queue for ML inference requests.
    
    Uses Redis data structures:
    - List (LPUSH/BRPOP): FIFO queue for requests
    - String (SET/GET): Results storage with TTL
    - String: Status tracking (queued/processing/completed)
    
    TTL (Time To Live):
    - Requests: 5 minutes (auto-cleanup of abandoned requests)
    - Results: 10 minutes (cached for client retrieval)
    """
    
    def __init__(self):
        """Initialize Redis connection from settings."""
        settings = get_settings()
        self.logger = get_logger(__name__)
        
        # Connect to Redis
        try:
            self.redis_client = redis.Redis(
                host=settings.redis.host,
                port=settings.redis.port,
                db=settings.redis.db,
                password=settings.redis.password,
                decode_responses=False  # Binary mode for pickle
            )
            
            # Test connection
            self.redis_client.ping()
            self.logger.info(f"Connected to Redis at {settings.redis.host}:{settings.redis.port}")
        except redis.ConnectionError as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            raise
        
        # Queue and key prefixes
        self.request_queue = settings.redis.inference_queue
        self.result_prefix = "inference:result:"
        self.status_prefix = "inference:status:"
        
        # TTL settings
        self.request_ttl = settings.redis.request_ttl
        self.result_ttl = settings.redis.result_ttl
    
    def submit_request(self, image_data: bytes, metadata: Dict = None) -> str:
        """
        Submit inference request to queue.
        
        Flow:
        1. Generate unique request ID (UUID)
        2. Create payload with image and metadata
        3. Serialize with pickle (handles numpy arrays)
        4. Push to Redis list (LPUSH = add to left)
        5. Set status to "queued"
        
        Args:
            image_data: Raw image bytes (JPEG, PNG, etc.)
            metadata: Optional metadata (vehicle state, timestamp, etc.)
            
        Returns:
            str: Unique request ID for tracking
        """
        request_id = str(uuid.uuid4())
        
        # Build payload
        payload = {
            'request_id': request_id,
            'image_data': base64.b64encode(image_data).decode('utf-8'),
            'metadata': metadata or {},
            'timestamp': time.time()
        }
        
        # Serialize (pickle handles complex Python objects)
        serialized = pickle.dumps(payload)
        
        # Push to queue (LPUSH = left push, FIFO with BRPOP)
        self.redis_client.lpush(self.request_queue, serialized)
        
        # Set initial status with TTL
        self.redis_client.setex(
            f"{self.status_prefix}{request_id}",
            self.request_ttl,
            "queued"
        )
        
        self.logger.debug(f"Request {request_id} queued")
        return request_id
    
    def get_request(self, timeout: int = 5) -> Optional[Dict]:
        """
        Get next request from queue (blocking call).
        
        This is called by worker processes.
        BRPOP = Blocking Right Pop (blocks until item available or timeout)
        
        Args:
            timeout: Seconds to wait for request (0 = wait forever)
            
        Returns:
            Dict: Request payload or None if timeout
        """
        # BRPOP blocks until data available
        result = self.redis_client.brpop(self.request_queue, timeout=timeout)
        
        if result is None:
            return None
        
        # result = (queue_name, data)
        _, serialized = result
        payload = pickle.loads(serialized)
        
        # Update status to processing
        request_id = payload['request_id']
        self.redis_client.setex(
            f"{self.status_prefix}{request_id}",
            self.request_ttl,
            "processing"
        )
        
        self.logger.debug(f"Request {request_id} dequeued for processing")
        return payload
    
    def store_result(self, request_id: str, result: Dict):
        """
        Store inference result.
        
        Args:
            request_id: Request identifier
            result: Inference results (lanes, signs, control event)
        """
        # Serialize result
        serialized = pickle.dumps(result)
        
        # Store with TTL (auto-expires after 10 minutes)
        self.redis_client.setex(
            f"{self.result_prefix}{request_id}",
            self.result_ttl,
            serialized
        )
        
        # Update status
        self.redis_client.setex(
            f"{self.status_prefix}{request_id}",
            self.result_ttl,
            "completed"
        )
        
        self.logger.debug(f"Result stored for {request_id}")
    
    def get_result(self, request_id: str, timeout: int = 30) -> Optional[Dict]:
        """
        Get inference result (with polling).
        
        Client-side method - polls Redis until result ready or timeout.
        
        Args:
            request_id: Request identifier
            timeout: Maximum seconds to wait
            
        Returns:
            Dict: Inference result or None if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            result_key = f"{self.result_prefix}{request_id}"
            serialized = self.redis_client.get(result_key)
            
            if serialized:
                return pickle.loads(serialized)
            
            # Check if request expired
            status = self.get_status(request_id)
            if status is None:
                self.logger.warning(f"Request {request_id} not found (expired?)")
                return None
            
            # Poll every 100ms
            time.sleep(0.1)
        
        self.logger.warning(f"Timeout waiting for result: {request_id}")
        return None
    
    def get_status(self, request_id: str) -> Optional[str]:
        """
        Get current request status.
        
        Returns:
            str: "queued", "processing", "completed", or None (expired)
        """
        status_key = f"{self.status_prefix}{request_id}"
        status = self.redis_client.get(status_key)
        return status.decode('utf-8') if status else None
    
    def queue_length(self) -> int:
        """
        Get current queue depth.
        
        Useful for monitoring and auto-scaling decisions.
        High queue depth = need more workers.
        """
        return self.redis_client.llen(self.request_queue)
    
    def clear_queue(self):
        """
        Clear all pending requests (emergency use only).
        
        WARNING: This deletes all queued requests!
        Use only during maintenance or debugging.
        """
        deleted = self.redis_client.delete(self.request_queue)
        self.logger.warning(f"Cleared {deleted} requests from queue")
        return deleted
