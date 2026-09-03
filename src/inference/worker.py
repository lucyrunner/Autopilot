"""
Worker Process for Async Inference

Continuously pulls requests from Redis queue, runs inference, stores results.
Designed to be run as multiple processes for horizontal scaling.

Deployment patterns:
1. Single machine: python -m src.inference.worker --worker-id 0
2. Multi-process: Run 4 workers with different IDs (0-3)
3. Multi-machine: Deploy workers on separate servers
4. Kubernetes: Deploy as StatefulSet with replicas
"""

import time
import signal
import sys
import cv2
import numpy as np
import base64

from src.inference.redis_queue import RedisInferenceQueue
from src.perception.lane_detector import ProductionLaneDetector
from src.perception.sign_detector import ONNXSignDetector
from src.control.decision_engine import BasicControlEngine
from config.settings import get_settings
from src.utils.logger import get_logger, log_performance, log_error


class InferenceWorker:
    """
    Worker process for async ML inference.
    
    Lifecycle:
    1. Initialize models (one-time startup cost)
    2. Connect to Redis queue
    3. Loop: pull request → process → store result
    4. Graceful shutdown on SIGTERM/SIGINT
    """
    
    def __init__(self, worker_id: int = 0):
        """
        Initialize worker.
        
        Args:
            worker_id: Unique identifier for this worker (for logging/debugging)
        """
        self.worker_id = worker_id
        self.logger = get_logger(f"worker-{worker_id}")
        self.settings = get_settings()
        
        # Control flags
        self.running = False
        self.processed_count = 0
        
        # Initialize queue
        self.queue = RedisInferenceQueue()
        
        # Initialize models (heavy operation, done once)
        self.logger.info("Loading models...")
        self._load_models()
        self.logger.info("Models loaded successfully")
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_models(self):
        """
        Load all ML models.
        
        This is done once at startup to avoid repeated loading.
        Models are kept in memory for fast inference.
        """
        try:
            # Lane detector (traditional CV, fast)
            self.lane_detector = ProductionLaneDetector(
                image_shape=(720, 1280)
            )
            
            # Sign detector (deep learning, GPU-accelerated if available)
            self.sign_detector = ONNXSignDetector(
                model_path=self.settings.model.sign_model_path,
                conf_threshold=self.settings.model.sign_confidence_threshold
            )
            
            # Control engine (rule-based, CPU)
            self.control_engine = BasicControlEngine()
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            raise
    
    def _signal_handler(self, signum, frame):
        """
        Handle shutdown signals (SIGTERM, SIGINT).
        
        Allows graceful shutdown:
        - Finish processing current request
        - Clean up resources
        - Log statistics
        """
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def process_request(self, request: dict) -> dict:
        """
        Process a single inference request.
        
        Pipeline:
        1. Decode image from base64
        2. Run lane detection
        3. Run sign detection (parallel with lane if GPU available)
        4. Generate control event
        5. Package results
        
        Args:
            request: Request payload from queue
            
        Returns:
            dict: Inference results
        """
        start_time = time.time()
        request_id = request['request_id']
        
        try:
            # Decode image
            image_bytes = base64.b64decode(request['image_data'])
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if frame is None:
                raise ValueError("Failed to decode image")
            
            # Run perception pipeline
            lane_result = self.lane_detector.detect(frame)
            sign_detections = self.sign_detector.detect(frame)
            
            # Generate control decision
            vehicle_state = request['metadata'].get('vehicle_state', {})
            control_event = self.control_engine.generate_control_event(
                lane_result, sign_detections, vehicle_state
            )
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Package results
            result = {
                'request_id': request_id,
                'lane_result': lane_result,
                'sign_detections': sign_detections,
                'control_event': control_event,
                'processing_time_ms': processing_time,
                'timestamp': time.time(),
                'worker_id': self.worker_id
            }
            
            # Log performance
            log_performance(
                self.logger,
                "inference_request",
                processing_time,
                {'request_id': request_id}
            )
            
            return result
            
        except Exception as e:
            # Log error with context
            log_error(
                self.logger,
                e,
                {'request_id': request_id, 'worker_id': self.worker_id}
            )
            
            # Return error result
            return {
                'request_id': request_id,
                'error': str(e),
                'error_type': type(e).__name__,
                'timestamp': time.time(),
                'worker_id': self.worker_id
            }
    
    def run(self):
        """
        Main worker loop.
        
        Continuously:
        1. Pull request from Redis (blocks if queue empty)
        2. Process request
        3. Store result
        4. Repeat
        
        Exits gracefully on SIGTERM/SIGINT.
        """
        self.running = True
        self.logger.info(f"Worker {self.worker_id} started")
        
        while self.running:
            try:
                # Get next request (blocks up to 5 seconds)
                request = self.queue.get_request(timeout=5)
                
                if request is None:
                    # No requests available, continue waiting
                    continue
                
                self.logger.info(f"Processing request {request['request_id']}")
                
                # Process request
                result = self.process_request(request)
                
                # Store result in Redis
                self.queue.store_result(request['request_id'], result)
                
                self.processed_count += 1
                
                self.logger.info(
                    f"Request {request['request_id']} completed in "
                    f"{result.get('processing_time_ms', 0):.1f}ms "
                    f"(total processed: {self.processed_count})"
                )
                
            except Exception as e:
                self.logger.error(f"Worker error: {e}")
                time.sleep(1)  # Brief pause before retry
        
        self.logger.info(
            f"Worker {self.worker_id} stopped "
            f"(processed {self.processed_count} requests)"
        )


def main():
    """
    Entry point for worker process.
    
    Usage:
        python -m src.inference.worker --worker-id 0
        python -m src.inference.worker --worker-id 1
        ...
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Inference worker process')
    parser.add_argument(
        '--worker-id',
        type=int,
        default=0,
        help='Worker ID (for logging and debugging)'
    )
    args = parser.parse_args()
    
    # Create and run worker
    worker = InferenceWorker(worker_id=args.worker_id)
    worker.run()


if __name__ == "__main__":
    main()
