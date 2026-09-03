"""
Simple Pipeline Mode - No Redis, No Complexity

What this teaches:
- When to choose simplicity over scalability
- Direct inference vs async queuing
- Trade-offs in system design

Use Cases:
- Learning/education (focus on ML, not infrastructure)
- Development/debugging (faster iteration)
- Single-user applications
- Edge devices with limited resources

Toggle via environment:
    export SIMPLE_PIPELINE=true
"""

from typing import Dict
import numpy as np

from src.perception.lane_detector import ProductionLaneDetector
from src.perception.sign_detector import ONNXSignDetector
from src.control.decision_engine import BasicControlEngine
from src.utils.logger import get_logger


class SimplePipeline:
    """
    Simplified in-process inference pipeline.
    
    No Redis, no workers, no async - just direct inference.
    Perfect for learning the core ML algorithms without
    infrastructure complexity.
    
    Educational:
    - Shows that async/queues are optional optimizations
    - Demonstrates that simpler is often better for learning
    - Proves core algorithms work independently of infrastructure
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize simple pipeline.
        
        Args:
            model_path: Path to ONNX model (optional)
        """
        self.logger = get_logger(__name__)
        self.logger.info("🎓 Simple Pipeline Mode - Optimized for Learning")
        
        # Initialize models (one-time cost)
        self.lane_detector = ProductionLaneDetector(image_shape=(720, 1280))
        
        # Sign detector only if model available
        self.sign_detector = None
        if model_path:
            try:
                self.sign_detector = ONNXSignDetector(model_path)
            except Exception as e:
                self.logger.warning(f"Sign detector unavailable: {e}")
        
        self.control_engine = BasicControlEngine()
        
        self.logger.info("✅ Simple pipeline ready")
    
    def process_frame(self, frame: np.ndarray, vehicle_state: Dict = None) -> Dict:
        """
        Process a single frame end-to-end.
        
        This is the simplest possible working pipeline:
        Frame → Lane Detection → Sign Detection → Decision → Result
        
        Args:
            frame: Input image (H, W, 3)
            vehicle_state: Optional vehicle state dict
            
        Returns:
            Dict with lane results, sign detections, and control event
        """
        # Default vehicle state
        vehicle_state = vehicle_state or {'speed': 0}
        
        # 1. Lane detection (traditional CV)
        lane_result = self.lane_detector.detect(frame)
        
        # 2. Sign detection (deep learning, if available)
        sign_detections = []
        if self.sign_detector:
            sign_detections = self.sign_detector.detect(frame)
        
        # 3. Control decision (rule-based)
        control_event = self.control_engine.generate_control_event(
            lane_result, sign_detections, vehicle_state
        )
        
        # 4. Package results
        return {
            'lane_result': lane_result,
            'sign_detections': sign_detections,
            'control_event': control_event,
            'mode': 'simple'  # Indicator that simple pipeline was used
        }


def should_use_simple_pipeline() -> bool:
    """
    Check if simple pipeline mode should be used.
    
    Returns:
        True if SIMPLE_PIPELINE=true in environment
    """
    import os
    return os.getenv('SIMPLE_PIPELINE', 'false').lower() == 'true'
