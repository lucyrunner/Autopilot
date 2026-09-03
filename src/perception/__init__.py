"""
Perception Package - Computer Vision & Deep Learning

═══════════════════════════════════════════════════════════════════════

📚 WHAT THIS MODULE TEACHES:

1. **Traditional Computer Vision** (lane_detector.py)
   - When to use classical CV vs deep learning
   - Canny edge detection mathematics
   - Hough Transform voting mechanism
   - Polynomial curve fitting
   - Why it's 10x faster than deep learning for lanes

2. **Deep Learning Inference** (sign_detector.py)
   - YOLOv8 architecture overview
   - ONNX Runtime optimization
   - NMS (Non-Maximum Suppression) algorithm
   - Production model deployment patterns
   - GPU vs CPU trade-offs

3. **Image Preprocessing** (preprocessor.py)
   - Multiple normalization strategies
   - Letterbox vs crop vs stretch
   - Augmentation for robustness
   - When to preprocess vs when to train for invariance

═══════════════════════════════════════════════════════════════════════

🎯 LEARNING PATH:

Beginner → intermediate → advanced:
1. Start with lane_detector.py (traditional CV, easier to debug)
2. Then sign_detector.py (deep learning, more complex)
3. Finally preprocessor.py (advanced techniques)

═══════════════════════════════════════════════════════════════════════

📖 KEY CONCEPTS EXPLAINED HERE:

- **Canny Edge Detection**: Finds rapid changes in pixel intensity
- **Hough Transform**: Voting-based line detection in parameter space
- **ONNX**: Cross-platform ML model format for production
- **YOLOv8**: Single-stage object detector (fast + accurate)
- **NMS**: Removes duplicate detections based on IoU overlap

═══════════════════════════════════════════════════════════════════════
"""

from .lane_detector import ProductionLaneDetector
from .sign_detector import ONNXSignDetector
from .preprocessor import ImagePreprocessor

__all__ = [
    'ProductionLaneDetector',
    'ONNXSignDetector',
    'ImagePreprocessor'
]
