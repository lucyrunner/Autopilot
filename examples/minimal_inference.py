#!/usr/bin/env python3
"""
Minimal Inference Example - 30 Lines End-to-End

This is the simplest possible working example.
Learn: Image → Preprocessing → Model → Decision → Result

Run: python examples/minimal_inference.py
"""

import cv2
import numpy as np
from src.perception.lane_detector import ProductionLaneDetector
from src.control.decision_engine import BasicControlEngine

def main():
    # 1. Load test image (or create synthetic)
    image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    # Real usage: image = cv2.imread('highway.jpg')
    
    # 2. Initialize detector
    detector = ProductionLaneDetector(image_shape=(720, 1280))
    
    # 3. Run detection
    result = detector.detect(image)
    
    # 4. Make decision
    engine = BasicControlEngine()
    vehicle_state = {'speed': 65}
    event = engine.generate_control_event(result, [], vehicle_state)
    
    # 5. Print results
    print(f"✓ Lane confidence: {result['confidence']:.2%}")
    print(f"✓ Control: {event['event_type'] if event else 'KEEP_LANE'}")
    print(f"✓ Success! System working end-to-end.")

if __name__ == '__main__':
    main()

# At the start of main():
def main():
    # Set seed for reproducibility
    from src.utils.reproducibility import set_global_seed
    set_global_seed(42)
    
    # ... rest of code ...
