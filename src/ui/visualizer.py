"""
Visual Overlay System

Renders detection results on video frames for:
- Debugging
- User feedback
- Demo visualization
"""

import cv2
import numpy as np
from typing import Dict, List
import time


class ProductionVisualizer:
    """
    Visualization system for perception outputs.
    """
    
    def __init__(self):
        self.frame_count = 0
        self.fps = 0.0
        self.last_time = time.time()
    
    def draw_lanes(self, frame: np.ndarray, lane_result: Dict) -> np.ndarray:
        """Draw lane overlays on frame."""
        overlay = frame.copy()
        
        left_lane = lane_result.get('left_lane')
        right_lane = lane_result.get('right_lane')
        
        if left_lane is None or right_lane is None:
            cv2.putText(overlay, "LANE NOT DETECTED", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return overlay
        
        height = frame.shape[0]
        y_coords = np.linspace(height, height * 0.6, num=50)
        
        # Evaluate polynomials
        left_x = np.polyval(left_lane, y_coords).astype(int)
        right_x = np.polyval(right_lane, y_coords).astype(int)
        
        # Draw lane lines
        left_points = np.array([np.transpose(np.vstack([left_x, y_coords]))], dtype=np.int32)
        right_points = np.array([np.transpose(np.vstack([right_x, y_coords]))], dtype=np.int32)
        
        cv2.polylines(overlay, left_points, False, (0, 255, 0), 5)
        cv2.polylines(overlay, right_points, False, (0, 0, 255), 5)
        
        # Blend with original
        result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        return result
    
    def draw_signs(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes around detected signs."""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class_name']
            
            # Color based on sign type
            if 'stop' in class_name or 'yield' in class_name:
                color = (0, 0, 255)  # Red for critical signs
            elif 'speed_limit' in class_name:
                color = (255, 0, 0)  # Blue for speed limits
            else:
                color = (0, 255, 255)  # Yellow for others
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
