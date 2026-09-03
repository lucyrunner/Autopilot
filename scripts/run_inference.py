"""
Example: Run inference on a single image

Usage:
    python scripts/run_inference.py --image highway.jpg
"""

import argparse
import cv2
import json
from src.perception.lane_detector import ProductionLaneDetector
from src.perception.sign_detector import ONNXSignDetector
from src.control.decision_engine import BasicControlEngine
from src.ui.visualizer import ProductionVisualizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Input image path')
    parser.add_argument('--output', default='output.jpg', help='Output image path')
    parser.add_argument('--model', default='models/sign_detector/yolov8s.onnx')
    args = parser.parse_args()
    
    # Load image
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Error: Could not load image {args.image}")
        return
    
    # Initialize components
    lane_detector = ProductionLaneDetector(image_shape=frame.shape[:2])
    sign_detector = ONNXSignDetector(args.model)
    control_engine = BasicControlEngine()
    visualizer = ProductionVisualizer()
    
    # Run inference
    lane_result = lane_detector.detect(frame)
    sign_detections = sign_detector.detect(frame)
    control_event = control_engine.generate_control_event(
        lane_result, sign_detections, {'speed': 60}
    )
    
    # Visualize
    vis_frame = visualizer.draw_lanes(frame, lane_result)
    vis_frame = visualizer.draw_signs(vis_frame, sign_detections)
    
    # Save
    cv2.imwrite(args.output, vis_frame)
    
    # Print results
    print(json.dumps({
        'lane_confidence': lane_result['confidence'],
        'signs_detected': len(sign_detections),
        'control_event': control_event['event_type'] if control_event else None
    }, indent=2))


if __name__ == '__main__':
    main()
