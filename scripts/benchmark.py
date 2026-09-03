"""
Benchmark inference performance

Measures:
- Latency (mean, P95, P99)
- Throughput (FPS)
- Memory usage
"""

import time
import numpy as np
import cv2
from src.perception.lane_detector import ProductionLaneDetector
from src.perception.sign_detector import ONNXSignDetector


def benchmark_lane_detection(num_iterations=100):
    detector = ProductionLaneDetector(image_shape=(720, 1280))
    image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    latencies = []
    for _ in range(num_iterations):
        start = time.time()
        detector.detect(image)
        latencies.append((time.time() - start) * 1000)
    
    print(f"Lane Detection:")
    print(f"  Mean: {np.mean(latencies):.2f}ms")
    print(f"  P95:  {np.percentile(latencies, 95):.2f}ms")
    print(f"  P99:  {np.percentile(latencies, 99):.2f}ms")
    print(f"  FPS:  {1000/np.mean(latencies):.1f}")


def benchmark_sign_detection(model_path, num_iterations=100):
    detector = ONNXSignDetector(model_path)
    image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    latencies = []
    for _ in range(num_iterations):
        start = time.time()
        detector.detect(image)
        latencies.append((time.time() - start) * 1000)
    
    print(f"\nSign Detection:")
    print(f"  Mean: {np.mean(latencies):.2f}ms")
    print(f"  P95:  {np.percentile(latencies, 95):.2f}ms")
    print(f"  P99:  {np.percentile(latencies, 99):.2f}ms")
    print(f"  FPS:  {1000/np.mean(latencies):.1f}")


if __name__ == '__main__':
    benchmark_lane_detection()
    # benchmark_sign_detection('models/sign_detector/yolov8s.onnx')
