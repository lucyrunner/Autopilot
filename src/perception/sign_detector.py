"""
Traffic Sign Detection Module - YOLOv8 Implementation

This module implements real-time traffic sign detection using YOLOv8 via ONNX Runtime.
Unlike lane detection (traditional CV), sign detection requires deep learning due to:
- High variability (40+ sign classes)
- Scale variation (signs can be 32x32 or 500x500 pixels)
- Rotation and perspective distortion
- Partial occlusion (trees, vehicles)

Architecture: YOLOv8 (You Only Look Once v8)
- Single-stage detector: one pass through network
- Anchor-free: no manual anchor box tuning
- Efficient: runs real-time on edge devices

Inference Pipeline:
Input Image → Preprocessing → ONNX Inference → Postprocessing → Detections
"""

import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Dict, Tuple
from config.settings import get_settings
from src.utils.logger import get_logger


class ONNXSignDetector:
    """
    Production traffic sign detector using ONNX Runtime.
    
    ONNX (Open Neural Network Exchange) is a format for ML models that:
    - Works across frameworks (PyTorch, TensorFlow, etc.)
    - Optimized for inference (faster than native PyTorch)
    - Supports multiple backends (CPU, CUDA, TensorRT)
    """
    
    def __init__(
        self, 
        model_path: str,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.4
    ):
        """
        Initialize sign detector.
        
        Args:
            model_path: Path to ONNX model file (.onnx)
            conf_threshold: Confidence threshold (0.0-1.0)
                          Detections below this are filtered out
                          Lower = more detections (but more false positives)
            iou_threshold: IoU threshold for Non-Maximum Suppression
                          Controls how much boxes can overlap
                          Lower = more aggressive filtering
                          
        Example:
            detector = ONNXSignDetector(
                model_path='models/yolov8s.onnx',
                conf_threshold=0.7,  # High confidence only
                iou_threshold=0.4
            )
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Initialize logger
        self.logger = get_logger(__name__)
        
        # Create ONNX Runtime session
        # Execution providers determine where inference runs
        # Priority order: CUDA (GPU) → CPU
        providers = [
            'CUDAExecutionProvider',  # NVIDIA GPU (if available)
            'CPUExecutionProvider'    # CPU fallback
        ]
        
        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.logger.info(f"ONNX model loaded from {model_path}")
            self.logger.info(f"Using provider: {self.session.get_providers()[0]}")
        except Exception as e:
            self.logger.error(f"Failed to load ONNX model: {e}")
            raise
        
        # Get model metadata
        # Input name: what the model expects (e.g., "images")
        # Input shape: expected dimensions (e.g., [1, 3, 640, 640])
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        # YOLOv8 typically expects square input (640x640)
        self.model_height = self.input_shape[2]
        self.model_width = self.input_shape[3]
        
        # Load class names (traffic sign types)
        # TODO: Load from YAML config file
        # For now, using placeholder names
        self.class_names = self._load_class_names()
        
        self.logger.info(
            f"Model expects input: {self.input_shape}, "
            f"Classes: {len(self.class_names)}"
        )
    
    def _load_class_names(self) -> List[str]:
        """
        Load class names for sign categories.
        
        In production, this should load from a YAML config file
        that maps class IDs to sign names.
        
        Returns:
            List[str]: Class names indexed by class ID
        """
        # GTSRB (German Traffic Sign Recognition Benchmark) classes
        # In production, load from: models/sign_detector/classes.yaml
        return [
            'speed_limit_20', 'speed_limit_30', 'speed_limit_50',
            'speed_limit_60', 'speed_limit_70', 'speed_limit_80',
            'end_speed_limit_80', 'speed_limit_100', 'speed_limit_120',
            'no_passing', 'no_passing_trucks', 'right_of_way',
            'priority_road', 'yield', 'stop', 'no_vehicles',
            'no_trucks', 'no_entry', 'general_caution',
            'dangerous_curve_left', 'dangerous_curve_right',
            'double_curve', 'bumpy_road', 'slippery_road',
            'road_narrows_right', 'road_work', 'traffic_signals',
            'pedestrians', 'children_crossing', 'bicycles_crossing',
            'beware_ice_snow', 'wild_animals_crossing', 'end_limits',
            'turn_right_ahead', 'turn_left_ahead', 'ahead_only',
            'go_straight_or_right', 'go_straight_or_left',
            'keep_right', 'keep_left', 'roundabout_mandatory',
            'end_no_passing', 'end_no_passing_trucks'
        ]
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for YOLOv8 inference.
        
        YOLOv8 preprocessing steps:
        1. Resize to model input size (640x640)
        2. Convert BGR to RGB (OpenCV uses BGR, model expects RGB)
        3. Normalize to [0, 1] range (divide by 255)
        4. Transpose to channel-first format (H,W,C → C,H,W)
        5. Add batch dimension (C,H,W → 1,C,H,W)
        
        Args:
            image: Input image in BGR format (H, W, 3)
            
        Returns:
            np.ndarray: Preprocessed tensor ready for inference
                       Shape: (1, 3, model_height, model_width)
        """
        # Step 1: Resize to model input size
        # Use cv2.INTER_LINEAR for good quality/speed balance
        resized = cv2.resize(
            image, 
            (self.model_width, self.model_height),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Step 2: BGR → RGB conversion
        # OpenCV loads images as BGR, but models expect RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Step 3: Normalize to [0, 1]
        # Neural networks work better with normalized inputs
        # Convert to float32 (required by ONNX)
        normalized = rgb.astype(np.float32) / 255.0
        
        # Step 4: Transpose from (H, W, C) to (C, H, W)
        # PyTorch/ONNX convention is channel-first
        # NumPy default is height-width-channel, need to transpose
        transposed = np.transpose(normalized, (2, 0, 1))
        
        # Step 5: Add batch dimension
        # Models expect batches: (batch_size, C, H, W)
        # For single image: (1, C, H, W)
        batched = np.expand_dims(transposed, axis=0)
        
        return batched
    
    def postprocess(
        self, 
        outputs: List[np.ndarray], 
        original_shape: Tuple[int, int]
    ) -> List[Dict]:
        """
        Convert model outputs to bounding boxes with labels.
        
        YOLOv8 output format:
        - Shape: (1, 4+num_classes, num_anchors)
        - First 4 values: bounding box (x_center, y_center, width, height)
        - Remaining values: class probabilities
        - num_anchors: number of detection candidates (~8400 for 640x640)
        
        Postprocessing steps:
        1. Extract boxes and class scores
        2. Get highest class score for each detection
        3. Filter by confidence threshold
        4. Convert box format (center → corners)
        5. Scale boxes to original image size
        6. Apply Non-Maximum Suppression (NMS) to remove duplicates
        
        Args:
            outputs: Raw model outputs (list of numpy arrays)
            original_shape: Original image shape (height, width)
                          Needed to scale boxes back from model size
                          
        Returns:
            List[Dict]: List of detections, each containing:
                - bbox: [x1, y1, x2, y2] in original image coordinates
                - class_name: Sign type (e.g., "stop")
                - class_id: Numeric class identifier
                - confidence: Detection confidence (0.0-1.0)
        """
        # Extract first output (YOLOv8 has one output tensor)
        predictions = outputs[0][0]  # Remove batch dimension
        
        # Transpose to (num_anchors, 4+num_classes)
        # This makes it easier to work with
        predictions = predictions.transpose()
        
        # Split into boxes and class scores
        # boxes: first 4 columns (x_center, y_center, width, height)
        # scores: remaining columns (one per class)
        boxes = predictions[:, :4]
        scores = predictions[:, 4:]
        
        # Get class with highest score for each detection
        # argmax returns class index, max returns the score value
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        # Filter by confidence threshold
        # Only keep detections above threshold
        mask = confidences > self.conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        # Convert box format: (x_center, y_center, w, h) → (x1, y1, x2, y2)
        boxes_xyxy = self._xywh_to_xyxy(boxes)
        
        # Scale boxes from model size to original image size
        boxes_scaled = self._scale_boxes(
            boxes_xyxy, 
            (self.model_height, self.model_width),
            original_shape
        )
        
        # Apply Non-Maximum Suppression (NMS)
        # Removes overlapping boxes (keeps highest confidence)
        keep_indices = self._nms(boxes_scaled, confidences, self.iou_threshold)
        
        # Format final detections
        detections = []
        for idx in keep_indices:
            x1, y1, x2, y2 = boxes_scaled[idx]
            conf = confidences[idx]
            class_id = class_ids[idx]
            
            # Get class name (handle out-of-range indices)
            class_name = (
                self.class_names[class_id] 
                if class_id < len(self.class_names) 
                else f"unknown_{class_id}"
            )
            
            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(conf),
                'class_id': int(class_id),
                'class_name': class_name
            })
        
        return detections
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Run inference on a single image.
        
        This is the main method to call for sign detection.
        
        Args:
            image: Input image in BGR format (H, W, 3)
            
        Returns:
            List[Dict]: List of detected signs with bounding boxes
            
        Example:
            detector = ONNXSignDetector('yolov8s.onnx')
            image = cv2.imread('highway.jpg')
            detections = detector.detect(image)
            
            for det in detections:
                print(f"Detected {det['class_name']} with "
                      f"{det['confidence']:.2%} confidence")
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        """
        original_shape = image.shape[:2]
        
        # Preprocess image
        input_tensor = self.preprocess(image)
        
        # Run inference
        # session.run() returns list of outputs
        outputs = self.session.run(
            self.output_names, 
            {self.input_name: input_tensor}
        )
        
        # Postprocess outputs to get detections
        detections = self.postprocess(outputs, original_shape)
        
        return detections
    
    def _xywh_to_xyxy(self, boxes: np.ndarray) -> np.ndarray:
        """
        Convert boxes from (x_center, y_center, width, height) to (x1, y1, x2, y2).
        
        x1, y1 = top-left corner
        x2, y2 = bottom-right corner
        
        Args:
            boxes: Array of boxes in xywh format, shape (N, 4)
            
        Returns:
            np.ndarray: Boxes in xyxy format, shape (N, 4)
        """
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1 = x_center - width/2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1 = y_center - height/2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2 = x_center + width/2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2 = y_center + height/2
        return boxes_xyxy
    
    def _scale_boxes(
        self, 
        boxes: np.ndarray, 
        from_shape: Tuple[int, int], 
        to_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Scale bounding boxes from one image size to another.
        
        Why needed: Model works with fixed size (640x640), but original
        image might be different size (1280x720). Need to scale predictions
        back to original coordinates.
        
        Args:
            boxes: Boxes to scale, shape (N, 4) in xyxy format
            from_shape: Source image size (height, width)
            to_shape: Target image size (height, width)
            
        Returns:
            np.ndarray: Scaled boxes
        """
        from_h, from_w = from_shape
        to_h, to_w = to_shape
        
        # Calculate scale factors
        scale_y = to_h / from_h
        scale_x = to_w / from_w
        
        # Scale boxes
        boxes_scaled = boxes.copy()
        boxes_scaled[:, [0, 2]] *= scale_x  # Scale x coordinates
        boxes_scaled[:, [1, 3]] *= scale_y  # Scale y coordinates
        
        return boxes_scaled
    
    def _nms(
        self, 
        boxes: np.ndarray, 
        scores: np.ndarray, 
        iou_threshold: float
    ) -> np.ndarray:
        """
        Non-Maximum Suppression (NMS) to remove duplicate detections.
        
        Problem: Object detectors often produce multiple overlapping boxes
        for the same object. NMS keeps only the best one.
        
        Algorithm:
        1. Sort boxes by confidence (highest first)
        2. Take box with highest confidence
        3. Remove all boxes that overlap significantly (IoU > threshold)
        4. Repeat with remaining boxes
        
        IoU (Intersection over Union):
        - Measures overlap between two boxes
        - 0.0 = no overlap, 1.0 = perfect overlap
        - Threshold of 0.4 means boxes with >40% overlap are considered duplicates
        
        Args:
            boxes: Bounding boxes in xyxy format, shape (N, 4)
            scores: Confidence scores, shape (N,)
            iou_threshold: IoU threshold for considering boxes as duplicates
            
        Returns:
            np.ndarray: Indices of boxes to keep
        """
        # Use OpenCV's built-in NMS (faster than pure Python)
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            score_threshold=self.conf_threshold,
            nms_threshold=iou_threshold
        )
        
        # OpenCV returns indices as nested array, flatten it
        return indices.flatten() if len(indices) > 0 else np.array([])
