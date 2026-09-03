"""
Lane Detection Module - Production Implementation

This module implements lane detection using traditional computer vision techniques.
Unlike deep learning approaches, this is deterministic, fast, and interpretable.

Algorithm Pipeline:
1. Gaussian Blur → Remove noise
2. Canny Edge Detection → Find edges
3. Region of Interest (ROI) → Focus on road area
4. Hough Transform → Detect lines
5. Line Separation → Separate left/right lanes
6. Polynomial Fitting → Fit smooth curves
7. Temporal Smoothing → Reduce jitter across frames

Why traditional CV for lanes?
- Fast: 200+ FPS on CPU (leaves GPU for sign detection)
- Deterministic: Same input always produces same output
- Interpretable: Can debug exactly why a line was detected
- Robust: Works well in 80% of scenarios (highway, good markings)
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
from collections import deque
from config.settings import get_settings
from src.utils.logger import get_logger


class ProductionLaneDetector:
    """
    Production-grade lane detection system.
    
    Detects left and right lane boundaries from video frames and returns:
    - Polynomial coefficients for each lane
    - Confidence score (0.0 to 1.0)
    - Road curvature in meters
    """
    
    def __init__(self, image_shape: Tuple[int, int]):
        """
        Initialize lane detector.
        
        Args:
            image_shape: Expected input image shape as (height, width)
                        Example: (720, 1280) for 720p video
        """
        self.image_shape = image_shape
        self.height, self.width = image_shape
        
        # Load configuration
        self.settings = get_settings()
        
        # Hyperparameters from config
        self.canny_low = self.settings.model.lane_canny_low
        self.canny_high = self.settings.model.lane_canny_high
        self.hough_threshold = self.settings.model.lane_hough_threshold
        self.min_line_length = self.settings.model.lane_min_line_length
        self.max_line_gap = self.settings.model.lane_max_line_gap
        
        # Create ROI mask once (reused for all frames)
        self.roi_mask = self._create_roi_mask()
        
        # Temporal smoothing: exponential moving average
        # Keeps track of recent lane detections to smooth jitter
        self.left_lane_history = deque(maxlen=self.settings.model.temporal_memory_frames)
        self.right_lane_history = deque(maxlen=self.settings.model.temporal_memory_frames)
        self.alpha = self.settings.model.temporal_alpha  # Smoothing factor
        
        # Current smoothed lane coefficients
        self.smoothed_left = None
        self.smoothed_right = None
        
        # Logger for debugging and monitoring
        self.logger = get_logger(__name__)
        
        self.logger.info(f"Lane detector initialized for {image_shape} images")
    
    def _create_roi_mask(self) -> np.ndarray:
        """
        Create Region of Interest (ROI) mask.
        
        The ROI is a trapezoid that focuses on the road ahead:
        - Excludes sky (upper portion)
        - Excludes vehicle hood (very bottom)
        - Excludes far left/right (adjacent lanes irrelevant)
        
        Why trapezoid?
        - Matches perspective projection of road
        - Parallel lanes converge toward horizon
        
        Returns:
            np.ndarray: Binary mask (255 inside ROI, 0 outside)
        """
        # Create blank mask
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Define trapezoid vertices
        # These percentages work well for highway driving with forward-facing camera
        vertices = np.array([[
            (int(self.width * 0.1), self.height),              # Bottom-left
            (int(self.width * 0.45), int(self.height * 0.6)),  # Top-left (horizon)
            (int(self.width * 0.55), int(self.height * 0.6)),  # Top-right (horizon)
            (int(self.width * 0.9), self.height)               # Bottom-right
        ]], dtype=np.int32)
        
        # Fill polygon with white (255)
        cv2.fillPoly(mask, vertices, 255)
        
        return mask
    
    def detect(self, frame: np.ndarray) -> Dict:
        """
        Detect lanes in a single frame.
        
        This is the main method called for each video frame.
        
        Args:
            frame: Input RGB frame (H, W, 3) as numpy array
            
        Returns:
            dict: Dictionary containing:
                - left_lane: Polynomial coefficients [a, b, c] for left lane
                            (y = ax² + bx + c) or None if not detected
                - right_lane: Polynomial coefficients for right lane or None
                - confidence: Detection confidence (0.0 to 1.0)
                - curvature: Road curvature in meters or None
                
        Example:
            detector = ProductionLaneDetector(image_shape=(720, 1280))
            frame = cv2.imread('highway.jpg')
            result = detector.detect(frame)
            
            if result['confidence'] > 0.7:
                print(f"Lane detected with {result['confidence']:.2%} confidence")
                print(f"Road curvature: {result['curvature']:.1f}m")
        """
        # Stage 1: Preprocess - convert to grayscale and blur
        # Grayscale: Lane markings are intensity-based (white on dark asphalt)
        # Blur: Reduces noise that would create false edges
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Stage 2: Canny Edge Detection
        # Finds pixels where intensity changes rapidly (edges)
        # Returns binary image: 255 = edge, 0 = no edge
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        # Stage 3: Apply ROI mask
        # Bitwise AND: keeps edges only inside ROI, zeros out everything else
        roi_edges = cv2.bitwise_and(edges, self.roi_mask)
        
        # Stage 4: Detect lines using Hough Transform
        # Hough Transform finds lines by voting in parameter space
        # Returns array of line segments: [[x1, y1, x2, y2], ...]
        lines = cv2.HoughLinesP(
            roi_edges,
            rho=1,                          # Distance resolution (pixels)
            theta=np.pi/180,                # Angle resolution (radians)
            threshold=self.hough_threshold, # Minimum votes to be a line
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        # Handle case where no lines detected
        if lines is None:
            return self._return_no_detection()
        
        # Stage 5: Separate left and right lane lines
        left_lines, right_lines = self._separate_lanes(lines)
        
        # Stage 6: Fit polynomial curves through line segments
        left_coeffs = self._fit_polynomial(left_lines)
        right_coeffs = self._fit_polynomial(right_lines)
        
        # Stage 7: Apply temporal smoothing (exponential moving average)
        smooth_left, smooth_right = self._apply_temporal_smoothing(left_coeffs, right_coeffs)
        
        # Calculate confidence based on detection consistency
        confidence = self._calculate_confidence()
        
        # Calculate road curvature if both lanes detected
        curvature = self._calculate_curvature(smooth_left, smooth_right)
        
        return {
            'left_lane': smooth_left,
            'right_lane': smooth_right,
            'confidence': confidence,
            'curvature': curvature
        }
    
    def _return_no_detection(self) -> Dict:
        """
        Return result structure when no lanes detected.
        
        Returns:
            dict: Result with None values and zero confidence
        """
        return {
            'left_lane': None,
            'right_lane': None,
            'confidence': 0.0,
            'curvature': None
        }
    
    def _separate_lanes(self, lines: np.ndarray) -> Tuple[List, List]:
        """
        Separate line segments into left and right lanes.
        
        Classification criteria:
        - Left lane: Negative slope, on left half of image
        - Right lane: Positive slope, on right half of image
        - Ignore near-horizontal lines (likely road edges, not lanes)
        
        Args:
            lines: Array of line segments [[x1, y1, x2, y2], ...]
            
        Returns:
            tuple: (left_lines, right_lines) as lists of line segments
        """
        left_lines = []
        right_lines = []
        
        # Image center for left/right classification
        middle_x = self.width / 2
        
        for line in lines:
            for x1, y1, x2, y2 in line:
                # Skip vertical lines (would cause division by zero)
                if x2 - x1 == 0:
                    continue
                
                # Calculate slope: (y2 - y1) / (x2 - x1)
                slope = (y2 - y1) / (x2 - x1)
                
                # Filter out near-horizontal lines (|slope| < 0.5)
                # These are likely road edges, shadows, or noise
                if abs(slope) < 0.5:
                    continue
                
                # Calculate line center point
                center_x = (x1 + x2) / 2
                
                # Classify based on slope and position
                # Left lane: negative slope (goes up-left), left side of image
                if slope < 0 and center_x < middle_x:
                    left_lines.append([[x1, y1, x2, y2]])
                
                # Right lane: positive slope (goes up-right), right side of image
                elif slope > 0 and center_x > middle_x:
                    right_lines.append([[x1, y1, x2, y2]])
        
        return left_lines, right_lines
    
    def _fit_polynomial(self, lines: List, degree: int = 2) -> Optional[np.ndarray]:
        """
        Fit a polynomial curve through line segments.
        
        Real roads curve, so we use a 2nd-order polynomial (parabola):
        y = ax² + bx + c
        
        Note: We actually fit x as a function of y (x = ay² + by + c)
        because lanes are nearly vertical in image space.
        Fitting y(x) would fail for steep lanes (multi-valued function).
        
        Args:
            lines: List of line segments for one lane
            degree: Polynomial degree (2 = quadratic for curves)
            
        Returns:
            np.ndarray: Polynomial coefficients [a, b, c] or None if insufficient data
        """
        if not lines or len(lines) == 0:
            return None
        
        # Extract all (x, y) points from line segments
        x_coords = []
        y_coords = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
        
        # Need at least 3 points for quadratic fit
        if len(x_coords) < 3:
            return None
        
        # Fit polynomial: x = f(y)
        # np.polyfit uses least squares to find best-fit polynomial
        # Returns coefficients in descending order [a, b, c]
        try:
            coeffs = np.polyfit(y_coords, x_coords, degree)
            return coeffs
        except np.linalg.LinAlgError:
            # Polyfit can fail if points are collinear or other edge cases
            return None
    
    def _apply_temporal_smoothing(
        self, 
        left_coeffs: Optional[np.ndarray], 
        right_coeffs: Optional[np.ndarray]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Apply exponential moving average for temporal smoothing.
        
        Single-frame detections can be jittery due to:
        - Shadows crossing lanes
        - Worn or faded markings
        - Small debris on road
        
        Temporal smoothing averages across recent frames to:
        - Reduce jitter (smoother lane overlays)
        - Increase stability (less oscillation in control commands)
        - Fill in brief gaps (if one frame fails, use recent history)
        
        Exponential Moving Average (EMA):
        new_value = alpha * current + (1 - alpha) * previous
        
        Alpha = 0.2 means:
        - 20% weight on current frame
        - 80% weight on historical average
        
        Args:
            left_coeffs: Current frame left lane coefficients
            right_coeffs: Current frame right lane coefficients
            
        Returns:
            tuple: (smoothed_left, smoothed_right) coefficients
        """
        # Update history with new detections
        if left_coeffs is not None:
            self.left_lane_history.append(left_coeffs)
        
        if right_coeffs is not None:
            self.right_lane_history.append(right_coeffs)
        
        # Apply EMA to left lane
        if left_coeffs is not None:
            if self.smoothed_left is None:
                # First detection: initialize with current value
                self.smoothed_left = left_coeffs
            else:
                # EMA formula: blend current with historical
                self.smoothed_left = (
                    self.alpha * left_coeffs + 
                    (1 - self.alpha) * self.smoothed_left
                )
        
        # Apply EMA to right lane
        if right_coeffs is not None:
            if self.smoothed_right is None:
                self.smoothed_right = right_coeffs
            else:
                self.smoothed_right = (
                    self.alpha * right_coeffs + 
                    (1 - self.alpha) * self.smoothed_right
                )
        
        return self.smoothed_left, self.smoothed_right
    
    def _calculate_confidence(self) -> float:
        """
        Calculate detection confidence based on consistency.
        
        Confidence = (frames with detection) / (total frames in history)
        
        High confidence (>0.7): Lanes detected consistently
        Medium confidence (0.4-0.7): Intermittent detections
        Low confidence (<0.4): Rarely detected, unreliable
        
        Returns:
            float: Confidence score from 0.0 to 1.0
        """
        # Count how many recent frames had lane detections
        left_conf = len(self.left_lane_history) / self.settings.model.temporal_memory_frames
        right_conf = len(self.right_lane_history) / self.settings.model.temporal_memory_frames
        
        # Overall confidence is average of both lanes
        return (left_conf + right_conf) / 2
    
    def _calculate_curvature(
        self, 
        left_coeffs: Optional[np.ndarray], 
        right_coeffs: Optional[np.ndarray]
    ) -> Optional[float]:
        """
        Calculate road curvature in meters.
        
        Curvature radius tells us how sharp the turn is:
        - Large radius (>1000m): Straight or gentle curve
        - Medium radius (100-1000m): Normal curve
        - Small radius (<100m): Sharp turn
        
        This is useful for:
        - Adjusting vehicle speed (slow down for sharp curves)
        - Path planning (predict future trajectory)
        - Driver warnings (alert on tight curves)
        
        Math: For polynomial x = ay² + by + c, curvature radius is:
        R = [(1 + (dx/dy)²)^(3/2)] / |d²x/dy²|
        
        Args:
            left_coeffs: Left lane polynomial coefficients
            right_coeffs: Right lane polynomial coefficients
            
        Returns:
            float: Average curvature radius in meters, or None if unavailable
        """
        if left_coeffs is None or right_coeffs is None:
            return None
        
        # Conversion factors: pixels to meters
        # These are estimates; real values depend on camera calibration
        ym_per_pix = 30 / 720   # 30 meters of road per 720 pixels (vertical)
        xm_per_pix = 3.7 / 700  # 3.7 meters lane width per 700 pixels (horizontal)
        
        # Evaluate at bottom of image (closest to vehicle)
        y_eval = self.height
        
        # Calculate curvature for left lane
        # Convert coefficients from pixel space to meter space
        left_a = left_coeffs[0] * xm_per_pix / (ym_per_pix ** 2)
        left_b = left_coeffs[1] * xm_per_pix / ym_per_pix
        
        # Curvature formula
        left_curvature = (
            ((1 + (2 * left_a * y_eval * ym_per_pix + left_b) ** 2) ** 1.5) / 
            abs(2 * left_a)
        )
        
        # Calculate curvature for right lane
        right_a = right_coeffs[0] * xm_per_pix / (ym_per_pix ** 2)
        right_b = right_coeffs[1] * xm_per_pix / ym_per_pix
        
        right_curvature = (
            ((1 + (2 * right_a * y_eval * ym_per_pix + right_b) ** 2) ** 1.5) / 
            abs(2 * right_a)
        )
        
        # Return average of both lanes
        return (left_curvature + right_curvature) / 2
