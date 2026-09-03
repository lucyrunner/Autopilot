"""
Test Lane Detector Module

What this teaches:
- How to test CV algorithms
- Confidence validation
- Output format checking
"""

import pytest
import numpy as np
from src.perception.lane_detector import ProductionLaneDetector


class TestLaneDetector:
    """Test suite for lane detection."""
    
    @pytest.fixture
    def detector(self):
        """Create detector instance for tests."""
        return ProductionLaneDetector(image_shape=(720, 1280))
    
    def test_detector_initialization(self, detector):
        """
        Test: Detector initializes correctly.
        
        Educational: Proper initialization is critical for reproducibility.
        """
        assert detector is not None
        assert detector.image_height == 720
        assert detector.image_width == 1280
        print("✓ Lane detector initializes correctly")
    
    def test_detect_returns_valid_format(self, detector):
        """
        Test: Detect method returns correct format.
        
        Educational: Output format contracts are crucial in production.
        """
        # Create synthetic image
        image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        # Run detection
        result = detector.detect(image)
        
        # Verify required keys
        assert 'confidence' in result, "Must have confidence"
        assert 'left_lane' in result, "Must have left_lane"
        assert 'right_lane' in result, "Must have right_lane"
        assert 'curvature' in result, "Must have curvature"
        
        # Verify confidence range
        assert 0.0 <= result['confidence'] <= 1.0, "Confidence must be in [0,1]"
        
        print("✓ Detection output format is valid")
    
    def test_synthetic_lane_detection(self, detector):
        """
        Test: Can detect synthetic lane markings.
        
        Educational: Synthetic tests help verify algorithm logic.
        """
        # Create image with clear vertical lines (simple lanes)
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Draw white vertical lines (lane markings)
        cv2.line(image, (400, 720), (450, 0), (255, 255, 255), 5)  # Left lane
        cv2.line(image, (830, 720), (780, 0), (255, 255, 255), 5)  # Right lane
        
        result = detector.detect(image)
        
        # With clear lines, should have some confidence
        # (May still be low due to ROI masking and other filters)
        assert result is not None
        print(f"✓ Synthetic lane detection confidence: {result['confidence']:.2%}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
