"""
Test Preprocessor Module

What this teaches:
- How to test image preprocessing
- Shape validation importance
- Normalization correctness
- Edge cases handling
"""

import pytest
import numpy as np
import cv2
from src.perception.preprocessor import ImagePreprocessor


class TestImagePreprocessor:
    """Test suite for image preprocessing utilities."""
    
    def test_resize_letterbox_shape(self):
        """
        Test: Letterbox resizing preserves aspect ratio.
        
        Educational: Letterboxing adds padding instead of stretching,
        preventing distortion that would hurt model accuracy.
        """
        # Create test image (non-square to test aspect ratio)
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Resize to square target
        resized, scale, padding = ImagePreprocessor.resize_letterbox(
            image, target_size=(640, 640)
        )
        
        # Verify output shape
        assert resized.shape == (640, 640, 3), "Output should match target size"
        
        # Verify scale factor is reasonable
        expected_scale = 640 / 640  # Limited by width
        assert abs(scale - expected_scale) < 0.01, "Scale factor incorrect"
        
        print("✓ Letterbox resize maintains aspect ratio correctly")
    
    def test_normalize_zero_one_range(self):
        """
        Test: Normalization to [0, 1] range.
        
        Educational: Neural networks work better with normalized inputs.
        [0, 1] range centers data and prevents saturation.
        """
        # Create image with known values
        image = np.array([[[0], [128], [255]]], dtype=np.uint8)
        
        # Normalize
        normalized = ImagePreprocessor.normalize_zero_one(image)
        
        # Check range
        assert normalized.min() >= 0.0, "Min should be >= 0"
        assert normalized.max() <= 1.0, "Max should be <= 1"
        
        # Check specific values
        assert abs(normalized[0, 0, 0] - 0.0) < 0.01, "0 should map to 0.0"
        assert abs(normalized[0, 1, 0] - 0.5) < 0.01, "128 should map to 0.5"
        assert abs(normalized[0, 2, 0] - 1.0) < 0.01, "255 should map to 1.0"
        
        print("✓ Normalization to [0, 1] works correctly")
    
    def test_brightness_adjustment(self):
        """
        Test: Brightness adjustment.
        
        Educational: Augmentation helps model handle varying lighting.
        """
        image = np.full((100, 100, 3), 128, dtype=np.uint8)
        
        # Increase brightness
        brighter = ImagePreprocessor.adjust_brightness(image, factor=1.5)
        assert np.mean(brighter) > np.mean(image), "Should be brighter"
        
        # Decrease brightness
        darker = ImagePreprocessor.adjust_brightness(image, factor=0.5)
        assert np.mean(darker) < np.mean(image), "Should be darker"
        
        print("✓ Brightness adjustment works correctly")
    
    def test_gaussian_noise_addition(self):
        """
        Test: Adding Gaussian noise for robustness testing.
        
        Educational: Testing with noise helps verify model robustness.
        """
        image = np.full((100, 100, 3), 128, dtype=np.uint8)
        
        # Add noise
        noisy = ImagePreprocessor.apply_gaussian_noise(image, mean=0, std=10)
        
        # Verify noise was added (variance should increase)
        assert np.var(noisy) > np.var(image), "Noise should increase variance"
        
        # Verify values stay in valid range
        assert noisy.min() >= 0, "No negative values"
        assert noisy.max() <= 255, "No values above 255"
        
        print("✓ Gaussian noise addition works correctly")


if __name__ == '__main__':
    # Run tests when executed directly
    pytest.main([__file__, '-v'])
