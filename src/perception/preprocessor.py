"""
Image Preprocessing Utilities

Common preprocessing operations for perception models:
- Resizing with aspect ratio preservation
- Normalization
- Augmentation for robustness testing
- Format conversion
"""

import cv2
import numpy as np
from typing import Tuple


class ImagePreprocessor:
    """
    Utility class for image preprocessing.
    
    Provides methods for:
    - Resizing (letterbox, crop, stretch)
    - Normalization (0-1, -1 to 1, ImageNet stats)
    - Color space conversion
    - Augmentation (brightness, contrast, etc.)
    """
    
    @staticmethod
    def resize_letterbox(
        image: np.ndarray,
        target_size: Tuple[int, int],
        fill_value: int = 114
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Resize image with letterboxing (preserves aspect ratio).
        
        Letterboxing adds padding to maintain aspect ratio.
        This prevents distortion but reduces effective image area.
        
        Args:
            image: Input image (H, W, 3)
            target_size: Target size (height, width)
            fill_value: Padding color (gray by default)
            
        Returns:
            Tuple of:
            - Resized image with padding
            - Scale factor used
            - Padding (top, left)
        """
        h, w = image.shape[:2]
        target_h, target_w = target_size
        
        # Calculate scale to fit within target
        scale = min(target_h / h, target_w / w)
        
        # New dimensions
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create canvas with padding
        canvas = np.full((target_h, target_w, 3), fill_value, dtype=np.uint8)
        
        # Calculate padding to center image
        top = (target_h - new_h) // 2
        left = (target_w - new_w) // 2
        
        # Place resized image on canvas
        canvas[top:top+new_h, left:left+new_w] = resized
        
        return canvas, scale, (top, left)
    
    @staticmethod
    def normalize_zero_one(image: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] range.
        
        Common for neural networks.
        """
        return image.astype(np.float32) / 255.0
    
    @staticmethod
    def normalize_neg_one_one(image: np.ndarray) -> np.ndarray:
        """
        Normalize image to [-1, 1] range.
        
        Used by some GAN architectures.
        """
        return (image.astype(np.float32) / 127.5) - 1.0
    
    @staticmethod
    def normalize_imagenet(image: np.ndarray) -> np.ndarray:
        """
        Normalize using ImageNet statistics.
        
        Mean: [0.485, 0.456, 0.406]
        Std:  [0.229, 0.224, 0.225]
        
        Used by models pretrained on ImageNet.
        """
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        normalized = image.astype(np.float32) / 255.0
        normalized = (normalized - mean) / std
        
        return normalized
    
    @staticmethod
    def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjust image brightness.
        
        Args:
            image: Input image
            factor: Brightness factor (0.5 = darker, 2.0 = brighter)
        """
        adjusted = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        return adjusted
    
    @staticmethod
    def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjust image contrast.
        
        Args:
            image: Input image
            factor: Contrast factor (0.5 = less contrast, 2.0 = more contrast)
        """
        adjusted = cv2.convertScaleAbs(image, alpha=factor, beta=128*(1-factor))
        return adjusted
    
    @staticmethod
    def apply_gaussian_noise(
        image: np.ndarray,
        mean: float = 0,
        std: float = 25
    ) -> np.ndarray:
        """
        Add Gaussian noise to image (for robustness testing).
        
        Args:
            image: Input image
            mean: Noise mean
            std: Noise standard deviation
        """
        noise = np.random.normal(mean, std, image.shape).astype(np.float32)
        noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
        return noisy.astype(np.uint8)
