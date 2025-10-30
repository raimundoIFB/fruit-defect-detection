"""
Tests for preprocessing module
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cv2
from fruit_defect_detection import ImagePreprocessor

def test_preprocessor_initialization():
    """Test preprocessor initialization"""
    preprocessor = ImagePreprocessor(target_size=(256, 256))
    assert preprocessor.target_size == (256, 256)
    print("✓ Preprocessor initialization test passed")

def test_image_resize():
    """Test image resizing"""
    preprocessor = ImagePreprocessor(target_size=(256, 256))
    sample_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    resized = preprocessor.resize_image(sample_image)
    assert resized.shape == (256, 256, 3)
    print("✓ Image resize test passed")

def test_median_filter():
    """Test median filter application"""
    preprocessor = ImagePreprocessor()
    sample_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    filtered = preprocessor.median_filter(sample_image)
    assert filtered.shape == sample_image.shape
    print("✓ Median filter test passed")

def test_contrast_enhancement():
    """Test contrast enhancement"""
    preprocessor = ImagePreprocessor()
    sample_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    enhanced = preprocessor.enhance_contrast(sample_image)
    assert enhanced.shape == sample_image.shape
    print("✓ Contrast enhancement test passed")

def test_full_pipeline():
    """Test complete preprocessing pipeline"""
    preprocessor = ImagePreprocessor()
    sample_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    processed, mask = preprocessor.full_preprocessing_pipeline(sample_image)
    assert processed is not None
    print("✓ Full pipeline test passed")

if __name__ == "__main__":
    test_preprocessor_initialization()
    test_image_resize()
    test_median_filter()
    test_contrast_enhancement()
    test_full_pipeline()
    print("All preprocessing tests passed! ✅")
