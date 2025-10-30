"""
Fruit Defect Detection Library
A comprehensive library for detecting external defects in fruits using deep learning
Based on the research paper: "Identification of External Defects on Fruits Using Deep Learning"
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .preprocessing import *
from .feature_extraction import *
from .models import *
from .utils import *

__all__ = [
    # Preprocessing
    'ImagePreprocessor',
    
    # Feature Extraction
    'GLCMFeatureExtractor',
    'GaborFilter',
    
    # Models
    'FruitDefectCNN',
    'FruitDefectDNN',
    
    # Utils
    'load_sample_image',
    'visualize_results',
    'create_sample_dataset'
]
