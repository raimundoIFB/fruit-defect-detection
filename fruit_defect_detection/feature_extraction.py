import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import gabor

class GLCMFeatureExtractor:
    """
    Extract GLCM texture features from images
    """
    
    def __init__(self, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=16):
        self.distances = distances
        self.angles = angles
        self.levels = levels
        self.feature_names = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'asm']
    
    def calculate_glcm_features(self, image):
        """Calculate GLCM features for a single channel image"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Quantize gray levels
        gray_quantized = (gray / (256 // self.levels)).astype(np.uint8)
        
        # Calculate GLCM
        glcm = graycomatrix(gray_quantized, distances=self.distances, angles=self.angles,
                           levels=self.levels, symmetric=True, normed=True)
        
        # Calculate texture properties
        features = []
        for prop in self.feature_names:
            if prop == 'asm':
                feature_val = graycoprops(glcm, 'ASM')
            else:
                feature_val = graycoprops(glcm, prop)
            features.extend(feature_val.flatten())
        
        return np.array(features)
    
    def extract_rgb_glcm_features(self, image):
        """Extract GLCM features for all RGB channels"""
        features_list = []
        
        for channel in range(3):
            channel_features = self.calculate_glcm_features(image[:, :, channel])
            features_list.extend(channel_features)
        
        return np.array(features_list)
    
    def get_feature_names(self):
        """Get names of all extracted features"""
        names = []
        for channel in ['R', 'G', 'B']:
            for dist in self.distances:
                for angle in self.angles:
                    for prop in self.feature_names:
                        names.append(f"{channel}_d{dist}_a{angle:.2f}_{prop}")
        return names

class GaborFilter:
    """
    Apply Gabor filters for texture analysis
    """
    
    def __init__(self, frequencies=[0.1, 0.3, 0.5], thetas=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
        self.frequencies = frequencies
        self.thetas = thetas
    
    def apply_gabor_filter(self, image):
        """Apply Gabor filter bank to image"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        gabor_features = []
        for frequency in self.frequencies:
            for theta in self.thetas:
                gabor_real, gabor_imag = gabor(gray, frequency=frequency, theta=theta)
                gabor_features.append(gabor_real)
                gabor_features.append(gabor_imag)
        
        return gabor_features
    
    def extract_gabor_features(self, image):
        """Extract statistical features from Gabor filter responses"""
        gabor_responses = self.apply_gabor_filter(image)
        features = []
        
        for response in gabor_responses:
            features.extend([
                np.mean(response),
                np.std(response),
                np.max(response),
                np.min(response)
            ])
        
        return np.array(features)
