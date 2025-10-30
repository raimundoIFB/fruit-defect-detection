import numpy as np
import cv2
from skimage import segmentation, filters
import matplotlib.pyplot as plt

class ImagePreprocessor:
    """
    Class for preprocessing fruit images including segmentation and enhancement
    """
    
    def __init__(self, target_size=(512, 512)):
        self.target_size = target_size
    
    def resize_image(self, image):
        """Resize image to target size"""
        return cv2.resize(image, self.target_size)
    
    def median_filter(self, image, kernel_size=5):
        """Apply median filter for noise reduction"""
        return cv2.medianBlur(image, kernel_size)
    
    def morphological_erosion(self, image, kernel_size=3):
        """Apply morphological erosion"""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.erode(image, kernel, iterations=1)
    
    def active_contour_segmentation(self, image):
        """Segment fruit using active contour model"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(gray)
            cv2.fillPoly(mask, [largest_contour], 255)
            return mask, largest_contour
        return None, None
    
    def remove_background(self, image, method='erosion'):
        """Remove background from fruit image"""
        resized = self.resize_image(image)
        
        if method == 'plain':
            smoothed = self.median_filter(resized)
        else:
            smoothed = self.morphological_erosion(resized)
        
        mask, contour = self.active_contour_segmentation(smoothed)
        
        if mask is not None:
            # Apply morphological erosion to mask
            kernel = np.ones((3, 3), np.uint8)
            eroded_mask = cv2.erode(mask, kernel, iterations=1)
            
            # Apply mask to original image
            if len(resized.shape) == 3:
                masked_image = cv2.bitwise_and(resized, resized, mask=eroded_mask)
            else:
                masked_image = cv2.bitwise_and(resized, resized, mask=eroded_mask)
            
            return masked_image, eroded_mask
        
        return resized, None
    
    def enhance_contrast(self, image, gamma=1.5):
        """Enhance image contrast using gamma correction"""
        image_normalized = image / 255.0
        corrected = np.power(image_normalized, gamma)
        return np.uint8(corrected * 255)
    
    def full_preprocessing_pipeline(self, image):
        """Complete preprocessing pipeline"""
        # Remove background
        segmented, mask = self.remove_background(image)
        
        # Enhance contrast
        enhanced = self.enhance_contrast(segmented)
        
        return enhanced, mask
    
    def visualize_preprocessing(self, original_image):
        """Visualize all preprocessing steps"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original
        axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Resized
        resized = self.resize_image(original_image)
        axes[0, 1].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Resized')
        axes[0, 1].axis('off')
        
        # Smoothed
        smoothed = self.median_filter(resized)
        axes[0, 2].imshow(cv2.cvtColor(smoothed, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title('Smoothed')
        axes[0, 2].axis('off')
        
        # Segmented
        segmented, mask = self.remove_background(original_image)
        axes[1, 0].imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('Background Removed')
        axes[1, 0].axis('off')
        
        # Mask
        if mask is not None:
            axes[1, 1].imshow(mask, cmap='gray')
            axes[1, 1].set_title('Segmentation Mask')
            axes[1, 1].axis('off')
        
        # Enhanced
        enhanced = self.enhance_contrast(segmented)
        axes[1, 2].imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title('Contrast Enhanced')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        return fig
