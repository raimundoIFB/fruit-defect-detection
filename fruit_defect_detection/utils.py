import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlretrieve

def load_sample_image():
    """Load a sample fruit image for testing"""
    # Create sample directory if it doesn't exist
    os.makedirs('sample_data', exist_ok=True)
    
    # Sample image URL (you can replace this with your own images)
    sample_url = "https://github.com/opencv/opencv/raw/master/samples/data/fruits.jpg"
    sample_path = "sample_data/sample_fruit.jpg"
    
    if not os.path.exists(sample_path):
        print("Downloading sample image...")
        urlretrieve(sample_url, sample_path)
    
    image = cv2.imread(sample_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def visualize_results(images, predictions, class_names=['Class A', 'Class B', 'Class C']):
    """Visualize classification results"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i in range(min(6, len(images))):
        axes[i].imshow(images[i])
        pred_class = np.argmax(predictions[i])
        confidence = np.max(predictions[i])
        axes[i].set_title(f'Pred: {class_names[pred_class]}\nConf: {confidence:.2f}')
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def create_sample_dataset(num_samples=10):
    """Create a sample dataset for testing"""
    images = []
    labels = []
    
    for i in range(num_samples):
        # Create synthetic fruit images (replace with real images in practice)
        if i % 3 == 0:
            # Class A - Good quality
            img = np.random.randint(100, 200, (512, 512, 3), dtype=np.uint8)
            label = 0
        elif i % 3 == 1:
            # Class B - Medium quality
            img = np.random.randint(50, 150, (512, 512, 3), dtype=np.uint8)
            label = 1
        else:
            # Class C - Poor quality
            img = np.random.randint(0, 100, (512, 512, 3), dtype=np.uint8)
            label = 2
        
        images.append(img)
        labels.append(label)
    
    return np.array(images), np.array(labels)
