"""
Basic usage example for the Fruit Defect Detection library
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fruit_defect_detection import ImagePreprocessor, GLCMFeatureExtractor, FruitDefectDNN
from fruit_defect_detection.utils import load_sample_image, create_sample_dataset
import numpy as np

def main():
    print("Fruit Defect Detection Library - Basic Usage Example")
    
    # Load sample image
    print("1. Loading sample image...")
    image = load_sample_image()
    
    # Preprocess image
    print("2. Preprocessing image...")
    preprocessor = ImagePreprocessor()
    processed_image, mask = preprocessor.full_preprocessing_pipeline(image)
    
    # Extract features
    print("3. Extracting GLCM features...")
    feature_extractor = GLCMFeatureExtractor()
    features = feature_extractor.extract_rgb_glcm_features(processed_image)
    
    print(f"Extracted {len(features)} features")
    print(f"Feature vector shape: {features.shape}")
    
    # Create and train model (with sample data)
    print("4. Creating and training DNN model...")
    
    # Create sample dataset
    X, y = create_sample_dataset(100)
    
    # Extract features for all samples
    X_features = []
    for img in X:
        features = feature_extractor.extract_rgb_glcm_features(img)
        X_features.append(features)
    
    X_features = np.array(X_features)
    
    # Convert labels to categorical
    from tensorflow.keras.utils import to_categorical
    y_categorical = to_categorical(y, num_classes=3)
    
    # Create and train model
    model = FruitDefectDNN(input_dim=X_features.shape[1])
    model.compile_model()
    
    # Split data
    split_idx = int(0.8 * len(X_features))
    X_train, X_val = X_features[:split_idx], X_features[split_idx:]
    y_train, y_val = y_categorical[:split_idx], y_categorical[split_idx:]
    
    # Train model
    history = model.train(X_train, y_train, X_val, y_val, epochs=10, batch_size=8)
    
    print("5. Training completed!")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

if __name__ == "__main__":
    main()
