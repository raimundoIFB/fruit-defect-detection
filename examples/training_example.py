"""
Training example for the Fruit Defect Detection library
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from fruit_defect_detection import ImagePreprocessor, GLCMFeatureExtractor, FruitDefectDNN, FruitDefectCNN
from fruit_defect_detection.utils import create_sample_dataset

def train_cnn_example():
    """Example of training CNN model"""
    print("Training CNN Model Example")
    
    # Create sample dataset
    X, y = create_sample_dataset(200)
    
    # Normalize images
    X_normalized = X.astype('float32') / 255.0
    
    # Convert labels to categorical
    from tensorflow.keras.utils import to_categorical
    y_categorical = to_categorical(y, num_classes=3)
    
    # Create CNN model
    cnn_model = FruitDefectCNN(input_shape=(512, 512, 3))
    cnn_model.compile_model()
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_val = X_normalized[:split_idx], X_normalized[split_idx:]
    y_train, y_val = y_categorical[:split_idx], y_categorical[split_idx:]
    
    # Train model
    history = cnn_model.train(X_train, y_train, X_val, y_val, epochs=20, batch_size=8)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    return cnn_model, history

def train_dnn_example():
    """Example of training DNN model with GLCM features"""
    print("Training DNN Model with GLCM Features Example")
    
    # Create sample dataset
    X, y = create_sample_dataset(150)
    
    # Extract GLCM features
    feature_extractor = GLCMFeatureExtractor()
    X_features = []
    
    for img in X:
        features = feature_extractor.extract_rgb_glcm_features(img)
        X_features.append(features)
    
    X_features = np.array(X_features)
    
    # Convert labels to categorical
    from tensorflow.keras.utils import to_categorical
    y_categorical = to_categorical(y, num_classes=3)
    
    # Create DNN model
    dnn_model = FruitDefectDNN(input_dim=X_features.shape[1])
    dnn_model.compile_model()
    
    # Split data
    split_idx = int(0.8 * len(X_features))
    X_train, X_val = X_features[:split_idx], X_features[split_idx:]
    y_train, y_val = y_categorical[:split_idx], y_categorical[split_idx:]
    
    # Train model
    history = dnn_model.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=16)
    
    print(f"Final DNN Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final DNN Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    return dnn_model, history

if __name__ == "__main__":
    # Train both models
    cnn_model, cnn_history = train_cnn_example()
    dnn_model, dnn_history = train_dnn_example()
