import tensorflow as tf
from tensorflow.keras import layers, models

class FruitDefectCNN:
    """
    CNN model for fruit defect classification
    """
    
    def __init__(self, input_shape, num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
    
    def _build_model(self):
        """Build CNN architecture as described in the paper"""
        model = models.Sequential([
            # Data augmentation
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            
            # First convolutional group
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.SpatialDropout2D(0.25),
            
            # Second convolutional group
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.SpatialDropout2D(0.25),
            
            # Third convolutional group
            layers.Conv2D(64, (3, 3), activation='relu'),
            
            # Global Average Pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(64, activation='swish'),
            layers.Dropout(0.5),
            layers.Dense(32, activation='elu'),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model"""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the model"""
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        return history

class FruitDefectDNN:
    """
    DNN model for GLCM feature-based classification
    """
    
    def __init__(self, input_dim, num_classes=3):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = self._build_model()
    
    def _build_model(self):
        """Build DNN architecture for GLCM features"""
        model = models.Sequential([
            layers.Dense(128, activation='swish', input_shape=(self.input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='elu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='swish'),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model"""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train the model"""
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        return history
