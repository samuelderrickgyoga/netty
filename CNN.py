import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class CNNSpatialClassifier:
    def __init__(self, input_shape=(128, 128, 3), num_classes=1):
        """
        Initialize CNN for spatial feature extraction
        
        Args:
            input_shape (tuple): Input image dimensions
            num_classes (int): Number of output classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
        self.history = None
    
    def _build_model(self):
        """
        Construct CNN architecture
        
        Returns:
            tf.keras.Model: Compiled CNN model
        """
        model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            MaxPooling2D((2, 2)),
            
            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            
            # Third Convolutional Block
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            
            # Flatten and Dense Layers
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']
        )
        
        return model
    
    def load_image_data(self, image_directory, labels_csv):
        """
        Load image data from directory with corresponding labels
        
        Args:
            image_directory (str): Path to image directory
            labels_csv (str): Path to CSV with image labels
        
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        # Load labels
        labels_df = pd.read_csv(labels_csv)
        
        # Image data generator with augmentation
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=0.2  # Split for validation
        )
        
        # Generate image dataset
        train_generator = datagen.flow_from_directory(
            image_directory,
            target_size=self.input_shape[:2],
            batch_size=32,
            class_mode='binary',
            subset='training'
        )
        
        validation_generator = datagen.flow_from_directory(
            image_directory,
            target_size=self.input_shape[:2],
            batch_size=32,
            class_mode='binary',
            subset='validation'
        )
        
        return train_generator, validation_generator
    
    def train(self, train_generator, validation_generator, epochs=50):
        """
        Train CNN model
        
        Args:
            train_generator (tf.keras.preprocessing.image.DirectoryIterator): Training data
            validation_generator (tf.keras.preprocessing.image.DirectoryIterator): Validation data
            epochs (int): Number of training epochs
        """
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2, 
            patience=5, 
            min_lr=0.00001
        )
        
        # Train model
        self.history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            callbacks=[early_stopping, reduce_lr]
        )
    
    def evaluate(self, validation_generator):
        """
        Evaluate model performance
        
        Args:
            validation_generator (tf.keras.preprocessing.image.DirectoryIterator): Validation data
        
        Returns:
            dict: Performance metrics
        """
        # Evaluate model
        evaluation = self.model.evaluate(validation_generator)
        
        metrics = {
            'loss': evaluation[0],
            'accuracy': evaluation[1],
            'auc': evaluation[2]
        }
        
        return metrics
    
    def plot_training_history(self):
        """
        Plot training and validation metrics
        """
        if self.history is None:
            print("No training history available")
            return
        
        # Accuracy plot
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """
        Save trained model
        
        Args:
            filepath (str): Path to save model
        """
        self.model.save(filepath)
    
    def load_model(self, filepath):
        """
        Load pre-trained model
        
        Args:
            filepath (str): Path to model file
        """
        self.model = tf.keras.models.load_model(filepath)

def main(image_directory, labels_csv):
    """
    Main function to run CNN pipeline
    
    Args:
        image_directory (str): Path to image directory
        labels_csv (str): Path to CSV with image labels
    """
    # Initialize CNN classifier
    cnn_classifier = CNNSpatialClassifier()
    
    # Load image data
    train_generator, validation_generator = cnn_classifier.load_image_data(
        image_directory, labels_csv
    )
    
    # Train model
    cnn_classifier.train(train_generator, validation_generator)
    
    # Evaluate model
    metrics = cnn_classifier.evaluate(validation_generator)
    
    # Print results
    print("Model Performance:")
    for metric, value in metrics.items():
        print(f"{metric.title()}: {value}")
    
    # Plot training history
    cnn_classifier.plot_training_history()
    
    return cnn_classifier

# Example usage (uncomment and provide actual paths)
# if __name__ == "__main__":
#     model = main(
#         image_directory='path/to/image/directory', 
#         labels_csv='path/to/labels.csv'
#     )
#     model.save_model('cnn_model.h5')