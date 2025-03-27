import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json

class LSTMTimeSeriesPredictor:
    def __init__(self, sequence_length=10, num_features=None):
        """
        Initialize LSTM for time series prediction
        
        Args:
            sequence_length (int): Number of time steps in input sequence
            num_features (int): Number of features in input data
        """
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.label_encoders = {}
        self.scalers = {}
        self.model = None
        self.history = None
    
    def _preprocess_data(self, data):
        """
        Preprocess data with handling for categorical and numerical features
        
        Args:
            data (pd.DataFrame): Input dataframe
        
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        # Make a copy of the dataframe
        processed_data = data.copy()
        
        # Handle categorical columns
        categorical_columns = ['Cow_ID', 'Breed', 'Previous_Mastits_status']
        for col in categorical_columns:
            # Use Label Encoding for categorical columns
            le = LabelEncoder()
            processed_data[col] = le.fit_transform(processed_data[col].astype(str))
            self.label_encoders[col] = le
        
        # Normalize numerical columns
        numerical_columns = [
            'Months_after_giving_birth', 'IUFL', 'EUFL', 'IUFR', 'EUFR', 
            'IURL', 'EURL', 'IURR', 'EURR', 'Temperature', 
            'Hardness', 'Pain', 'Milk_visibility'
        ]
        
        for col in numerical_columns:
            scaler = MinMaxScaler()
            processed_data[col] = scaler.fit_transform(processed_data[[col]])
            self.scalers[col] = scaler
        
        return processed_data
    
    def prepare_sequence_data(self, data, target_column='class1'):
        """
        Prepare time series data for LSTM
        
        Args:
            data (pd.DataFrame): Input time series data
            target_column (str): Name of target column
        
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        # Preprocess data
        processed_data = self._preprocess_data(data)
        
        # Encode target column if it's categorical
        target_le = LabelEncoder()
        processed_data[target_column] = target_le.fit_transform(processed_data[target_column].astype(str))
        
        # Select features (excluding Cow_ID and target)
        feature_columns = [
            col for col in processed_data.columns 
            if col not in ['Cow_ID', target_column]
        ]
        
        # Set number of features
        self.num_features = len(feature_columns)
        
        # Build model with correct feature dimensions
        self.model = self._build_model()
        
        # Create sequences
        X, y = [], []
        for cow_id in processed_data['Cow_ID'].unique():
            cow_data = processed_data[processed_data['Cow_ID'] == cow_id].sort_values('Day')
            
            for i in range(len(cow_data) - self.sequence_length):
                X.append(cow_data[feature_columns].iloc[i:i+self.sequence_length].values)
                y.append(cow_data[target_column].iloc[i+self.sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def _build_model(self):
        """
        Construct LSTM model architecture
        
        Returns:
            tf.keras.Model: Compiled LSTM model
        """
        model = Sequential([
            # First LSTM Layer
            LSTM(64, 
                 return_sequences=True, 
                 input_shape=(self.sequence_length, self.num_features)),
            Dropout(0.2),
            
            # Second LSTM Layer
            LSTM(32),
            Dropout(0.2),
            
            # Output Layer
            Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'mae']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50):
        """
        Train LSTM model
        
        Args:
            X_train (array): Training sequence features
            y_train (array): Training labels
            X_val (array, optional): Validation sequence features
            y_val (array, optional): Validation labels
            epochs (int): Number of training epochs
        """
        # Default validation split if not provided
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
        
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
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr]
        )
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test (array): Test sequence features
            y_test (array): Test labels
        
        Returns:
            dict: Performance metrics
        """
        # Evaluate model
        evaluation = self.model.evaluate(X_test, y_test)
        
        metrics = {
            'loss': evaluation[0],
            'accuracy': evaluation[1],
            'mae': evaluation[2]
        }
        
        return metrics
    
    def plot_training_history(self, output_path=None):
        """
        Plot training and validation metrics
        
        Args:
            output_path (str, optional): Path to save the plot
        """
        if self.history is None:
            print("No training history available")
            return
        
        # MAE plot
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['mae'], label='Training MAE')
        plt.plot(self.history.history['val_mae'], label='Validation MAE')
        plt.title('Mean Absolute Error')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
    
    def save_model(self, filepath):
        """
        Save trained model
        
        Args:
            filepath (str): Path to save model
        """
        # Save model
        self.model.save(filepath)
        
        # Save label encoders and scalers
        import joblib
        joblib.dump({
            'label_encoders': self.label_encoders,
            'scalers': self.scalers
        }, filepath + '_preprocessing.joblib')

def main(filepath):
    """
    Main function to run LSTM pipeline
    
    Args:
        filepath (str): Path to time series CSV
    """
    # Create outputs folder
    os.makedirs('outputs', exist_ok=True)
    
    # Load data
    data = pd.read_csv(filepath)
    
    # Initialize LSTM predictor
    lstm_predictor = LSTMTimeSeriesPredictor(sequence_length=5)
    
    # Prepare sequence data
    X_train, X_test, y_train, y_test = lstm_predictor.prepare_sequence_data(
        data, target_column='class1'
    )
    
    # Train model
    lstm_predictor.train(X_train, y_train)
    
    # Evaluate model
    metrics = lstm_predictor.evaluate(X_test, y_test)
    
    # Save metrics
    with open('outputs/model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Plot and save training history
    lstm_predictor.plot_training_history(
        output_path='outputs/training_history.png'
    )
    
    # Save model
    lstm_predictor.save_model('outputs/lstm_model')
    
    return lstm_predictor

# Example usage
if __name__ == "__main__":
    model = main(r'C:\Users\derick\Desktop\netty\cow_mastitis_data.csv')