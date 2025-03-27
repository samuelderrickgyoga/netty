import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix,
    roc_curve,
    roc_auc_score
)
from imblearn.over_sampling import SMOTE
import joblib

def load_and_preprocess_data(filepath):
    """
    Load IoT sensor data and preprocess for logistic regression
     
    Args:
        filepath (str): Path to CSV file with IoT sensor data
     
    Returns:
        tuple: Processed X_train, X_test, y_train, y_test
    """
    # Load data
    data = pd.read_csv(filepath)
     
    # Identify numeric columns
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove any non-numeric columns from features
    X = data[numeric_columns].drop('class1', axis=1)
    y = data['class1']
     
    # Normalize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
     
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
     
    # Apply SMOTE for handling class imbalance
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
     
    return X_train_resampled, X_test, y_train_resampled, y_test, scaler, X

def train_logistic_regression(X_train, y_train):
    """
    Train Logistic Regression model
     
    Args:
        X_train (array): Training features
        y_train (array): Training labels
     
    Returns:
        LogisticRegression: Trained model
    """
    model = LogisticRegression(
        solver='lbfgs',  # Default solver
        C=1.0,  # L2 regularization strength
        max_iter=1000,  # Increase max iterations if convergence fails
        random_state=42
    )
     
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, output_dir='output'):
    """
    Evaluate model performance
     
    Args:
        model (LogisticRegression): Trained model
        X_test (array): Test features
        y_test (array): Test labels
        output_dir (str): Directory to save outputs
     
    Returns:
        dict: Performance metrics
    """
    # Predict probabilities and labels
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
     
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save performance metrics
    with open(os.path.join(output_dir, 'logistic_regression_metrics.txt'), 'w') as f:
        f.write("Model Performance Metrics:\n")
        f.write("=======================\n")
        for metric, value in metrics.items():
            f.write(f"{metric.replace('_', ' ').title()}: {value}\n")
    
    # Plot and save ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["roc_auc"]:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()
    
    # Plot and save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
     
    return metrics

def main(filepath='cow_mastitis_data.csv', output_dir='output'):
    """
    Main function to run logistic regression pipeline
     
    Args:
        filepath (str): Path to IoT sensor data CSV
        output_dir (str): Directory to save outputs
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler, X = load_and_preprocess_data(filepath)
     
    # Train model
    model = train_logistic_regression(X_train, y_train)
     
    # Evaluate model
    results = evaluate_model(model, X_test, y_test, output_dir)
     
    # Print results
    print("Model Performance:")
    for metric, value in results.items():
        print(f"{metric.replace('_', ' ').title()}: {value}")
    
    # Save model
    model_path = os.path.join(output_dir, 'logistic_regression_model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(output_dir, 'data_scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # Save feature names for reference
    feature_names_path = os.path.join(output_dir, 'feature_names.txt')
    with open(feature_names_path, 'w') as f:
        f.write('\n'.join(X.columns.tolist()))
    print(f"Feature names saved to {feature_names_path}")
     
    return model

# Example usage
if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs('output', exist_ok=True)
    
    # Run main function
    model = main(
        filepath='cow_mastitis_data.csv', 
        output_dir='output'
    )