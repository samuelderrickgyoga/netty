import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, log_loss, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


class XGBoostClassifier:
    def __init__(self, n_estimators=200, max_depth=6, learning_rate=0.1):
        """
        Initialize XGBoost Classifier
        
        Args:
            n_estimators (int): Number of boosting rounds
            max_depth (int): Maximum tree depth
            learning_rate (float): Boosting learning rate
        """
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective='binary:logistic',
            eval_metric='auc',
            use_label_encoder=False,
            random_state=42
        )
        
        self.feature_importances_ = None
        self.cv_results_ = None
    
    def preprocess_data(self, filepath):
        """
        Preprocess tabular data
        
        Args:
            filepath (str): Path to CSV file
        
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        # Load data
        data = pd.read_csv(filepath)
        
        # Handle categorical features
        categorical_columns = data.select_dtypes(include=['object']).columns
        label_encoders = {}
        
        for col in categorical_columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            label_encoders[col] = le
        
        # Separate features and class1
        X = data.drop('class1', axis=1)
        y = data['class1']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_with_cross_validation(self, X_train, y_train):
        """
        Train XGBoost with cross-validation
        
        Args:
            X_train (array): Training features
            y_train (array): Training labels
        """
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Prepare CV data
        eval_set = [(X_train, y_train)]
        
        # Train with cross-validation
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        # Store feature importances
        self.feature_importances_ = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test (array): Test features
            y_test (array): Test labels
        
        Returns:
            dict: Performance metrics
        """
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = {
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'log_loss': log_loss(y_test, y_pred_proba),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        return metrics
    
    def plot_feature_importance(self, top_n=10, output_dir='output'):
        """
        Plot top N feature importances and save to file
        
        Args:
            top_n (int): Number of top features to plot
            output_dir (str): Directory to save plot
        """
        if self.feature_importances_ is None:
            raise ValueError("Model not trained yet")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        top_features = self.feature_importances_.head(top_n)
        
        # Updated to avoid deprecation warning
        sns.barplot(
            x='importance', 
            y='feature', 
            data=top_features
        )
        
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
        plt.close()
    
    def save_model(self, filepath):
        """
        Save trained model
        
        Args:
            filepath (str): Path to save model
        """
        import joblib
        joblib.dump(self.model, filepath)
    
    def save_performance_metrics(self, metrics, output_dir='output'):
        """
        Save performance metrics to a text file
        
        Args:
            metrics (dict): Performance metrics
            output_dir (str): Directory to save metrics
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Write metrics to file
        with open(os.path.join(output_dir, 'model_performance.txt'), 'w') as f:
            f.write("Model Performance Metrics:\n")
            f.write("=======================\n")
            for metric, value in metrics.items():
                f.write(f"{metric.replace('_', ' ').title()}: {value}\n")
        
        # Save feature importances
        if self.feature_importances_ is not None:
            self.feature_importances_.to_csv(
                os.path.join(output_dir, 'feature_importances.csv'), 
                index=False
            )


def main(filepath='cow_mastitis_data.csv', output_dir='output'):
    """
    Main function to run XGBoost pipeline
    
    Args:
        filepath (str): Path to IoT sensor data CSV
        output_dir (str): Directory to save outputs
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize XGBoost classifier
    xgboost_classifier = XGBoostClassifier()
    
    # Preprocess data
    X_train, X_test, y_train, y_test = xgboost_classifier.preprocess_data(filepath)
    
    # Train model with cross-validation
    xgboost_classifier.train_with_cross_validation(X_train, y_train)
    
    # Evaluate model
    results = xgboost_classifier.evaluate(X_test, y_test)
    
    # Print results
    print("Model Performance:")
    for metric, value in results.items():
        print(f"{metric.replace('_', ' ').title()}: {value}")
    
    # Plot and save feature importances
    xgboost_classifier.plot_feature_importance(output_dir=output_dir)
    
    # Save performance metrics
    xgboost_classifier.save_performance_metrics(results, output_dir)
    
    # Save model
    model_path = os.path.join(output_dir, 'xgboost_model.joblib')
    xgboost_classifier.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    return xgboost_classifier


# Example usage
if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs('output', exist_ok=True)
    
    # Run main function
    model = main(
        filepath='cow_mastitis_data.csv', 
        output_dir='output'
    )