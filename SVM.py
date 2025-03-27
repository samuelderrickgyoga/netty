import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

class SVMClassifier:
    def __init__(self, kernel='rbf'):
        """
        Initialize SVM Classifier for high-dimensional IoT data
        
        Args:
            kernel (str): Kernel type (rbf, linear, poly)
        """
        self.kernel = kernel
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(
                kernel=kernel, 
                probability=True, 
                random_state=42
            ))
        ])
        
        # Parameter grid for grid search
        self.param_grid = {
            'svm__C': [0.1, 1, 10, 100],
            'svm__gamma': ['scale', 'auto', 0.1, 1, 10]
        }
        
        self.best_model = None
        self.grid_search = None
    
    def preprocess_data(self, filepath):
        """
        Preprocess high-dimensional IoT data
        
        Args:
            filepath (str): Path to CSV file
        
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        # Load data
        data = pd.read_csv(filepath)
        
        # Separate features and target
        X = data.drop('target', axis=1)
        y = data['target']
        
        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_with_grid_search(self, X_train, y_train):
        """
        Perform grid search for hyperparameter tuning
        
        Args:
            X_train (array): Training features
            y_train (array): Training labels
        """
        # Cross-validation strategy
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        
        # Grid search
        self.grid_search = GridSearchCV(
            self.pipeline, 
            param_grid=self.param_grid, 
            cv=cv, 
            scoring='accuracy',
            n_jobs=-1
        )
        
        self.grid_search.fit(X_train, y_train)
        
        # Best model
        self.best_model = self.grid_search.best_estimator_
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test (array): Test features
            y_test (array): Test labels
        
        Returns:
            dict: Performance metrics
        """
        if self.best_model is None:
            raise ValueError("Model not trained. Call train_with_grid_search first.")
        
        # Predictions
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'best_params': self.grid_search.best_params_
        }
        
        return metrics
    
    def plot_decision_boundary(self, X, y):
        """
        Plot decision boundary for 2D data
        
        Args:
            X (array): Features
            y (array): Labels
        """
        if X.shape[1] != 2:
            print("Decision boundary plot requires 2D data")
            return
        
        # Fit model on entire dataset
        self.pipeline.fit(X, y)
        
        # Plot decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, 0.1),
            np.arange(y_min, y_max, 0.1)
        )
        
        Z = self.pipeline.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
        plt.title('SVM Decision Boundary')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()
    
    def save_model(self, filepath):
        """
        Save trained model
        
        Args:
            filepath (str): Path to save model
        """
        import joblib
        joblib.dump(self.best_model, filepath)
    
    def load_model(self, filepath):
        """
        Load pre-trained model
        
        Args:
            filepath (str): Path to model file
        """
        import joblib
        self.best_model = joblib.load(filepath)

def main(filepath):
    """
    Main function to run SVM pipeline
    
    Args:
        filepath (str): Path to IoT sensor data CSV
    """
    # Initialize SVM classifier
    svm_classifier = SVMClassifier(kernel='rbf')
    
    # Preprocess data
    X_train, X_test, y_train, y_test = svm_classifier.preprocess_data(filepath)
    
    # Train with grid search
    svm_classifier.train_with_grid_search(X_train, y_train)
    
    # Evaluate model
    results = svm_classifier.evaluate(X_test, y_test)
    
    # Print results
    print("Model Performance:")
    print(f"Accuracy: {results['accuracy']}")
    print("\nClassification Report:")
    print(results['classification_report'])
    print("\nBest Hyperparameters:")
    print(results['best_params'])
    
    # Optional: Plot decision boundary for 2D data
    if X_train.shape[1] == 2:
        svm_classifier.plot_decision_boundary(X_train.values, y_train.values)
    
    return svm_classifier

# Example usage (uncomment and provide actual filepath)
# if __name__ == "__main__":
#     model = main('path/to/your/iot_sensor_data.csv')
#     model.save_model('svm_model.joblib')