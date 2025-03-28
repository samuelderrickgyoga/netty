import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.io as pio

class Mastitis_XAI_Visualizer:
    def __init__(self, seed=42):
        """
        Initialize the visualizer with a random seed for reproducibility
        """
        np.random.seed(seed)
        # Predefined color palette for models
        self.model_colors = {
            'XGBoost': '#1E88E5',      # Blue (Best Performer)
            'CNN': '#4CAF50',           # Green (Best Performer)
            'Logistic Regression': '#FFC107',  # Amber (Worst Performer)
            'DQN': '#E53935'            # Red (Worst Performer)
        }

    def feature_importance_analysis(self, save_path='mastitis_feature_importance.png'):
        """
        Generate simulated feature importance plot
        """
        plt.figure(figsize=(12, 6))
        
        # Simulated feature names and importance
        features = [
            'Udder Temp', 'Somatic Cell Count', 
            'Milk Quality', 'Udder Size', 
            'Age', 'Breed Factors'
        ]
        
        # Importance values for different models
        xgboost = [0.35, 0.45, 0.30, 0.25, 0.15, 0.10]
        cnn = [0.32, 0.40, 0.35, 0.28, 0.18, 0.12]
        logistic_regression = [0.15, 0.20, 0.10, 0.08, 0.05, 0.03]
        dqn = [0.10, 0.15, 0.08, 0.06, 0.04, 0.02]
        
        # Create beeswarm-like plot
        plt.scatter(xgboost, range(len(features)), 
                    label='XGBoost', color=self.model_colors['XGBoost'], alpha=0.7)
        plt.scatter(cnn, range(len(features)), 
                    label='CNN', color=self.model_colors['CNN'], alpha=0.7)
        plt.scatter(logistic_regression, range(len(features)), 
                    label='Logistic Regression', color=self.model_colors['Logistic Regression'], alpha=0.7)
        plt.scatter(dqn, range(len(features)), 
                    label='DQN', color=self.model_colors['DQN'], alpha=0.7)
        
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title('Simulated Feature Importance Across Mastitis Detection Models')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def confusion_matrix_visualization(self, save_path='mastitis_confusion_matrix.png'):
        """
        Generate simulated confusion matrices for different models
        """
        # Define labels for mastitis detection
        labels = ['Healthy', 'Mastitis']
        
        # Simulate confusion matrices based on provided metrics
        matrices = {
            'XGBoost': np.array([
                [0.94, 0.06],  # High accuracy
                [0.03, 0.97]   # Low false negatives
            ]),
            'CNN': np.array([
                [0.92, 0.08],  # High accuracy
                [0.02, 0.98]   # Very low false negatives
            ]),
            'Logistic Regression': np.array([
                [0.85, 0.15],  # Lower accuracy
                [0.13, 0.87]   # More false negatives/positives
            ]),
            'DQN': np.array([
                [0.75, 0.25],  # Low accuracy
                [0.21, 0.79]   # High false negatives
            ])
        }
        
        plt.figure(figsize=(16, 12))
        
        for i, (model, matrix) in enumerate(matrices.items(), 1):
            plt.subplot(2, 2, i)
            sns.heatmap(matrix, annot=True, cmap='Blues', 
                        xticklabels=labels, yticklabels=labels, 
                        vmin=0, vmax=1)
            plt.title(f'Confusion Matrix: {model}')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def latent_space_analysis(self, save_path='mastitis_latent_space.html'):
        """
        Generate latent space visualization
        """
        # Simulate latent representations
        n_samples = 500
        xgboost = np.random.normal(0, 0.5, (n_samples, 2))
        cnn = np.random.normal(1, 0.5, (n_samples, 2))
        logistic_regression = np.random.normal(2, 1, (n_samples, 2))
        dqn = np.random.normal(3, 1.5, (n_samples, 2))
        
        # Combine and create labels
        X = np.vstack([xgboost, cnn, logistic_regression, dqn])
        labels = ['XGBoost']*n_samples + ['CNN']*n_samples + \
                 ['Logistic Regression']*n_samples + ['DQN']*n_samples
        
        # Create interactive Plotly scatter plot
        trace_data = []
        for model, color in self.model_colors.items():
            mask = [label == model for label in labels]
            trace = go.Scatter(
                x=X[mask, 0], 
                y=X[mask, 1], 
                mode='markers',
                name=model,
                marker=dict(color=color, size=8, opacity=0.7)
            )
            trace_data.append(trace)
        
        layout = go.Layout(
            title='Simulated Latent Space Representation of Mastitis Detection Models',
            xaxis={'title': 'Dimension 1'},
            yaxis={'title': 'Dimension 2'}
        )
        
        fig = go.Figure(data=trace_data, layout=layout)
        pio.write_html(fig, file=save_path)

    def generate_all_visualizations(self):
        """
        Generate all XAI visualizations for mastitis detection models
        """
        self.feature_importance_analysis()
        self.confusion_matrix_visualization()
        self.latent_space_analysis()
        print("All mastitis detection XAI visualizations generated successfully!")

# Execute the visualization generation
if __name__ == "__main__":
    xai_viz = Mastitis_XAI_Visualizer()
    xai_viz.generate_all_visualizations()