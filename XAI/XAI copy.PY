import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc

class Mastitis_XAI_Comprehensive_Visualizer:
    def __init__(self, seed=42):
        """
        Initialize the comprehensive XAI visualizer
        """
        np.random.seed(seed)
        # Performance metrics for each model
        self.model_metrics = {
            'XGBoost': {
                'accuracy': 0.94,
                'precision': 0.93,
                'recall': 0.95,
                'f1_score': 0.94
            },
            'CNN': {
                'accuracy': 0.98,
                'precision': 0.93,
                'recall': 0.98,
                'f1_score': 0.96
            },
            'Logistic Regression': {
                'accuracy': 0.85,
                'precision': 0.83,
                'recall': 0.87,
                'f1_score': 0.85
            },
            'DQN': {
                'accuracy': 0.75,
                'precision': 0.84,
                'recall': 0.79,
                'f1_score': 0.72
            }
        }
        
        # Color palette
        self.color_palette = {
            'XGBoost': '#1E88E5',      # Blue
            'CNN': '#4CAF50',          # Green
            'Logistic Regression': '#FFC107',  # Amber
            'DQN': '#E53935'           # Red
        }

    def performance_bar_chart(self, save_path='performance_bar_chart.png'):
        """
        Create bar chart comparing model performance metrics
        """
        # Prepare data for plotting
        metrics_df = pd.DataFrame(self.model_metrics).T.reset_index()
        metrics_df = metrics_df.melt(id_vars='index', var_name='Metric', value_name='Value')
        
        plt.figure(figsize=(15, 8))
        chart = sns.barplot(x='index', y='Value', hue='Metric', data=metrics_df, 
                            palette='viridis')
        plt.title('Model Performance Metrics Comparison', fontsize=16)
        plt.xlabel('Models', fontsize=12)
        plt.ylabel('Performance Score', fontsize=12)
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def radar_chart(self, save_path='radar_chart.html'):
        """
        Create interactive radar chart for model performance
        """
        # Prepare data
        models = list(self.model_metrics.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Create traces for each model
        traces = []
        for model in models:
            values = [self.model_metrics[model][metric] for metric in metrics]
            values += [values[0]]  # Repeat first value to close the polygon
            
            trace = go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                fill='toself',
                name=model,
                line_color=self.color_palette[model]
            )
            traces.append(trace)
        
        # Layout
        layout = go.Layout(
            title='Multi-Dimensional Model Performance',
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True
        )
        
        # Create figure and save
        fig = go.Figure(data=traces, layout=layout)
        pio.write_html(fig, file=save_path)

    def roc_and_precision_recall_curves(self, save_path='roc_pr_curves.png'):
        """
        Generate ROC and Precision-Recall curves
        """
        plt.figure(figsize=(15, 6))
        
        # Simulate true labels and prediction probabilities
        np.random.seed(42)
        
        # ROC Curves (subplot 1)
        plt.subplot(1, 2, 1)
        for model, color in self.color_palette.items():
            # Simulate probabilistic predictions
            y_true = np.random.randint(0, 2, 100)
            y_scores = np.random.rand(100)
            
            # Compute ROC curve and AUC
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color=color, 
                     label=f'{model} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.title('ROC Curves')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        
        # Precision-Recall Curves (subplot 2)
        plt.subplot(1, 2, 2)
        for model, color in self.color_palette.items():
            # Simulate probabilistic predictions
            y_true = np.random.randint(0, 2, 100)
            y_scores = np.random.rand(100)
            
            # Compute Precision-Recall curve
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            pr_auc = auc(recall, precision)
            
            plt.plot(recall, precision, color=color, 
                     label=f'{model} (AUC = {pr_auc:.2f})')
        
        plt.title('Precision-Recall Curves')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc='lower left')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def feature_importance_advanced(self, save_path='advanced_feature_importance.png'):
        """
        Advanced feature importance visualization
        """
        # Simulated feature importance data
        features = [
            'Udder Temperature', 'Somatic Cell Count', 
            'Milk Quality', 'Udder Size', 
            'Cow Age', 'Breed Factors'
        ]
        
        # Simulated SHAP-like values
        xgboost_importance = [0.35, 0.45, 0.30, 0.25, 0.15, 0.10]
        cnn_importance = [0.32, 0.40, 0.35, 0.28, 0.18, 0.12]
        
        plt.figure(figsize=(12, 6))
        
        # Create horizontal bar plot
        plt.barh(features, xgboost_importance, 
                 label='XGBoost', alpha=0.7, color=self.color_palette['XGBoost'])
        plt.barh(
            [f'{f} (CNN)' for f in features], 
            cnn_importance, 
            label='CNN', alpha=0.7, color=self.color_palette['CNN']
        )
        
        plt.title('Feature Importance Comparison')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def confusion_matrix_advanced(self, save_path='advanced_confusion_matrix.png'):
        """
        Advanced confusion matrix visualization
        """
        # Simulated confusion matrix data
        models = list(self.model_metrics.keys())
        
        plt.figure(figsize=(16, 12))
        
        for i, model in enumerate(models, 1):
            # Simulate confusion matrix
            cm = np.random.rand(2, 2)
            cm /= cm.sum(axis=1)[:, np.newaxis]  # Normalize
            
            plt.subplot(2, 2, i)
            sns.heatmap(cm, annot=True, cmap='YlGnBu', 
                        xticklabels=['Healthy', 'Mastitis'],
                        yticklabels=['Healthy', 'Mastitis'])
            plt.title(f'Normalized Confusion Matrix: {model}')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def generate_comprehensive_visualizations(self):
        """
        Generate all advanced XAI visualizations
        """
        self.performance_bar_chart()
        self.radar_chart()
        self.roc_and_precision_recall_curves()
        self.feature_importance_advanced()
        self.confusion_matrix_advanced()
        print("Comprehensive XAI visualizations generated successfully!")

# Execute the visualization generation
if __name__ == "__main__":
    xai_viz = Mastitis_XAI_Comprehensive_Visualizer()
    xai_viz.generate_comprehensive_visualizations()