# visualization.py
"""Visualization module for generating paper figures"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import precision_recall_curve
from config import VIZ_CONFIG
import warnings
warnings.filterwarnings('ignore')

class SmartGridVisualizer:
    """Visualization suite for smart grid predictive maintenance results"""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        self.colors = VIZ_CONFIG['colors']
        self.figsize = VIZ_CONFIG['figure_size']
        
    def plot_precision_recall_curves(self, models_data, save_path=None):
        """Generate precision-recall curves for multiple models"""
        plt.figure(figsize=self.figsize)
        
        for idx, (model_name, data) in enumerate(models_data.items()):
            y_true, y_scores = data['y_true'], data['y_scores']
            
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            
            plt.plot(recall, precision, 
                    color=self.colors[idx % len(self.colors)],
                    linewidth=2.5,
                    label=f'{model_name}')
        
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title('Precision-Recall Curves for Imbalance Handling Techniques', 
                 fontsize=16, pad=20)
        plt.legend(loc='lower left', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add operational constraints
        plt.axhline(y=0.20, color='red', linestyle='--', alpha=0.7, 
                   label='Min Precision Threshold (0.20)')
        plt.axvline(x=0.80, color='red', linestyle='--', alpha=0.7,
                   label='Target Recall (0.80)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=VIZ_CONFIG['dpi'], 
                       format=VIZ_CONFIG['save_format'], bbox_inches='tight')
        plt.show()
    
    def plot_performance_comparison(self, results_dict, save_path=None):
        """Create bar chart comparing model performance"""
        models = list(results_dict.keys())
        f1_scores = [results_dict[model]['f1_score'] for model in models]
        precisions = [results_dict[model]['precision'] for model in models]
        recalls = [results_dict[model]['recall'] for model in models]
        
        x = np.arange(len(models))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        bars1 = ax.bar(x - width, f1_scores, width, label='F1-Score', 
                      color=self.colors[0], alpha=0.8)
        bars2 = ax.bar(x, precisions, width, label='Precision', 
                      color=self.colors[1], alpha=0.8)
        bars3 = ax.bar(x + width, recalls, width, label='Recall', 
                      color=self.colors[2], alpha=0.8)
        
        ax.set_xlabel('Models', fontsize=14)
        ax.set_ylabel('Performance Score', fontsize=14)
        ax.set_title('Comparative Performance: Imbalance Handling Techniques', 
                    fontsize=16, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=10)
        
        autolabel(bars1)
        autolabel(bars2)
        autolabel(bars3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=VIZ_CONFIG['dpi'], 
                       format=VIZ_CONFIG['save_format'], bbox_inches='tight')
        plt.show()
    
    def plot_cost_sensitivity_analysis(self, cost_analysis_data, save_path=None):
        """Plot cost parameter sensitivity analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Alpha_load sensitivity
        alpha_load_values = cost_analysis_data['alpha_load']['values']
        f1_scores = cost_analysis_data['alpha_load']['f1_scores']
        ax1.plot(alpha_load_values, f1_scores, 'o-', color=self.colors[0], linewidth=2)
        ax1.set_xlabel('α_load', fontsize=12)
        ax1.set_ylabel('F1-Score', fontsize=12)
        ax1.set_title('Sensitivity to Load Factor', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Alpha_season sensitivity
        alpha_season_values = cost_analysis_data['alpha_season']['values']
        f1_scores_season = cost_analysis_data['alpha_season']['f1_scores']
        ax2.plot(alpha_season_values, f1_scores_season, 's-', color=self.colors[1], linewidth=2)
        ax2.set_xlabel('α_season', fontsize=12)
        ax2.set_ylabel('F1-Score', fontsize=12)
        ax2.set_title('Sensitivity to Seasonal Factor', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Threshold optimization
        thresholds = cost_analysis_data['threshold']['values']
        precisions = cost_analysis_data['threshold']['precisions']
        recalls = cost_analysis_data['threshold']['recalls']
        f1s = cost_analysis_data['threshold']['f1_scores']
        
        ax3.plot(thresholds, precisions, 'o-', label='Precision', color=self.colors[0])
        ax3.plot(thresholds, recalls, 's-', label='Recall', color=self.colors[1])
        ax3.plot(thresholds, f1s, '^-', label='F1-Score', color=self.colors[2])
        ax3.set_xlabel('Decision Threshold', fontsize=12)
        ax3.set_ylabel('Performance Score', fontsize=12)
        ax3.set_title('Precision-Recall Trade-off', fontsize=14)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Feature importance (SHAP values)
        feature_names = cost_analysis_data['features']['names'][:10]
        importance_values = cost_analysis_data['features']['importance'][:10]
        
        ax4.barh(range(len(feature_names)), importance_values, color=self.colors[3])
        ax4.set_yticks(range(len(feature_names)))
        ax4.set_yticklabels(feature_names)
        ax4.set_xlabel('SHAP Value', fontsize=12)
        ax4.set_title('Top 10 Feature Importance', fontsize=14)
        ax4.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=VIZ_CONFIG['dpi'], 
                       format=VIZ_CONFIG['save_format'], bbox_inches='tight')
        plt.show()
    
    def plot_economic_analysis(self, economic_data, save_path=None):
        """Plot economic analysis results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Cost comparison
        models = list(economic_data['costs'].keys())
        costs = list(economic_data['costs'].values())
        
        bars = ax1.bar(models, costs, color=self.colors[:len(models)], alpha=0.8)
        ax1.set_ylabel('Annual Cost ($)', fontsize=12)
        ax1.set_title('Expected Annual Costs by Model', fontsize=14)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add cost values on bars
        for bar, cost in zip(bars, costs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'${cost:,.0f}',
                    ha='center', va='bottom')
        
        # ROI analysis
        scenarios = list(economic_data['roi'].keys())
        roi_values = list(economic_data['roi'].values())
        
        colors_roi = ['green' if roi > 0 else 'red' for roi in roi_values]
        bars_roi = ax2.bar(scenarios, roi_values, color=colors_roi, alpha=0.8)
        ax2.set_ylabel('ROI (%)', fontsize=12)
        ax2.set_title('ROI Analysis by Scenario', fontsize=14)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add ROI values on bars
        for bar, roi in zip(bars_roi, roi_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{roi:.1f}%',
                    ha='center', va='bottom' if height > 0 else 'top')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=VIZ_CONFIG['dpi'], 
                       format=VIZ_CONFIG['save_format'], bbox_inches='tight')
        plt.show()
    
    def plot_temporal_validation_comparison(self, temporal_data, save_path=None):
        """Compare temporal vs non-temporal validation results"""
        methods = list(temporal_data.keys())
        temporal_f1 = [temporal_data[method]['temporal_f1'] for method in methods]
        standard_f1 = [temporal_data[method]['standard_f1'] for method in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        bars1 = ax.bar(x - width/2, standard_f1, width, label='Standard CV', 
                      color=self.colors[0], alpha=0.8)
        bars2 = ax.bar(x + width/2, temporal_f1, width, label='Temporal CV', 
                      color=self.colors[1], alpha=0.8)
        
        ax.set_xlabel('Methods', fontsize=14)
        ax.set_ylabel('F1-Score', fontsize=14)
        ax.set_title('Temporal vs Standard Cross-Validation Results', 
                    fontsize=16, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add difference annotations
        for i, (std, temp) in enumerate(zip(standard_f1, temporal_f1)):
            diff = std - temp
            ax.annotate(f'+{diff:.3f}', 
                       xy=(i, max(std, temp) + 0.01),
                       ha='center', va='bottom',
                       fontsize=10, color='red',
                       weight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=VIZ_CONFIG['dpi'], 
                       format=VIZ_CONFIG['save_format'], bbox_inches='tight')
        plt.show()

