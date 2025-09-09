# evaluation_metrics.py
"""Comprehensive evaluation metrics for imbalanced classification"""
import numpy as np
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                           confusion_matrix, precision_recall_curve,
                           average_precision_score)
from utils import bootstrap_confidence_interval, mcnemar_test
from config import COST_PARAMS
class ImbalanceEvaluator:
    """Comprehensive evaluation framework for extreme imbalance scenarios"""
    
    def __init__(self, cost_params=None):
        self.cost_params = cost_params or COST_PARAMS
        self.results = {}
        
    def evaluate_model(self, y_true, y_pred, y_proba=None, model_name="Model"):
        """Comprehensive evaluation of a single model"""
        results = {}
        
        # Basic classification metrics
        results['precision'] = precision_score(y_true, y_pred, zero_division=0)
        results['recall'] = recall_score(y_true, y_pred, zero_division=0)
        results['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix components
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle case where only one class is predicted
            tn = fp = fn = tp = 0
            if len(np.unique(y_pred)) == 1:
                if np.unique(y_pred)[0] == 0:
                    tn = np.sum(y_true == 0)
                    fn = np.sum(y_true == 1)
                else:
                    tp = np.sum(y_true == 1)
                    fp = np.sum(y_true == 0)
        
        results['true_positives'] = tp
        results['false_positives'] = fp
        results['true_negatives'] = tn
        results['false_negatives'] = fn
        
        # Operational metrics
        total_negatives = tn + fp
        results['false_alarm_rate'] = fp / total_negatives if total_negatives > 0 else 0
        results['false_alarm_percentage'] = results['false_alarm_rate'] * 100
        
        # Economic metrics
        if self.cost_params:
            cost_fn = fn * self.cost_params['C_base']
            cost_fp = fp * self.cost_params['C_FP']
            results['total_cost'] = cost_fn + cost_fp
            results['cost_per_sample'] = results['total_cost'] / len(y_true)
        
        # Advanced metrics for probability predictions
        if y_proba is not None:
            results['average_precision'] = average_precision_score(y_true, y_proba)
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
            results['pr_auc'] = np.trapz(precision_curve, recall_curve)
        
        self.results[model_name] = results
        return results
    
    def compare_models(self, results_dict):
        """Statistical comparison between multiple models"""
        model_names = list(results_dict.keys())
        comparison_results = {}
        
        # Pairwise comparisons
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                model1, model2 = model_names[i], model_names[j]
                
                # Extract predictions (would need to be stored separately in practice)
                # This is a simplified version for demonstration
                comparison_key = f"{model1}_vs_{model2}"
                comparison_results[comparison_key] = {
                    'f1_diff': results_dict[model1]['f1_score'] - results_dict[model2]['f1_score'],
                    'precision_diff': results_dict[model1]['precision'] - results_dict[model2]['precision'],
                    'recall_diff': results_dict[model1]['recall'] - results_dict[model2]['recall'],
                    'cost_diff': results_dict[model1].get('total_cost', 0) - results_dict[model2].get('total_cost', 0)
                }
        
        return comparison_results
    
    def calculate_confidence_intervals(self, metric_values, metric_name="F1-Score"):
        """Calculate bootstrap confidence intervals for metrics"""
        if len(metric_values) < 2:
            return metric_values[0], metric_values[0]
        
        return bootstrap_confidence_interval(metric_values)
    
    def generate_performance_summary(self):
        """Generate comprehensive performance summary"""
        if not self.results:
            return "No results available"
        
        summary = "\n=== PERFORMANCE SUMMARY ===\n"
        
        # Sort models by F1-score
        sorted_models = sorted(self.results.items(), 
                             key=lambda x: x[1]['f1_score'], 
                             reverse=True)
        
        for rank, (model_name, results) in enumerate(sorted_models, 1):
            summary += f"\n{rank}. {model_name}:\n"
            summary += f"   F1-Score: {results['f1_score']:.4f}\n"
            summary += f"   Precision: {results['precision']:.4f}\n"
            summary += f"   Recall: {results['recall']:.4f}\n"
            summary += f"   False Negatives: {results['false_negatives']}\n"
            summary += f"   False Alarms: {results['false_alarm_percentage']:.4f}%\n"
            if 'total_cost' in results:
                summary += f"   Total Cost: ${results['total_cost']:,.0f}\n"
        
        return summary

