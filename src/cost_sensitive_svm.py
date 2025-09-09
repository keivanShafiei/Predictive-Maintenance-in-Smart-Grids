# cost_sensitive_svm.py
"""Cost-sensitive SVM implementation"""
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from config import COST_PARAMS, MODEL_PARAMS

class CostSensitiveSVM:
    """SVM with utility-specific cost matrix optimization"""
    
    def __init__(self, cost_params=None):
        self.cost_params = cost_params or COST_PARAMS
        self.model = None
        self.optimal_threshold = 0.5
        
    def _calculate_cost_weights(self, y, load_factors=None, seasonal_factors=None):
        """Calculate sample-specific cost weights"""
        n_samples = len(y)
        weights = np.ones(n_samples)
        
        for i in range(n_samples):
            if y[i] == 1:  # Failure samples
                load_factor = load_factors[i] if load_factors is not None else 0.5
                seasonal_factor = seasonal_factors[i] if seasonal_factors is not None else 0.5
                
                # Calculate dynamic cost
                cost_multiplier = (1 + 
                                 self.cost_params['alpha_load'] * load_factor +
                                 self.cost_params['alpha_season'] * seasonal_factor)
                
                weights[i] = cost_multiplier * self.cost_params['C_base'] / self.cost_params['C_FP']
            else:  # Normal samples
                weights[i] = 1.0
                
        return weights
    
    def fit(self, X, y, load_factors=None, seasonal_factors=None):
        """Train cost-sensitive SVM"""
        # Calculate cost-sensitive weights
        sample_weights = self._calculate_cost_weights(y, load_factors, seasonal_factors)
        
        # Create cost-sensitive class weights
        class_weight = {
            0: 1.0,
            1: self.cost_params['C_base'] / self.cost_params['C_FP']
        }
        
        # Train SVM with cost-sensitive parameters
        self.model = SVC(
            C=MODEL_PARAMS['svm_C'],
            gamma=MODEL_PARAMS['svm_gamma'],
            kernel='rbf',
            class_weight=class_weight,
            probability=True,
            random_state=42
        )
        
        self.model.fit(X, y, sample_weight=sample_weights)
        
        # Optimize decision threshold
        self._optimize_threshold(X, y)
        
        return self
    
    def _optimize_threshold(self, X, y):
        """Optimize decision threshold for F1-score"""
        y_proba = self.model.predict_proba(X)[:, 1]
        
        best_f1 = 0
        best_threshold = 0.5
        
        thresholds = np.arange(0.1, 0.9, 0.05)
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            # Calculate F1-score
            tp = np.sum((y_pred == 1) & (y == 1))
            fp = np.sum((y_pred == 1) & (y == 0))
            fn = np.sum((y_pred == 0) & (y == 1))
            
            if tp + fp > 0 and tp + fn > 0:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
        
        self.optimal_threshold = best_threshold
    
    def predict(self, X):
        """Make predictions using optimized threshold"""
        y_proba = self.model.predict_proba(X)[:, 1]
        return (y_proba >= self.optimal_threshold).astype(int)
    
    def predict_proba(self, X):
        """Return prediction probabilities"""
        return self.model.predict_proba(X)

