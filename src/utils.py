# utils.py
"""Utility functions for the Smart Grid Framework"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def generate_synthetic_grid_data(n_samples=120000, n_features=847, failure_rate=0.00029):
    """Generate synthetic smart grid data matching the paper's specifications"""
    np.random.seed(42)
    
    # Generate normal operation data
    normal_samples = int(n_samples * (1 - failure_rate))
    failure_samples = n_samples - normal_samples
    
    # Normal operation features (slightly positive skew)
    normal_data = np.random.gamma(2, 2, (normal_samples, n_features))
    
    # Failure features (higher variance, different distribution)
    failure_data = np.random.gamma(4, 3, (failure_samples, n_features))
    
    # Combine data
    X = np.vstack([normal_data, failure_data])
    y = np.hstack([np.zeros(normal_samples), np.ones(failure_samples)])
    
    # Add temporal component
    timestamps = pd.date_range('2022-01-01', periods=n_samples, freq='1H')
    
    # Add physics-based patterns
    # Thermal stress (cumulative)
    thermal_stress = np.cumsum(np.random.exponential(0.1, n_samples))
    
    # Load patterns (daily/seasonal cycles)
    hours = np.arange(n_samples) % 24
    load_pattern = 0.7 + 0.3 * np.sin(2 * np.pi * hours / 24)
    
    # Seasonal effects
    days = np.arange(n_samples) // 24
    seasonal_effect = 0.8 + 0.2 * np.sin(2 * np.pi * days / 365.25)
    
    # Add these as features
    X[:, 0] = thermal_stress
    X[:, 1] = load_pattern
    X[:, 2] = seasonal_effect

    if np.any(np.isnan(X)):
        print("هشدار: داده‌های X شامل مقادیر NaN هستند!")
    if np.any(np.isnan(y)):
        print("هشدار: داده‌های y شامل مقادیر NaN هستند!")
    
    return X, y, timestamps

def calculate_cost_matrix(load_factor, seasonal_factor, time_factor, cost_params):
    """Calculate dynamic cost matrix based on operational conditions"""
    C_FN = cost_params['C_base'] * (
        1 + cost_params['alpha_load'] * load_factor +
        cost_params['alpha_season'] * seasonal_factor +
        cost_params['alpha_time'] * time_factor
    )
    
    return np.array([[0, cost_params['C_FP']], 
                     [C_FN, 0]])

def mcnemar_test(y_true, pred1, pred2):
    """Perform McNemar's test for comparing two models"""
    # Create contingency table
    cm1 = confusion_matrix(y_true, pred1)
    cm2 = confusion_matrix(y_true, pred2)
    
    # McNemar's test statistic
    b = np.sum((pred1 != y_true) & (pred2 == y_true))
    c = np.sum((pred1 == y_true) & (pred2 != y_true))
    
    if b + c == 0:
        return 1.0  # Perfect agreement
    
    mcnemar_stat = (abs(b - c) - 1)**2 / (b + c)
    p_value = 1 - stats.chi2.cdf(mcnemar_stat, 1)
    
    return p_value

def bootstrap_confidence_interval(metric_values, confidence=0.95, n_bootstrap=1000):
    """Calculate bootstrap confidence interval"""
    np.random.seed(42)
    bootstrap_values = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(metric_values, len(metric_values), replace=True)
        bootstrap_values.append(np.mean(sample))
    
    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_values, alpha * 100)
    upper = np.percentile(bootstrap_values, (1 - alpha) * 100)
    
    return lower, upper

