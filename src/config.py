"""Configuration file for Smart Grid Predictive Maintenance Framework"""
import numpy as np

# Dataset Configuration
DATASET_CONFIG = {
    'total_samples': 60000,
    'failure_events': 35,
    'failure_rate': 0.029,  # 0.029%
    'imbalance_ratio': 3448,
    'original_features': 847,
    'physics_features': 127,
    'temporal_split': [0.70, 0.15, 0.15],
    'cv_folds': 5
}

# Cost-Sensitive Parameters (from DOE data)
COST_PARAMS = {
    'C_base': 2.85e6,  # Base outage cost ($2.85M)
    'C_FP': 75000,     # False positive cost ($75K)
    'alpha_load': 0.8,  # Load-dependent multiplier
    'alpha_season': 0.3, # Seasonal multiplier
    'alpha_time': 0.3,   # Time-dependent multiplier
    'beta_load': 0.64,
    'beta_season': 0.38,
    'beta_time': 0.32
}

# Model Hyperparameters
MODEL_PARAMS = {
    'svm_C': 10.0,
    'svm_gamma': 0.05,
    'rf_n_estimators': 100,
    'lstm_units': 64,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 50
}

# Evaluation Thresholds
PERFORMANCE_TARGETS = {
    'min_recall': 0.80,
    'min_precision': 0.20,
    'max_false_alarm_rate': 0.005,
    'target_f1': 0.40,
    'min_fn_reduction': 0.50
}

# Visualization Settings
VIZ_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8',
    'colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
    'save_format': 'png'
}

