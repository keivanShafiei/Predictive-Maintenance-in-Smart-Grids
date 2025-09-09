# temporal_validation.py
"""
Temporal validation framework for time-series data with extreme imbalance.
This implementation strictly prevents data leakage by maintaining temporal separation
between training, validation, and test sets without moving samples across time boundaries.
"""
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, precision_score, recall_score
from config import DATASET_CONFIG

class TemporalValidator:
    """
    Implements temporal-aware cross-validation for imbalanced data with strict data leakage prevention.
    
    This validator ensures that training data never contains information from future time periods,
    maintaining the integrity of temporal validation by accepting data limitations rather than
    artificially balancing splits through future data redistribution.
    """
    
    def __init__(self, n_splits=5, test_size_ratio=0.15):
        self.n_splits = n_splits
        self.test_size_ratio = test_size_ratio
        self.cv_results = []
        
    def temporal_stratified_split(self, X, y, timestamps):
        """
        Create temporal splits while strictly preventing data leakage.
        
        This method creates chronologically ordered train/validation/test splits without
        moving samples across temporal boundaries. If training sets lack failure samples
        due to natural temporal distribution, this is reported as a limitation rather
        than artificially corrected through future data redistribution.
        
        Args:
            X: Feature matrix
            y: Target labels  
            timestamps: Temporal ordering information
            
        Returns:
            Tuple of (X_train, X_val, X_test), (y_train, y_val, y_test)
        """
        n_samples = len(X)
        failure_indices = np.where(y == 1)[0]
        normal_indices = np.where(y == 0)[0]
        
        print(f"Total samples: {n_samples}, Failures: {len(failure_indices)}, Normal: {len(normal_indices)}")
        
        # Sort by timestamp to ensure chronological order
        sort_idx = np.argsort(timestamps)
        X_sorted = X[sort_idx]
        y_sorted = y[sort_idx]
        timestamps_sorted = timestamps[sort_idx]
        
        # Find positions of failure events in the chronologically sorted array
        failure_positions = np.where(y_sorted == 1)[0]
        
        # Determine split points based on failure distribution and temporal constraints
        if len(failure_positions) < 3:
            print("[INFO] Very few failures available. Using fixed temporal splits to prevent data leakage.")
            # Use fixed time-based splits when failures are extremely sparse
            # This prevents any attempt to artificially move failures into training
            train_end = int(n_samples * 0.7)
            val_end = int(n_samples * 0.85)
            
        else:
            print("[INFO] Multiple failures detected. Calculating failure-aware temporal splits.")
            # Calculate splits that naturally include early failures in training
            # without violating temporal order
            
            # Determine how many failures should ideally be in training/validation
            n_train_failures = max(1, len(failure_positions) // 3)
            n_val_failures = max(1, len(failure_positions) // 4)
            
            # Find temporal position that includes desired training failures
            # This ensures failures fall naturally into training due to temporal ordering
            if n_train_failures <= len(failure_positions):
                train_failure_position = failure_positions[n_train_failures - 1]
                # Set training end to include this failure plus some buffer
                # Respect minimum training size while staying within temporal bounds
                train_end = max(train_failure_position + 1000, int(n_samples * 0.6))
                train_end = min(train_end, int(n_samples * 0.8))  # Maximum 80% for training
            else:
                # Fallback to fixed split if failure distribution doesn't support our strategy
                train_end = int(n_samples * 0.7)
            
            # Calculate validation end similarly, ensuring no future data leakage
            remaining_failures = failure_positions[failure_positions >= train_end]
            if len(remaining_failures) >= n_val_failures:
                val_failure_position = remaining_failures[n_val_failures - 1]
                val_end = max(train_end + 1000, val_failure_position + 500)
                val_end = min(val_end, int(n_samples * 0.9))  # Maximum 90% for train+val
            else:
                val_end = int(n_samples * 0.85)
        
        # Create strictly temporal splits - no data redistribution allowed
        X_train = X_sorted[:train_end]
        y_train = y_sorted[:train_end]
        
        X_val = X_sorted[train_end:val_end]
        y_val = y_sorted[train_end:val_end]
        
        X_test = X_sorted[val_end:]
        y_test = y_sorted[val_end:]
        
        # Count actual failures in each split after temporal separation
        train_failures = np.sum(y_train)
        val_failures = np.sum(y_val)
        test_failures = np.sum(y_test)
        
        # Report the natural distribution of failures across temporal splits
        print(f"Temporal split results - Train: {len(y_train)} samples ({train_failures} failures)")
        print(f"                         Val: {len(y_val)} samples ({val_failures} failures)")
        print(f"                        Test: {len(y_test)} samples ({test_failures} failures)")
        
        # Issue explicit warnings for data limitations without attempting to fix them
        # This maintains scientific integrity by honestly reporting constraints
        if train_failures == 0:
            print(f"[WARNING] Training set contains 0 failure samples due to temporal distribution.")
            print(f"          Models will likely struggle to learn positive class without synthetic techniques.")
            print(f"          This reflects the natural temporal sparsity of failure events.")
            
        if val_failures == 0:
            print(f"[WARNING] Validation set contains 0 failure samples due to temporal distribution.")
            print(f"          Validation metrics may not reflect true performance on failure prediction.")
            
        if test_failures == 0:
            print(f"[WARNING] Test set contains 0 failure samples due to temporal distribution.")
            print(f"          Cannot evaluate failure prediction performance on this split.")
        
        # Additional warning for extremely challenging scenarios
        if train_failures <= 1:
            print(f"[WARNING] Training set has very few failures ({train_failures}).")
            print(f"          Consider using robust oversampling techniques or collecting more historical data.")
        
        return (X_train, X_val, X_test), (y_train, y_val, y_test)
    
    def time_series_cv(self, X, y, model, scoring_func=f1_score):
        """
        Perform time-series cross-validation with strict temporal constraints.
        
        This method ensures that each fold maintains temporal order without any
        data leakage between training and testing periods. Folds with insufficient
        data are transparently reported and skipped rather than artificially balanced.
        """
        # Adjust number of splits based on available failures to ensure meaningful evaluation
        n_splits = min(self.n_splits, max(2, np.sum(y) // 2))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        
        print(f"[INFO] Performing {n_splits}-fold temporal cross-validation")
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx]
            
            # Check temporal data constraints without attempting to fix them
            train_classes = len(np.unique(y_train_fold))
            train_failures = np.sum(y_train_fold)
            test_failures = np.sum(y_test_fold)
            
            print(f"Fold {fold}: Train failures={train_failures}, Test failures={test_failures}")
            
            # Skip folds that cannot provide meaningful evaluation due to data constraints
            if train_classes < 2:
                print(f"[WARNING] Fold {fold}: Skipping - only {train_classes} class(es) in training data")
                print(f"          This reflects natural temporal distribution of failure events")
                continue
                
            if test_failures < 1:
                print(f"[WARNING] Fold {fold}: Skipping - no failures in test period")
                print(f"          Cannot evaluate failure prediction performance on this fold")
                continue
            
            try:
                # Train model on temporally constrained data
                model.fit(X_train_fold, y_train_fold)
                y_pred = model.predict(X_test_fold)
                
                # Calculate performance score
                score = scoring_func(y_test_fold, y_pred, zero_division=0)
                cv_scores.append(score)
                
                print(f"[INFO] Fold {fold}: Score = {score:.4f}")
                
            except Exception as e:
                print(f"[ERROR] Fold {fold}: Model training failed - {str(e)}")
                print(f"        This may indicate insufficient training data in this temporal period")
                continue
        
        # Report cross-validation results with appropriate warnings
        if len(cv_scores) == 0:
            print("[WARNING] No valid CV folds completed due to temporal data constraints")
            print("          Consider collecting more historical data or using different validation strategy")
            return np.array([0.0])
        elif len(cv_scores) < n_splits:
            print(f"[WARNING] Only {len(cv_scores)}/{n_splits} folds completed due to data limitations")
            print("          CV variance estimates may be less reliable")
            
        return np.array(cv_scores)
    
    def calculate_cv_variance(self, scores):
        """
        Calculate cross-validation variance with appropriate handling of limited data.
        
        Args:
            scores: Array of cross-validation scores
            
        Returns:
            Variance of scores, or 0.0 if insufficient data points
        """
        if len(scores) <= 1:
            print("[WARNING] Insufficient CV scores for reliable variance calculation")
            return 0.0
        
        return np.var(scores)
    
    def validate_temporal_integrity(self, timestamps_train, timestamps_val, timestamps_test):
        """
        Validate that temporal splits maintain chronological order without leakage.
        
        This method provides an additional check to ensure the temporal validation
        framework maintains scientific integrity.
        
        Args:
            timestamps_train: Training set timestamps
            timestamps_val: Validation set timestamps  
            timestamps_test: Test set timestamps
            
        Returns:
            Boolean indicating whether temporal integrity is maintained
        """
        # Check that all training timestamps come before validation timestamps
        if len(timestamps_train) > 0 and len(timestamps_val) > 0:
            if np.max(timestamps_train) >= np.min(timestamps_val):
                print("[ERROR] Temporal leakage detected: Training data overlaps with validation period")
                return False
        
        # Check that all validation timestamps come before test timestamps
        if len(timestamps_val) > 0 and len(timestamps_test) > 0:
            if np.max(timestamps_val) >= np.min(timestamps_test):
                print("[ERROR] Temporal leakage detected: Validation data overlaps with test period")
                return False
        
        # Check that all training timestamps come before test timestamps
        if len(timestamps_train) > 0 and len(timestamps_test) > 0:
            if np.max(timestamps_train) >= np.min(timestamps_test):
                print("[ERROR] Temporal leakage detected: Training data overlaps with test period")
                return False
        
        print("[INFO] Temporal integrity validated - no data leakage detected")
        return True