# imbalance_handlers.py
"""Advanced imbalance handling techniques with robust fallback mechanisms"""
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE, RandomOverSampler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from config import MODEL_PARAMS
import warnings
warnings.filterwarnings('ignore')

class TemporalSMOTE:
    """
    Temporal-aware SMOTE for time-series data with robust fallback mechanisms.
    
    This implementation provides a three-tier approach to handle extreme class imbalance:
    1. Primary: Apply Temporal-SMOTE if sufficient minority samples are available
    2. Level 1 Fallback: Use RandomOverSampler for very small minority samples
    3. Level 2 Fallback: Return original data if no oversampling is possible
    
    All fallback decisions are transparently logged to ensure scientific reproducibility.
    """
    
    def __init__(self, k_neighbors=5, sampling_strategy='auto'):
        self.k_neighbors = k_neighbors
        self.sampling_strategy = sampling_strategy
        self.nn = NearestNeighbors(n_neighbors=k_neighbors)
        
    def fit_resample(self, X, y, timestamps=None):
        """
        Generate synthetic samples with temporal constraints and robust fallbacks.
        
        Args:
            X: Feature matrix
            y: Target labels
            timestamps: Optional timestamp information for temporal constraints
            
        Returns:
            X_resampled, y_resampled: Resampled data using the most appropriate technique
        """
        minority_indices = np.where(y == 1)[0]
        majority_indices = np.where(y == 0)[0]
        
        X_minority = X[minority_indices]
        n_minority_samples = len(X_minority)
        
        # Check if Temporal SMOTE is feasible
        # SMOTE requires at least k_neighbors + 1 samples to function properly
        if n_minority_samples <= self.k_neighbors:
            
            # Level 1 Fallback: Use RandomOverSampler for 2 or more minority samples
            if n_minority_samples >= 2:
                print(f"      [WARNING] TemporalSMOTE cannot be applied with {n_minority_samples} minority samples and k_neighbors={self.k_neighbors}. Falling back to RandomOverSampler.")
                ros = RandomOverSampler(random_state=42)
                return ros.fit_resample(X, y)
            
            # Level 2 Fallback: Return original data for 0-1 minority samples
            else:
                print(f"      [WARNING] Insufficient minority samples ({n_minority_samples}) to perform any oversampling. Returning original data.")
                return X, y
        
        # Primary approach: Apply Temporal SMOTE
        print(f"      [INFO] Applying TemporalSMOTE with {n_minority_samples} minority samples and k_neighbors={self.k_neighbors}")
        
        try:
            # Fit nearest neighbors on minority class
            self.nn.fit(X_minority)
            
            # Calculate number of synthetic samples needed
            n_synthetic = len(majority_indices) - len(minority_indices)
            
            if n_synthetic <= 0:
                print(f"      [INFO] Classes already balanced. Returning original data.")
                return X, y
                
            synthetic_samples = []
            synthetic_labels = []
            
            for _ in range(n_synthetic):
                # Select random minority sample
                idx = np.random.randint(0, len(minority_indices))
                sample = X_minority[idx]
                
                # Find k nearest neighbors
                distances, neighbor_indices = self.nn.kneighbors([sample])
                neighbor_idx = np.random.choice(neighbor_indices[0])
                neighbor = X_minority[neighbor_idx]
                
                # Generate synthetic sample
                diff = neighbor - sample
                gap = np.random.random()
                synthetic = sample + gap * diff
                
                # Apply temporal and physical constraints
                if timestamps is not None:
                    synthetic = self._apply_temporal_constraints(synthetic, sample, neighbor)
                
                synthetic = self._apply_physical_constraints(synthetic)
                
                synthetic_samples.append(synthetic)
                synthetic_labels.append(1)
            
            # Combine original and synthetic data
            X_resampled = np.vstack([X, np.array(synthetic_samples)])
            y_resampled = np.hstack([y, np.array(synthetic_labels)])
            
            print(f"      [INFO] TemporalSMOTE successfully generated {len(synthetic_samples)} synthetic samples")
            return X_resampled, y_resampled
            
        except Exception as e:
            # Unexpected error in TemporalSMOTE - fallback to RandomOverSampler
            print(f"      [ERROR] TemporalSMOTE failed due to unexpected error: {str(e)}. Falling back to RandomOverSampler.")
            ros = RandomOverSampler(random_state=42)
            return ros.fit_resample(X, y)
    
    def _apply_temporal_constraints(self, synthetic, sample, neighbor):
        """Apply temporal degradation constraints to synthetic samples"""
        max_degradation_rate = 0.1  # Maximum allowable degradation rate
        
        # Ensure synthetic sample respects degradation physics
        degradation = np.abs(synthetic - sample)
        if np.max(degradation) > max_degradation_rate:
            scaling_factor = max_degradation_rate / np.max(degradation)
            synthetic = sample + scaling_factor * (synthetic - sample)
            
        return synthetic
    
    def _apply_physical_constraints(self, synthetic):
        """Apply physical feasibility constraints to synthetic samples"""
        # Ensure non-negative values for physical quantities
        synthetic = np.maximum(synthetic, 0)
        
        # Apply realistic bounds based on equipment specifications
        synthetic = np.minimum(synthetic, 10)  # Max normalized value
        
        return synthetic

class LSTMPredictor:
    """LSTM-based predictor for temporal patterns"""
    
    def __init__(self, sequence_length=24, units=64):
        self.sequence_length = sequence_length
        self.units = units
        self.model = None
        
    def _create_sequences(self, X, y, sequence_length):
        """Create sequences for LSTM input"""
        sequences = []
        labels = []
        
        for i in range(sequence_length, len(X)):
            sequences.append(X[i-sequence_length:i])
            labels.append(y[i])
            
        return np.array(sequences), np.array(labels)
    
    def fit(self, X, y):
        """Train LSTM model"""
        try:
            import tensorflow as tf
            
            # Create sequences
            X_seq, y_seq = self._create_sequences(X, y, self.sequence_length)
            
            if len(X_seq) == 0:
                raise ValueError("Not enough data for sequence creation")
            
            # Build model
            self.model = Sequential([
                LSTM(self.units, return_sequences=True, input_shape=(self.sequence_length, X.shape[1])),
                Dropout(0.2),
                LSTM(self.units, return_sequences=False),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            
            self.model.compile(optimizer='adam', 
                             loss='binary_crossentropy',
                             metrics=['precision', 'recall'])
            
            # Handle class imbalance with class weights
            class_weight = {0: 1, 1: len(y_seq[y_seq==0])/len(y_seq[y_seq==1])}
            
            # Train model
            self.model.fit(X_seq, y_seq, 
                          epochs=MODEL_PARAMS['epochs'], 
                          batch_size=MODEL_PARAMS['batch_size'],
                          class_weight=class_weight,
                          verbose=0)
            
        except ImportError:
            # Fallback to simple classifier if TensorFlow not available
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(class_weight='balanced')
            self.model.fit(X, y)
    
    def predict(self, X):
        """Make predictions"""
        if hasattr(self.model, 'predict_proba'):
            # Sklearn model
            return (self.model.predict_proba(X)[:, 1] > 0.5).astype(int)
        else:
            # TensorFlow model
            X_seq, _ = self._create_sequences(X, np.zeros(len(X)), self.sequence_length)
            if len(X_seq) == 0:
                return np.zeros(len(X), dtype=int)
            
            predictions = self.model.predict(X_seq, verbose=0)
            full_predictions = np.zeros(len(X), dtype=int)
            full_predictions[self.sequence_length:] = (predictions.flatten() > 0.5).astype(int)
            return full_predictions

class OneClassTemporalSVM:
    """
    One-Class SVM with temporal feature enhancement for time-series anomaly detection.
    
    This implementation enhances the feature space with a normalized 'time-elapsed' feature
    before applying a standard One-Class SVM, allowing the model to factor in temporal
    progression when defining the boundary of normal operation. This is particularly
    valuable for equipment degradation patterns where failures become more likely over time.
    
    The temporal enhancement helps the model understand that equipment behavior may
    naturally drift over time, improving anomaly detection accuracy in temporal contexts.
    """
    
    def __init__(self, gamma=0.05, nu=0.1):
        self.gamma = gamma
        self.nu = nu
        from sklearn.svm import OneClassSVM
        from sklearn.preprocessing import MinMaxScaler
        self.model = OneClassSVM(gamma=gamma, nu=nu)
        self.time_scaler = MinMaxScaler()  # For normalizing temporal features
        self.is_fitted = False
        
    def fit(self, X, y=None, timestamps=None):
        """
        Train one-class SVM on normal samples with temporal feature enhancement.
        
        Args:
            X: Feature matrix
            y: Target labels (used to select normal samples only)
            timestamps: Array of timestamps for temporal feature engineering
        """
        # Use only normal samples for training (standard One-Class SVM approach)
        normal_indices = np.where(y == 0)[0] if y is not None else np.arange(len(X))
        X_normal = X[normal_indices]
        
        # Apply temporal feature enhancement
        if timestamps is not None:
            timestamps_normal = timestamps[normal_indices] if y is not None else timestamps
            X_enhanced = self._add_temporal_features(X_normal, timestamps_normal, fit_scaler=True)
            print(f"      [INFO] OneClassTemporalSVM: Enhanced features from {X_normal.shape[1]} to {X_enhanced.shape[1]} dimensions")
        else:
            X_enhanced = X_normal
            print(f"      [WARNING] OneClassTemporalSVM: No timestamps provided, using standard One-Class SVM")
        
        self.model.fit(X_enhanced)
        self.is_fitted = True
        return self
    
    def predict(self, X, timestamps=None):
        """
        Predict anomalies using temporal-enhanced features.
        
        Args:
            X: Feature matrix for prediction
            timestamps: Array of timestamps for temporal feature engineering
            
        Returns:
            Binary predictions (0 for normal, 1 for anomaly/failure)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Apply temporal feature enhancement
        if timestamps is not None:
            X_enhanced = self._add_temporal_features(X, timestamps, fit_scaler=False)
        else:
            X_enhanced = X
            print(f"      [WARNING] OneClassTemporalSVM: No timestamps provided for prediction")
        
        # One-class SVM returns +1 for normal, -1 for anomaly
        # Convert to 0 for normal, 1 for failure
        predictions = self.model.predict(X_enhanced)
        return (predictions == -1).astype(int)
    
    def _add_temporal_features(self, X, timestamps, fit_scaler=False):
        """
        Add normalized temporal features to the input feature matrix.
        
        Args:
            X: Original feature matrix
            timestamps: Array of timestamps
            fit_scaler: Whether to fit the time scaler (True for training, False for prediction)
            
        Returns:
            Enhanced feature matrix with temporal features
        """
        # Calculate time elapsed since the first timestamp
        time_elapsed = (timestamps - timestamps.min())
        
        # Handle different timestamp formats (assume seconds if numeric, otherwise datetime)
        if hasattr(time_elapsed, 'total_seconds'):
            # datetime.timedelta objects
            time_feature = np.array([t.total_seconds() for t in time_elapsed])
        else:
            # Numeric timestamps (already in seconds or similar units)
            time_feature = time_elapsed.astype(float)
        
        # Reshape to column vector
        time_feature = time_feature.reshape(-1, 1)
        
        # Normalize temporal feature to [0, 1] range
        if fit_scaler:
            # Fit scaler on training data
            time_feature_scaled = self.time_scaler.fit_transform(time_feature)
        else:
            # Transform using previously fitted scaler
            time_feature_scaled = self.time_scaler.transform(time_feature)
        
        # Concatenate temporal feature with original features
        X_enhanced = np.hstack([X, time_feature_scaled])
        
        return X_enhanced