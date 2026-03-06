"""
ChronoCast - Advanced Time Series Forecasting Library
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import logging

# Import missing functions from main chronocast package
try:
    from chronocast.core.model_wrapper import ChronoModel
    from chronocast.core.evaluation import compare_models
    from chronocast.core.explainability import ModelExplainer
except ImportError:
    # Fallback if chronocast package is not available
    ChronoModel = None
    compare_models = None
    ModelExplainer = None

logger = logging.getLogger(__name__)


class ChronoCastModel:
    """
    Advanced ChronoCast forecasting model with trend, seasonality, and anomaly detection
    """
    
    def __init__(self, trend_sensitivity='medium', seasonality_detection='auto', 
                 anomaly_threshold='2sigma', confidence_level=95):
        self.trend_sensitivity = trend_sensitivity
        self.seasonality_detection = seasonality_detection
        self.anomaly_threshold = anomaly_threshold
        self.confidence_level = confidence_level
        self.trend_info = None
        self.seasonality_info = None
        self.anomaly_info = None
        self.training_history = {}
        
    def detect_trend(self, data):
        """Detect trend in time series data"""
        if len(data) < 3:
            return {'trend': 'insufficient_data', 'strength': 0}
        
        first_half = data[:len(data)//2]
        second_half = data[len(data)//2:]
        
        first_avg = np.mean(first_half)
        second_avg = np.mean(second_half)
        
        trend_strength = abs((second_avg - first_avg) / first_avg) * 100
        
        if trend_strength < 5:
            return {'trend': 'stable', 'strength': trend_strength}
        elif trend_strength < 15:
            return {'trend': 'weak', 'strength': trend_strength, 'direction': 'upward' if second_avg > first_avg else 'downward'}
        elif trend_strength < 30:
            return {'trend': 'moderate', 'strength': trend_strength, 'direction': 'upward' if second_avg > first_avg else 'downward'}
        else:
            return {'trend': 'strong', 'strength': trend_strength, 'direction': 'upward' if second_avg > first_avg else 'downward'}
    
    def detect_seasonality(self, data, dates):
        """Detect seasonal patterns"""
        if len(data) < 14:
            return {'seasonality': 'insufficient_data', 'pattern': None}
        
        df = pd.DataFrame({'date': dates, 'value': data})
        df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
        
        weekly_averages = df.groupby('day_of_week')['value'].mean()
        overall_avg = np.mean(data)
        
        seasonality_strength = ((weekly_averages.max() - weekly_averages.min()) / overall_avg) * 100
        
        if seasonality_strength < 10:
            return {'seasonality': 'weak', 'pattern': weekly_averages.tolist(), 'strength': seasonality_strength}
        elif seasonality_strength < 25:
            return {'seasonality': 'moderate', 'pattern': weekly_averages.tolist(), 'strength': seasonality_strength}
        else:
            return {'seasonality': 'strong', 'pattern': weekly_averages.tolist(), 'strength': seasonality_strength}
    
    def detect_anomalies(self, data):
        """Detect anomalies using statistical methods"""
        if len(data) < 3:
            return {'anomalies': [], 'anomaly_detected': False}
        
        mean = np.mean(data)
        std = np.std(data)
        
        # Set threshold based on sigma level
        sigma_levels = {'1sigma': 1, '2sigma': 2, '3sigma': 3}
        threshold_multiplier = sigma_levels.get(self.anomaly_threshold, 2)
        threshold = mean + (threshold_multiplier * std)
        
        anomalies = []
        for i, value in enumerate(data):
            if value > threshold:
                severity = 'extreme' if value > (mean + 3 * std) else 'moderate'
                anomalies.append({
                    'index': i,
                    'value': value,
                    'threshold': threshold,
                    'severity': severity
                })
        
        return {
            'anomalies': anomalies,
            'anomaly_detected': len(anomalies) > 0,
            'anomaly_count': len(anomalies),
            'confidence': 1 - (len(anomalies) / len(data))
        }
    
    def fit(self, X, y):
        """Fit the ChronoCast model"""
        start_time = time.time()
        
        # Detect patterns in target variable
        self.trend_info = self.detect_trend(y.values)
        self.seasonality_info = self.detect_seasonality(y.values, X.index if hasattr(X, 'index') else range(len(y)))
        self.anomaly_info = self.detect_anomalies(y.values)
        
        # Create enhanced features
        enhanced_X = self._create_enhanced_features(X, y)
        
        # Train ensemble model (combination of multiple algorithms)
        self.models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=0.01),
            'rf': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
            'gbm': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        }
        
        # Train all models
        for name, model in self.models.items():
            model.fit(enhanced_X, y)
        
        self.training_history = {
            'training_time': time.time() - start_time,
            'trend_analysis': self.trend_info,
            'seasonality_analysis': self.seasonality_info,
            'anomaly_analysis': self.anomaly_info,
            'models_trained': list(self.models.keys())
        }
        
        logger.info(f"ChronoCast model trained in {self.training_history['training_time']:.2f}s")
        return self
    
    def _create_enhanced_features(self, X, y):
        """Create enhanced features based on detected patterns"""
        enhanced_X = X.copy()
        
        # Add trend features
        if self.trend_info['trend'] != 'stable':
            trend_direction = 1 if self.trend_info.get('direction') == 'upward' else -1
            enhanced_X['trend_feature'] = np.arange(len(X)) * trend_direction * 0.01
        
        # Add seasonality features
        if self.seasonality_info['pattern']:
            day_of_week = pd.to_datetime(X.index if hasattr(X, 'index') else range(len(X))).dayofweek
            seasonal_pattern = self.seasonality_info['pattern']
            enhanced_X['seasonal_feature'] = [seasonal_pattern[d] / np.mean(seasonal_pattern) for d in day_of_week]
        
        # Add anomaly features
        if self.anomaly_info['anomaly_detected']:
            enhanced_X['anomaly_feature'] = (y > np.mean(y)).astype(int)
        
        return enhanced_X
    
    def predict(self, X):
        """Make predictions using ensemble of models"""
        if not hasattr(self, 'models'):
            raise ValueError("Model must be fitted before making predictions")
        
        # Create enhanced features for prediction
        enhanced_X = self._create_enhanced_features(X, pd.Series([0] * len(X)))
        
        # Ensemble predictions (weighted average)
        predictions = []
        weights = {'linear': 0.2, 'ridge': 0.2, 'rf': 0.3, 'gbm': 0.3}
        
        for name, model in self.models.items():
            pred = model.predict(enhanced_X)
            predictions.append(pred * weights[name])
        
        # Weighted ensemble
        final_prediction = np.sum(predictions, axis=0)
        
        # Apply confidence intervals based on confidence level
        confidence_multiplier = self.confidence_level / 100
        final_prediction = final_prediction * confidence_multiplier
        
        return final_prediction


class TimeSeriesDataLoader:
    """Utility class for loading and validating time series data"""
    
    def load_csv(self, file_path, date_col=None):
        """Load CSV file with proper parsing"""
        try:
            data = pd.read_csv(file_path)
            if date_col and date_col in data.columns:
                data[date_col] = pd.to_datetime(data[date_col])
            return data
        except Exception as e:
            raise ValueError(f"Error loading CSV: {str(e)}")
    
    def validate_time_series(self, date_col, target_col):
        """Validate time series data structure"""
        validation = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        if not date_col:
            validation['valid'] = False
            validation['errors'].append("Date column not specified")
        
        if not target_col:
            validation['valid'] = False
            validation['errors'].append("Target column not specified")
        
        return validation


def create_all_features(data, date_col, target_col, lags=None, windows=None):
    """Create all features for time series forecasting"""
    if lags is None:
        lags = [1, 7, 14]
    if windows is None:
        windows = [7, 14]
    
    featured_data = data.copy()
    
    # Create lag features
    for lag in lags:
        featured_data[f'lag_{lag}'] = data[target_col].shift(lag)
    
    # Create rolling window features
    for window in windows:
        featured_data[f'rolling_mean_{window}'] = data[target_col].rolling(window=window).mean()
        featured_data[f'rolling_std_{window}'] = data[target_col].rolling(window=window).std()
    
    # Create time-based features
    if date_col in data.columns:
        data[date_col] = pd.to_datetime(data[date_col])
        featured_data['day_of_week'] = data[date_col].dt.dayofweek
        featured_data['month'] = data[date_col].dt.month
        featured_data['quarter'] = data[date_col].dt.quarter
    
    return featured_data


def evaluate_model(y_true, y_pred, y_train=None):
    """Evaluate model performance"""
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred)
    }
    
    # Calculate MAPE if y_train is available
    if y_train is not None:
        baseline = np.mean(y_train)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        metrics['mape'] = mape
        metrics['baseline_mae'] = mean_absolute_error(y_true, [baseline] * len(y_true))