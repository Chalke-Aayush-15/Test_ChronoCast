"""
ChronoCast - Advanced Time Series Forecasting Library
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import logging
import sys
sys.path.append('c:\\Users\\sairajkale\\OneDrive\\Desktop\\ChronoCast\\Test_ChronoCast')

# Import missing functions from main chronocast package
try:
    from chronocast.core.model_wrapper import ChronoModel
    from chronocast.core.evaluation import compare_models
    from chronocast.core.explainability import ModelExplainer
except ImportError:
    # Fallback if chronocast package is not available
    print("Warning: ChronoModel not imported, using fallback implementation")
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
        # Ensure confidence_level is an integer
        self.confidence_level = int(confidence_level) if confidence_level else 95
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
            'gbm': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
            'xgb': XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, objective='reg:squarederror')
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
        else:
            enhanced_X['trend_feature'] = 0
        
        # Add seasonality features (simplified - use day of week from index or position)
        if self.seasonality_info['pattern']:
            if hasattr(X, 'index'):
                try:
                    day_of_week = pd.to_datetime(X.index).dayofweek
                except:
                    day_of_week = np.arange(len(X)) % 7
            else:
                day_of_week = np.arange(len(X)) % 7
            
            seasonal_pattern = self.seasonality_info['pattern']
            # Use modulo to prevent index out of bounds
            day_indices = [d % len(seasonal_pattern) for d in day_of_week]
            enhanced_X['seasonal_feature'] = [seasonal_pattern[d] / np.mean(seasonal_pattern) for d in day_indices]
        else:
            enhanced_X['seasonal_feature'] = 1.0
        
        # Add anomaly features
        if self.anomaly_info['anomaly_detected']:
            enhanced_X['anomaly_feature'] = (np.array(y) > np.mean(y)).astype(int)
        else:
            enhanced_X['anomaly_feature'] = 0
        
        return enhanced_X
    
    def predict(self, X):
        """Make predictions using ensemble of models"""
        if not hasattr(self, 'models'):
            raise ValueError("Model must be fitted before making predictions")
        
        # Create enhanced features for prediction
        # Use dummy y values for prediction (since we don't have actual y during prediction)
        dummy_y = np.zeros(len(X))
        enhanced_X = self._create_enhanced_features(X, dummy_y)
        
        # Ensure enhanced_X has same number of features as training data
        # Get expected feature count from first model
        expected_features = None
        for name, model in self.models.items():
            if hasattr(model, 'n_features_in_'):
                expected_features = model.n_features_in_
                break
        
        if expected_features and len(enhanced_X.columns) != expected_features:
            # Pad or truncate features to match expected count
            current_cols = list(enhanced_X.columns)
            if len(current_cols) < expected_features:
                # Add missing columns with zeros
                for i in range(len(current_cols), expected_features):
                    enhanced_X[f'missing_feature_{i}'] = 0
            else:
                # Truncate extra columns
                enhanced_X = enhanced_X.iloc[:, :expected_features]
        
        # Ensemble predictions (weighted average)
        predictions = []
        weights = {'linear': 0.15, 'ridge': 0.15, 'rf': 0.25, 'gbm': 0.2, 'xgb': 0.25}
        
        for name, model in self.models.items():
            try:
                pred = model.predict(enhanced_X)
                predictions.append(pred * weights[name])
            except Exception as e:
                print(f"Warning: Model {name} prediction failed: {str(e)}")
                # Use zero prediction for failed model
                predictions.append(np.zeros(len(X)) * weights[name])
        
        # Weighted ensemble
        final_prediction = np.sum(predictions, axis=0)
        
        return final_prediction
    
    def get_feature_importance(self):
        """Get feature importance from ensemble models"""
        if not hasattr(self, 'models'):
            return None
        
        # Get feature importance from tree-based models
        importances = {}
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances[name] = model.feature_importances_
        
        # Average importances from all models that have them
        if importances:
            avg_importance = np.mean(list(importances.values()), axis=0)
            return pd.Series(avg_importance)
        
        return None
    
    def save(self, filepath):
        """Save the model to disk"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath):
        """Load the model from disk"""
        import pickle
        with open(filepath, 'rb') as f:
            return pickle.load(f)


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
    
    # Handle categorical variables - one-hot encode them
    categorical_cols = featured_data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col not in [date_col, target_col]:
            # One-hot encode categorical variables
            dummies = pd.get_dummies(featured_data[col], prefix=col)
            featured_data = pd.concat([featured_data, dummies], axis=1)
            featured_data.drop(col, axis=1, inplace=True)
    
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
    
    # Handle NaN values - fill with appropriate values
    # For lag features: use forward fill then backward fill
    lag_cols = [col for col in featured_data.columns if col.startswith('lag_')]
    for col in lag_cols:
        featured_data[col] = featured_data[col].ffill().bfill()
    
    # For rolling features: use forward fill then backward fill
    rolling_cols = [col for col in featured_data.columns if 'rolling_' in col]
    for col in rolling_cols:
        featured_data[col] = featured_data[col].ffill().bfill()
    
    # For any remaining NaN values: fill with 0 or mean
    featured_data = featured_data.fillna(0)
    
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