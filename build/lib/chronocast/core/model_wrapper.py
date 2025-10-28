"""
Model Wrapper Module for ChronoCast
Provides unified interface for multiple ML models
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
import joblib
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import json
from datetime import datetime


class ChronoModel:
    """
    Unified model wrapper for time series forecasting
    Supports multiple model types with consistent interface
    """
    
    AVAILABLE_MODELS = {
        'linear': LinearRegression,
        'ridge': Ridge,
        'lasso': Lasso,
        'rf': RandomForestRegressor,
        'dt': DecisionTreeRegressor,
        'gbm': GradientBoostingRegressor,
        'xgb': XGBRegressor
    }
    
    def __init__(self, model_type: str = 'linear', **kwargs):
        """
        Initialize ChronoModel
        
        Args:
            model_type: Type of model ('linear', 'ridge', 'lasso', 'rf', 'dt', 'gbm', 'xgb')
            **kwargs: Model-specific parameters
        """
        if model_type not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model type must be one of {list(self.AVAILABLE_MODELS.keys())}")
        
        self.model_type = model_type
        self.model_params = kwargs
        self.model = self._initialize_model()
        self.is_fitted = False
        self.feature_names = None
        self.training_history = {
            'model_type': model_type,
            'params': kwargs,
            'training_time': None,
            'n_samples': None,
            'n_features': None
        }
    
    def _initialize_model(self):
        """Initialize the underlying model with parameters"""
        model_class = self.AVAILABLE_MODELS[self.model_type]
        
        # Default parameters for each model type
        default_params = self._get_default_params()
        
        # Merge default with user-provided params
        params = {**default_params, **self.model_params}
        
        return model_class(**params)
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for each model type"""
        defaults = {
            'linear': {},
            'ridge': {'alpha': 1.0},
            'lasso': {'alpha': 1.0},
            'rf': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'random_state': 42
            },
            'dt': {
                'max_depth': 10,
                'min_samples_split': 5,
                'random_state': 42
            },
            'gbm': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': 42
            },
            'xgb': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': 42
            }
        }
        return defaults.get(self.model_type, {})
    
    def fit(self, X, y):
        """
        Train the model
        
        Args:
            X: Feature matrix (DataFrame or array)
            y: Target variable (Series or array)
        
        Returns:
            self
        """
        start_time = datetime.now()
        
        # Store feature names if DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        
        if isinstance(y, pd.Series):
            y = y.values
        
        # Train model
        self.model.fit(X, y)
        self.is_fitted = True
        
        # Update training history
        training_time = (datetime.now() - start_time).total_seconds()
        self.training_history.update({
            'training_time': training_time,
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'timestamp': datetime.now().isoformat()
        })
        
        return self
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Feature matrix (DataFrame or array)
        
        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance (for tree-based models)
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Check if model has feature_importances_
        if not hasattr(self.model, 'feature_importances_'):
            return None
        
        importances = self.model.feature_importances_
        
        if self.feature_names:
            return pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
        else:
            return pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(len(importances))],
                'importance': importances
            }).sort_values('importance', ascending=False)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        return self.model.get_params()
    
    def set_params(self, **params):
        """Set model parameters"""
        self.model.set_params(**params)
        self.model_params.update(params)
        return self
    
    def save(self, filepath: str):
        """
        Save model to disk
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """
        Load model from disk
        
        Args:
            filepath: Path to the saved model
        
        Returns:
            ChronoModel instance
        """
        model_data = joblib.load(filepath)
        
        # Create new instance
        instance = cls(model_type=model_data['model_type'])
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.training_history = model_data['training_history']
        instance.is_fitted = model_data['is_fitted']
        
        return instance
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get training metadata"""
        return self.training_history
    
    def __repr__(self):
        status = "fitted" if self.is_fitted else "not fitted"
        return f"ChronoModel(type='{self.model_type}', status='{status}')"


class ModelRegistry:
    """
    Registry for custom models
    Allows users to register their own models
    """
    
    def __init__(self):
        self.custom_models = {}
    
    def register(self, name: str, model_class):
        """
        Register a custom model
        
        Args:
            name: Name for the custom model
            model_class: Model class (must have fit/predict methods)
        """
        # Validate model has required methods
        required_methods = ['fit', 'predict']
        for method in required_methods:
            if not hasattr(model_class, method):
                raise ValueError(f"Model must implement {method} method")
        
        self.custom_models[name] = model_class
        ChronoModel.AVAILABLE_MODELS[name] = model_class
        print(f"Model '{name}' registered successfully")
    
    def list_models(self):
        """List all available models"""
        return list(ChronoModel.AVAILABLE_MODELS.keys())
    
    def unregister(self, name: str):
        """Remove a custom model from registry"""
        if name in self.custom_models:
            del self.custom_models[name]
            del ChronoModel.AVAILABLE_MODELS[name]
            print(f"Model '{name}' unregistered")
        else:
            print(f"Model '{name}' not found in registry")


# Global registry instance
model_registry = ModelRegistry()


if __name__ == "__main__":
    # Example usage
    print("Model Wrapper - Example Usage\n")
    
    # Create sample data
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=1000, n_features=20, random_state=42)
    X_train, y_train = X[:800], y[:800]
    X_test, y_test = X[800:], y[800:]
    
    # Test different models
    models = ['linear', 'rf', 'xgb']
    
    for model_type in models:
        print(f"\n{'='*50}")
        print(f"Testing {model_type.upper()} model")
        print('='*50)
        
        # Initialize and train
        model = ChronoModel(model_type=model_type)
        print(f"Model: {model}")
        
        model.fit(X_train, y_train)
        print(f"Training completed in {model.training_history['training_time']:.2f} seconds")
        
        # Predict
        predictions = model.predict(X_test)
        print(f"Predictions shape: {predictions.shape}")
        
        # Feature importance (if available)
        importance = model.get_feature_importance()
        if importance is not None:
            print(f"\nTop 5 important features:")
            print(importance.head())
    
    print("\n\nAvailable models:", model_registry.list_models())