"""
Unit tests for model wrapper module
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from sklearn.datasets import make_regression
from chronocast.core.model_wrapper import ChronoModel, ModelRegistry, model_registry


@pytest.fixture
def sample_data():
    """Create sample regression data"""
    X, y = make_regression(n_samples=200, n_features=10, random_state=42)
    X_train, y_train = X[:150], y[:150]
    X_test, y_test = X[150:], y[150:]
    return X_train, y_train, X_test, y_test


@pytest.fixture
def sample_df_data():
    """Create sample DataFrame data"""
    X, y = make_regression(n_samples=200, n_features=10, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    y_series = pd.Series(y, name='target')
    
    X_train = X_df[:150]
    y_train = y_series[:150]
    X_test = X_df[150:]
    y_test = y_series[150:]
    
    return X_train, y_train, X_test, y_test


def test_model_initialization():
    """Test model initialization with different types"""
    models = ['linear', 'ridge', 'lasso', 'rf', 'dt', 'gbm', 'xgb']
    
    for model_type in models:
        model = ChronoModel(model_type=model_type)
        assert model.model_type == model_type
        assert not model.is_fitted
        assert model.feature_names is None


def test_invalid_model_type():
    """Test that invalid model type raises error"""
    with pytest.raises(ValueError):
        ChronoModel(model_type='invalid_model')


def test_model_fit_predict(sample_data):
    """Test basic fit and predict functionality"""
    X_train, y_train, X_test, y_test = sample_data
    
    model = ChronoModel(model_type='linear')
    
    # Fit
    model.fit(X_train, y_train)
    assert model.is_fitted
    assert model.training_history['n_samples'] == 150
    assert model.training_history['n_features'] == 10
    
    # Predict
    predictions = model.predict(X_test)
    assert len(predictions) == len(y_test)
    assert isinstance(predictions, np.ndarray)


def test_model_with_dataframe(sample_df_data):
    """Test model with DataFrame input"""
    X_train, y_train, X_test, y_test = sample_df_data
    
    model = ChronoModel(model_type='rf')
    model.fit(X_train, y_train)
    
    # Check feature names are stored
    assert model.feature_names is not None
    assert len(model.feature_names) == 10
    
    predictions = model.predict(X_test)
    assert len(predictions) == len(y_test)


def test_predict_before_fit():
    """Test that predict raises error before fitting"""
    model = ChronoModel(model_type='linear')
    X = np.random.randn(10, 5)
    
    with pytest.raises(ValueError, match="Model must be fitted"):
        model.predict(X)


def test_feature_importance_tree_models(sample_df_data):
    """Test feature importance for tree-based models"""
    X_train, y_train, X_test, y_test = sample_df_data
    
    tree_models = ['rf', 'dt', 'gbm', 'xgb']
    
    for model_type in tree_models:
        model = ChronoModel(model_type=model_type)
        model.fit(X_train, y_train)
        
        importance = model.get_feature_importance()
        assert importance is not None
        assert isinstance(importance, pd.DataFrame)
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns
        assert len(importance) == 10


def test_feature_importance_linear_models(sample_data):
    """Test that linear models return None for feature importance"""
    X_train, y_train, X_test, y_test = sample_data
    
    model = ChronoModel(model_type='linear')
    model.fit(X_train, y_train)
    
    importance = model.get_feature_importance()
    assert importance is None


def test_model_save_load(sample_data):
    """Test model saving and loading"""
    X_train, y_train, X_test, y_test = sample_data
    
    # Train model
    model = ChronoModel(model_type='rf', n_estimators=50)
    model.fit(X_train, y_train)
    predictions_before = model.predict(X_test)
    
    # Save model
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
        tmp_path = tmp.name
    
    try:
        model.save(tmp_path)
        
        # Load model
        loaded_model = ChronoModel.load(tmp_path)
        
        # Check loaded model
        assert loaded_model.model_type == 'rf'
        assert loaded_model.is_fitted
        
        # Compare predictions
        predictions_after = loaded_model.predict(X_test)
        np.testing.assert_array_almost_equal(predictions_before, predictions_after)
    
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_get_set_params(sample_data):
    """Test getting and setting parameters"""
    X_train, y_train, X_test, y_test = sample_data
    
    model = ChronoModel(model_type='rf', n_estimators=50)
    
    # Get params
    params = model.get_params()
    assert params['n_estimators'] == 50
    
    # Set params
    model.set_params(n_estimators=100)
    params = model.get_params()
    assert params['n_estimators'] == 100


def test_training_history(sample_data):
    """Test training history tracking"""
    X_train, y_train, X_test, y_test = sample_data
    
    model = ChronoModel(model_type='xgb')
    model.fit(X_train, y_train)
    
    history = model.get_training_info()
    
    assert 'model_type' in history
    assert 'training_time' in history
    assert 'n_samples' in history
    assert 'n_features' in history
    assert 'timestamp' in history
    assert history['model_type'] == 'xgb'
    assert history['n_samples'] == 150


def test_model_registry():
    """Test custom model registration"""
    from sklearn.linear_model import ElasticNet
    
    registry = ModelRegistry()
    
    # Register custom model
    registry.register('elastic', ElasticNet)
    
    # Check it's in available models
    assert 'elastic' in registry.list_models()
    
    # Create model with custom type
    model = ChronoModel(model_type='elastic')
    assert model.model_type == 'elastic'
    
    # Unregister
    registry.unregister('elastic')
    assert 'elastic' not in registry.list_models()


def test_model_repr():
    """Test model string representation"""
    model = ChronoModel(model_type='linear')
    repr_str = repr(model)
    
    assert 'linear' in repr_str
    assert 'not fitted' in repr_str
    
    # After fitting
    X, y = make_regression(n_samples=100, n_features=5)
    model.fit(X, y)
    repr_str = repr(model)
    
    assert 'fitted' in repr_str


def test_multiple_models_comparison(sample_data):
    """Test training multiple models for comparison"""
    X_train, y_train, X_test, y_test = sample_data
    
    models = {
        'linear': ChronoModel('linear'),
        'rf': ChronoModel('rf', n_estimators=50),
        'xgb': ChronoModel('xgb', n_estimators=50)
    }
    
    # Train all models
    for name, model in models.items():
        model.fit(X_train, y_train)
        assert model.is_fitted
        
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])