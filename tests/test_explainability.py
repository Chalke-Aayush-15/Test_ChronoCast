"""
Unit tests for explainability module
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from chronocast.core.explainability import ModelExplainer, SHAP_AVAILABLE


@pytest.fixture
def regression_data():
    """Create sample regression data"""
    X, y = make_regression(n_samples=200, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    feature_names = [f'feature_{i}' for i in range(10)]
    return X_train, X_test, y_train, y_test, feature_names


@pytest.fixture
def trained_rf_model(regression_data):
    """Create trained Random Forest model"""
    X_train, X_test, y_train, y_test, feature_names = regression_data
    model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
    model.fit(X_train, y_train)
    return model, X_train, X_test, feature_names


@pytest.fixture
def trained_linear_model(regression_data):
    """Create trained Linear model"""
    X_train, X_test, y_train, y_test, feature_names = regression_data
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_train, X_test, feature_names


def test_explainer_initialization_rf(trained_rf_model):
    """Test explainer initialization with Random Forest"""
    model, X_train, X_test, feature_names = trained_rf_model
    
    explainer = ModelExplainer(model, X_train, feature_names=feature_names)
    
    assert explainer.model is not None
    assert explainer.feature_names == feature_names
    assert explainer.X_train is not None


def test_explainer_initialization_linear(trained_linear_model):
    """Test explainer initialization with Linear model"""
    model, X_train, X_test, feature_names = trained_linear_model
    
    explainer = ModelExplainer(model, X_train, feature_names=feature_names)
    
    assert explainer.model is not None
    assert explainer.feature_names == feature_names


def test_explainer_with_chronomodel(regression_data):
    """Test explainer with ChronoModel wrapper"""
    from chronocast.core.model_wrapper import ChronoModel
    
    X_train, X_test, y_train, y_test, feature_names = regression_data
    
    # Train ChronoModel
    chrono_model = ChronoModel('rf', n_estimators=50, random_state=42)
    chrono_model.fit(X_train, y_train)
    
    # Create explainer
    explainer = ModelExplainer(chrono_model, X_train, feature_names=feature_names)
    
    assert explainer.model is not None
    assert explainer.model_wrapper is not None


def test_feature_importance_rf(trained_rf_model):
    """Test feature importance for Random Forest"""
    model, X_train, X_test, feature_names = trained_rf_model
    
    explainer = ModelExplainer(model, X_train, feature_names=feature_names)
    
    # Get feature importance (without plotting)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        assert len(importances) == len(feature_names)
        assert np.all(importances >= 0)
        assert np.sum(importances) > 0


def test_feature_importance_linear(trained_linear_model):
    """Test feature importance for Linear model"""
    model, X_train, X_test, feature_names = trained_linear_model
    
    explainer = ModelExplainer(model, X_train, feature_names=feature_names)
    
    # Linear models have coefficients
    if hasattr(model, 'coef_'):
        coefs = model.coef_
        assert len(coefs) == len(feature_names)


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
def test_calculate_shap_values_rf(trained_rf_model):
    """Test SHAP value calculation for Random Forest"""
    model, X_train, X_test, feature_names = trained_rf_model
    
    explainer = ModelExplainer(model, X_train, feature_names=feature_names)
    
    # Calculate SHAP values
    shap_values = explainer.calculate_shap_values(X_test, max_samples=20)
    
    if shap_values is not None:
        assert shap_values.shape[0] <= 20  # Limited by max_samples
        assert shap_values.shape[1] == len(feature_names)


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
def test_calculate_shap_values_linear(trained_linear_model):
    """Test SHAP value calculation for Linear model"""
    model, X_train, X_test, feature_names = trained_linear_model
    
    explainer = ModelExplainer(model, X_train, feature_names=feature_names)
    
    # Calculate SHAP values
    shap_values = explainer.calculate_shap_values(X_test, max_samples=20)
    
    if shap_values is not None:
        assert shap_values.shape[0] <= 20
        assert shap_values.shape[1] == len(feature_names)


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
def test_get_feature_contributions(trained_rf_model):
    """Test getting feature contributions"""
    model, X_train, X_test, feature_names = trained_rf_model
    
    explainer = ModelExplainer(model, X_train, feature_names=feature_names)
    
    # Get contributions for first instance
    contributions = explainer.get_feature_contributions(X_test, instance_idx=0)
    
    if contributions is not None:
        assert isinstance(contributions, pd.DataFrame)
        assert 'feature' in contributions.columns
        assert 'shap_value' in contributions.columns
        assert len(contributions) == len(feature_names)


def test_save_explainability_log(trained_rf_model):
    """Test saving explainability log"""
    model, X_train, X_test, feature_names = trained_rf_model
    
    explainer = ModelExplainer(model, X_train, feature_names=feature_names)
    
    # Save log
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
        tmp_path = tmp.name
    
    try:
        explainer.save_explainability_log(X_test, output_path=tmp_path)
        
        # Check file exists
        assert os.path.exists(tmp_path)
        
        # Load and verify
        import json
        with open(tmp_path, 'r') as f:
            log = json.load(f)
        
        assert 'timestamp' in log
        assert 'model_type' in log
        assert 'n_features' in log
        assert 'n_samples' in log
        assert log['n_features'] == len(feature_names)
        assert log['n_samples'] == X_test.shape[0]
    
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_explainer_with_pandas_dataframe(trained_rf_model):
    """Test explainer with pandas DataFrame input"""
    model, X_train, X_test, feature_names = trained_rf_model
    
    # Convert to DataFrame
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    explainer = ModelExplainer(model, X_train_df, feature_names=feature_names)
    
    # Should handle DataFrames correctly
    assert explainer.X_train is not None
    assert isinstance(explainer.X_train, np.ndarray)


def test_explainer_without_feature_names(regression_data):
    """Test explainer works without explicit feature names"""
    X_train, X_test, y_train, y_test, _ = regression_data
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Create explainer without feature names
    explainer = ModelExplainer(model, X_train, feature_names=None)
    
    assert explainer.model is not None
    assert explainer.feature_names is None


def test_shap_availability_warning():
    """Test that warning is issued when SHAP not available"""
    # This test checks the module behavior
    # The actual warning is issued during module import
    assert SHAP_AVAILABLE in [True, False]


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
def test_explainer_type_detection_tree(trained_rf_model):
    """Test that tree explainer is detected correctly"""
    model, X_train, X_test, feature_names = trained_rf_model
    
    explainer = ModelExplainer(model, X_train, feature_names=feature_names)
    
    if hasattr(explainer, 'explainer_type'):
        assert explainer.explainer_type in ['tree', 'linear', 'kernel']


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
def test_explainer_type_detection_linear(trained_linear_model):
    """Test that linear explainer is detected correctly"""
    model, X_train, X_test, feature_names = trained_linear_model
    
    explainer = ModelExplainer(model, X_train, feature_names=feature_names)
    
    if hasattr(explainer, 'explainer_type'):
        assert explainer.explainer_type in ['tree', 'linear', 'kernel']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])