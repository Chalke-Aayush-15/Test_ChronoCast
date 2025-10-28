"""
Unit tests for evaluation module
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from chronocast.core.evaluation import (
    MetricCalculator,
    evaluate_model,
    compare_models,
    save_evaluation_report,
    load_evaluation_report
)


@pytest.fixture
def sample_predictions():
    """Create sample actual and predicted values"""
    np.random.seed(42)
    y_true = np.random.randn(100) * 10 + 100
    y_pred = y_true + np.random.randn(100) * 2
    y_train = np.random.randn(200) * 10 + 100
    return y_true, y_pred, y_train


@pytest.fixture
def perfect_predictions():
    """Create perfect predictions for testing"""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    return y_true, y_pred


def test_mse_calculation(sample_predictions):
    """Test Mean Squared Error calculation"""
    y_true, y_pred, _ = sample_predictions
    calc = MetricCalculator()
    
    mse = calc.mse(y_true, y_pred)
    assert isinstance(mse, float)
    assert mse >= 0


def test_rmse_calculation(sample_predictions):
    """Test Root Mean Squared Error calculation"""
    y_true, y_pred, _ = sample_predictions
    calc = MetricCalculator()
    
    rmse = calc.rmse(y_true, y_pred)
    mse = calc.mse(y_true, y_pred)
    
    assert isinstance(rmse, float)
    assert rmse >= 0
    assert np.isclose(rmse, np.sqrt(mse))


def test_mae_calculation(sample_predictions):
    """Test Mean Absolute Error calculation"""
    y_true, y_pred, _ = sample_predictions
    calc = MetricCalculator()
    
    mae = calc.mae(y_true, y_pred)
    assert isinstance(mae, float)
    assert mae >= 0


def test_mape_calculation(sample_predictions):
    """Test Mean Absolute Percentage Error calculation"""
    y_true, y_pred, _ = sample_predictions
    calc = MetricCalculator()
    
    mape = calc.mape(y_true, y_pred)
    assert isinstance(mape, float)
    assert mape >= 0


def test_r2_calculation(sample_predictions):
    """Test R-squared calculation"""
    y_true, y_pred, _ = sample_predictions
    calc = MetricCalculator()
    
    r2 = calc.r2(y_true, y_pred)
    assert isinstance(r2, float)
    assert r2 <= 1.0


def test_perfect_predictions_metrics(perfect_predictions):
    """Test metrics with perfect predictions"""
    y_true, y_pred = perfect_predictions
    calc = MetricCalculator()
    
    # Perfect predictions should have zero error
    assert calc.mse(y_true, y_pred) == 0.0
    assert calc.rmse(y_true, y_pred) == 0.0
    assert calc.mae(y_true, y_pred) == 0.0
    
    # R² should be 1.0 (or very close due to floating point)
    assert np.isclose(calc.r2(y_true, y_pred), 1.0)


def test_evaluate_model_default_metrics(sample_predictions):
    """Test evaluate_model with default metrics"""
    y_true, y_pred, y_train = sample_predictions
    
    metrics = evaluate_model(y_true, y_pred, y_train)
    
    # Check that default metrics are present
    expected_metrics = ['MSE', 'RMSE', 'MAE', 'MAPE', 'R²', 'SMAPE', 'Bias']
    for metric in expected_metrics:
        assert metric in metrics
        assert isinstance(metrics[metric], (int, float))


def test_evaluate_model_specific_metrics(sample_predictions):
    """Test evaluate_model with specific metrics"""
    y_true, y_pred, y_train = sample_predictions
    
    specific_metrics = ['rmse', 'mae', 'r2']
    metrics = evaluate_model(y_true, y_pred, y_train, metrics=specific_metrics)
    
    # Check that only specified metrics are present
    assert 'RMSE' in metrics
    assert 'MAE' in metrics
    assert 'R²' in metrics
    assert len(metrics) == 3


def test_evaluate_model_with_mase(sample_predictions):
    """Test MASE calculation"""
    y_true, y_pred, y_train = sample_predictions
    
    metrics = evaluate_model(y_true, y_pred, y_train, metrics=['mase'])
    
    assert 'MASE' in metrics
    assert isinstance(metrics['MASE'], float)


def test_compare_models(sample_predictions):
    """Test model comparison"""
    y_true, y_pred, y_train = sample_predictions
    
    # Create predictions for two models
    y_pred_model1 = y_pred
    y_pred_model2 = y_true + np.random.randn(100) * 5
    
    models = {
        'Model 1': {'y_true': y_true, 'y_pred': y_pred_model1, 'y_train': y_train},
        'Model 2': {'y_true': y_true, 'y_pred': y_pred_model2, 'y_train': y_train}
    }
    
    comparison = compare_models(models)
    
    # Check output is a DataFrame
    assert isinstance(comparison, pd.DataFrame)
    
    # Check it has both models
    assert len(comparison) == 2
    assert 'Model' in comparison.columns
    assert 'Model 1' in comparison['Model'].values
    assert 'Model 2' in comparison['Model'].values
    
    # Check metrics are present
    assert 'RMSE' in comparison.columns
    assert 'MAE' in comparison.columns
    assert 'R²' in comparison.columns


def test_compare_models_sorting(sample_predictions):
    """Test that compare_models sorts correctly"""
    y_true, y_pred, y_train = sample_predictions
    
    # Create one good and one bad model
    y_pred_good = y_true + np.random.randn(100) * 1
    y_pred_bad = y_true + np.random.randn(100) * 10
    
    models = {
        'Bad Model': {'y_true': y_true, 'y_pred': y_pred_bad},
        'Good Model': {'y_true': y_true, 'y_pred': y_pred_good}
    }
    
    comparison = compare_models(models, metric='RMSE', sort_ascending=True)
    
    # Good model should be first (lower RMSE)
    assert comparison.iloc[0]['Model'] == 'Good Model'
    assert comparison.iloc[1]['Model'] == 'Bad Model'


def test_save_load_evaluation_report(sample_predictions):
    """Test saving and loading evaluation reports"""
    y_true, y_pred, y_train = sample_predictions
    
    metrics = evaluate_model(y_true, y_pred, y_train)
    
    # Save report
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
        tmp_path = tmp.name
    
    try:
        save_evaluation_report(metrics, 'Test Model', tmp_path)
        
        # Load report
        loaded_report = load_evaluation_report(tmp_path)
        
        # Check report contents
        assert 'model_name' in loaded_report
        assert loaded_report['model_name'] == 'Test Model'
        assert 'timestamp' in loaded_report
        assert 'metrics' in loaded_report
        
        # Check metrics match
        for key in metrics.keys():
            assert key in loaded_report['metrics']
            assert np.isclose(metrics[key], loaded_report['metrics'][key])
    
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_bias_calculation(sample_predictions):
    """Test bias calculation"""
    y_true, y_pred, _ = sample_predictions
    calc = MetricCalculator()
    
    bias = calc.bias(y_true, y_pred)
    assert isinstance(bias, (int, float, np.number))
    
    # If predictions exactly equal true values, bias should be 0
    bias_perfect = calc.bias(y_true, y_true)
    assert np.isclose(bias_perfect, 0.0)


def test_smape_calculation(sample_predictions):
    """Test SMAPE calculation"""
    y_true, y_pred, _ = sample_predictions
    calc = MetricCalculator()
    
    smape = calc.smape(y_true, y_pred)
    assert isinstance(smape, (int, float, np.number))
    assert smape >= 0
    assert smape <= 200  # SMAPE is bounded between 0 and 200


def test_metrics_with_pandas_series(sample_predictions):
    """Test that metrics work with pandas Series"""
    y_true, y_pred, y_train = sample_predictions
    
    # Convert to Series
    y_true_series = pd.Series(y_true)
    y_pred_series = pd.Series(y_pred)
    y_train_series = pd.Series(y_train)
    
    # Should work without errors
    metrics = evaluate_model(y_true_series, y_pred_series, y_train_series)
    
    assert len(metrics) > 0
    assert all(isinstance(v, (int, float)) for v in metrics.values())


def test_evaluate_model_handles_invalid_metric():
    """Test that evaluate_model handles invalid metric names gracefully"""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    
    # Should not raise error, just skip invalid metric
    metrics = evaluate_model(y_true, y_pred, metrics=['rmse', 'invalid_metric', 'mae'])
    
    assert 'RMSE' in metrics
    assert 'MAE' in metrics
    assert 'invalid_metric' not in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])