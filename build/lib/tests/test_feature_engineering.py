"""
Unit tests for feature engineering module
"""

import pytest
import pandas as pd
import numpy as np
from chronocast.core.feature_engineering import (
    create_time_features,
    create_lag_features,
    create_rolling_features,
    create_domain_features,
    create_all_features
)


@pytest.fixture
def sample_data():
    """Create sample time series data for testing"""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    return pd.DataFrame({
        'date': dates,
        'views': np.random.randint(100, 1000, 100),
        'category': np.random.choice(['tech', 'lifestyle', 'business'], 100)
    })


def test_create_time_features(sample_data):
    """Test time feature creation"""
    result = create_time_features(sample_data, 'date')
    
    # Check if new columns are created
    expected_cols = ['year', 'month', 'day', 'day_of_week', 'is_weekend']
    for col in expected_cols:
        assert col in result.columns
    
    # Check is_weekend logic
    assert result['is_weekend'].isin([0, 1]).all()
    
    # Check cyclical encoding
    assert 'month_sin' in result.columns
    assert 'month_cos' in result.columns


def test_create_lag_features(sample_data):
    """Test lag feature creation"""
    result = create_lag_features(sample_data, 'views', lags=[1, 7])
    
    # Check if lag columns exist
    assert 'views_lag_1' in result.columns
    assert 'views_lag_7' in result.columns
    
    # Check lag values
    assert pd.isna(result['views_lag_1'].iloc[0])
    assert result['views_lag_1'].iloc[1] == result['views'].iloc[0]


def test_create_rolling_features(sample_data):
    """Test rolling feature creation"""
    result = create_rolling_features(sample_data, 'views', windows=[7])
    
    # Check if rolling columns exist
    assert 'views_rolling_mean_7' in result.columns
    assert 'views_rolling_std_7' in result.columns
    assert 'views_rolling_min_7' in result.columns
    assert 'views_rolling_max_7' in result.columns
    
    # Check first values are NaN
    assert pd.isna(result['views_rolling_mean_7'].iloc[0])


def test_create_domain_features(sample_data):
    """Test categorical feature encoding"""
    result = create_domain_features(sample_data, categorical_cols=['category'])
    
    # Check if one-hot encoded columns exist
    assert any('category_' in col for col in result.columns)
    
    # Original column should still exist
    assert 'category' in result.columns


def test_create_all_features(sample_data):
    """Test complete feature creation pipeline"""
    result = create_all_features(
        sample_data,
        date_col='date',
        target_col='views',
        categorical_cols=['category'],
        lags=[1, 7],
        windows=[7]
    )
    
    # Check that we have more features
    assert result.shape[1] > sample_data.shape[1]
    
    # Check that rows with NaN are dropped
    assert not result.isna().any().any()
    
    # Check that data is sorted by date
    assert result['date'].is_monotonic_increasing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])