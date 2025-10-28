"""
Feature Engineering Module for ChronoCast
Creates time-based and domain-specific features for forecasting
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def create_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Create time-based features from a datetime column
    
    Args:
        df: Input DataFrame
        date_col: Name of the datetime column
    
    Returns:
        DataFrame with additional time features
    """
    df = df.copy()
    
    # Ensure datetime format
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Basic time features
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['day_of_year'] = df[date_col].dt.dayofyear
    df['week_of_year'] = df[date_col].dt.isocalendar().week
    df['quarter'] = df[date_col].dt.quarter
    
    # Binary features
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
    df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df[date_col].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df[date_col].dt.is_quarter_end.astype(int)
    
    # Cyclical encoding for periodic features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df


def create_lag_features(df: pd.DataFrame, 
                       target_col: str, 
                       lags: List[int] = [1, 7, 14, 30]) -> pd.DataFrame:
    """
    Create lag features for time series forecasting
    
    Args:
        df: Input DataFrame (must be sorted by date)
        target_col: Name of the target column
        lags: List of lag periods to create
    
    Returns:
        DataFrame with lag features
    """
    df = df.copy()
    
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    return df


def create_rolling_features(df: pd.DataFrame,
                           target_col: str,
                           windows: List[int] = [7, 14, 30]) -> pd.DataFrame:
    """
    Create rolling window statistics
    
    Args:
        df: Input DataFrame (must be sorted by date)
        target_col: Name of the target column
        windows: List of window sizes
    
    Returns:
        DataFrame with rolling features
    """
    df = df.copy()
    
    for window in windows:
        # Rolling mean
        df[f'{target_col}_rolling_mean_{window}'] = \
            df[target_col].rolling(window=window).mean()
        
        # Rolling standard deviation
        df[f'{target_col}_rolling_std_{window}'] = \
            df[target_col].rolling(window=window).std()
        
        # Rolling min/max
        df[f'{target_col}_rolling_min_{window}'] = \
            df[target_col].rolling(window=window).min()
        
        df[f'{target_col}_rolling_max_{window}'] = \
            df[target_col].rolling(window=window).max()
    
    return df


def create_domain_features(df: pd.DataFrame,
                          categorical_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Create domain-specific features (categorical encoding)
    
    Args:
        df: Input DataFrame
        categorical_cols: List of categorical column names
    
    Returns:
        DataFrame with encoded categorical features
    """
    df = df.copy()
    
    if categorical_cols:
        for col in categorical_cols:
            if col in df.columns:
                # One-hot encoding
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
    
    return df


def create_all_features(df: pd.DataFrame,
                       date_col: str,
                       target_col: str,
                       categorical_cols: Optional[List[str]] = None,
                       lags: List[int] = [1, 7, 14, 30],
                       windows: List[int] = [7, 14, 30]) -> pd.DataFrame:
    """
    Create all features in one go
    
    Args:
        df: Input DataFrame
        date_col: Name of datetime column
        target_col: Name of target column
        categorical_cols: List of categorical columns
        lags: Lag periods
        windows: Rolling window sizes
    
    Returns:
        DataFrame with all features
    """
    # Sort by date
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # Time features
    df = create_time_features(df, date_col)
    
    # Lag features
    df = create_lag_features(df, target_col, lags)
    
    # Rolling features
    df = create_rolling_features(df, target_col, windows)
    
    # Domain features
    if categorical_cols:
        df = create_domain_features(df, categorical_cols)
    
    # Drop rows with NaN values created by lag/rolling features
    df = df.dropna()
    
    return df


if __name__ == "__main__":
    # Example usage
    print("Feature Engineering Module - Example")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'views': np.random.randint(100, 1000, 100),
        'category': np.random.choice(['tech', 'lifestyle', 'business'], 100)
    })
    
    # Create features
    featured_data = create_all_features(
        data, 
        date_col='date', 
        target_col='views',
        categorical_cols=['category']
    )
    
    print(f"\nOriginal shape: {data.shape}")
    print(f"After feature engineering: {featured_data.shape}")
    print(f"\nNew features: {list(featured_data.columns)}")