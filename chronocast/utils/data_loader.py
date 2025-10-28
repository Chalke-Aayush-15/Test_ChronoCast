"""
Data Loader Module for ChronoCast
Utilities for loading and preprocessing time series data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union, List, Dict, Any
from datetime import datetime
import warnings


class TimeSeriesDataLoader:
    """
    Load and preprocess time series data
    """
    
    def __init__(self):
        """Initialize data loader"""
        self.data = None
        self.metadata = {}
    
    def load_csv(self, 
                 filepath: str,
                 date_col: Optional[str] = None,
                 parse_dates: bool = True,
                 **kwargs) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Args:
            filepath: Path to CSV file
            date_col: Name of date column
            parse_dates: Whether to parse dates
            **kwargs: Additional arguments for pd.read_csv
        
        Returns:
            Loaded DataFrame
        """
        try:
            if parse_dates and date_col:
                self.data = pd.read_csv(filepath, parse_dates=[date_col], **kwargs)
            else:
                self.data = pd.read_csv(filepath, **kwargs)
            
            self.metadata['source'] = filepath
            self.metadata['n_rows'] = len(self.data)
            self.metadata['n_cols'] = len(self.data.columns)
            self.metadata['columns'] = list(self.data.columns)
            
            print(f"✓ Loaded {self.metadata['n_rows']} rows, {self.metadata['n_cols']} columns")
            return self.data
        
        except Exception as e:
            raise ValueError(f"Error loading CSV: {str(e)}")
    
    def validate_time_series(self,
                            date_col: str,
                            target_col: str,
                            check_gaps: bool = True,
                            check_duplicates: bool = True) -> Dict[str, Any]:
        """
        Validate time series data
        
        Args:
            date_col: Name of date column
            target_col: Name of target column
            check_gaps: Whether to check for date gaps
            check_duplicates: Whether to check for duplicates
        
        Returns:
            Dictionary with validation results
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv first.")
        
        validation = {
            'is_valid': True,
            'issues': []
        }
        
        # Check if columns exist
        if date_col not in self.data.columns:
            validation['is_valid'] = False
            validation['issues'].append(f"Date column '{date_col}' not found")
            return validation
        
        if target_col not in self.data.columns:
            validation['is_valid'] = False
            validation['issues'].append(f"Target column '{target_col}' not found")
            return validation
        
        # Check for missing values
        missing_dates = self.data[date_col].isna().sum()
        missing_target = self.data[target_col].isna().sum()
        
        if missing_dates > 0:
            validation['issues'].append(f"{missing_dates} missing dates")
            validation['is_valid'] = False
        
        if missing_target > 0:
            validation['issues'].append(f"{missing_target} missing target values")
            validation['is_valid'] = False
        
        # Check for duplicates
        if check_duplicates:
            duplicates = self.data[date_col].duplicated().sum()
            if duplicates > 0:
                validation['issues'].append(f"{duplicates} duplicate dates")
                validation['is_valid'] = False
        
        # Check for gaps in dates
        if check_gaps and validation['is_valid']:
            sorted_dates = pd.to_datetime(self.data[date_col]).sort_values()
            date_diffs = sorted_dates.diff()
            
            # Infer frequency
            most_common_diff = date_diffs.mode()[0] if len(date_diffs.mode()) > 0 else None
            
            if most_common_diff:
                gaps = (date_diffs != most_common_diff).sum() - 1  # -1 for first NaN
                if gaps > 0:
                    validation['issues'].append(f"{gaps} date gaps detected")
                    validation['inferred_frequency'] = str(most_common_diff)
        
        # Check data range
        validation['date_range'] = {
            'start': self.data[date_col].min(),
            'end': self.data[date_col].max(),
            'n_periods': len(self.data)
        }
        
        validation['target_stats'] = {
            'min': float(self.data[target_col].min()),
            'max': float(self.data[target_col].max()),
            'mean': float(self.data[target_col].mean()),
            'std': float(self.data[target_col].std())
        }
        
        return validation
    
    def handle_missing_values(self,
                             method: str = 'forward_fill',
                             columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Handle missing values in data
        
        Args:
            method: Method to handle missing values
                   ('forward_fill', 'backward_fill', 'interpolate', 'drop')
            columns: Specific columns to process (None for all)
        
        Returns:
            DataFrame with handled missing values
        """
        if self.data is None:
            raise ValueError("No data loaded.")
        
        df = self.data.copy()
        cols = columns if columns else df.columns
        
        if method == 'forward_fill':
            df[cols] = df[cols].fillna(method='ffill')
        elif method == 'backward_fill':
            df[cols] = df[cols].fillna(method='bfill')
        elif method == 'interpolate':
            df[cols] = df[cols].interpolate(method='linear')
        elif method == 'drop':
            df = df.dropna(subset=cols)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"✓ Handled missing values using {method}")
        self.data = df
        return df
    
    def remove_outliers(self,
                       target_col: str,
                       method: str = 'iqr',
                       threshold: float = 3.0) -> pd.DataFrame:
        """
        Remove outliers from target column
        
        Args:
            target_col: Name of target column
            method: Method to detect outliers ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
        
        Returns:
            DataFrame with outliers removed
        """
        if self.data is None:
            raise ValueError("No data loaded.")
        
        df = self.data.copy()
        original_len = len(df)
        
        if method == 'iqr':
            Q1 = df[target_col].quantile(0.25)
            Q3 = df[target_col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            df = df[(df[target_col] >= lower) & (df[target_col] <= upper)]
        
        elif method == 'zscore':
            z_scores = np.abs((df[target_col] - df[target_col].mean()) / df[target_col].std())
            df = df[z_scores < threshold]
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        removed = original_len - len(df)
        print(f"✓ Removed {removed} outliers ({removed/original_len*100:.2f}%)")
        
        self.data = df
        return df
    
    def resample_data(self,
                     date_col: str,
                     target_col: str,
                     freq: str = 'D',
                     agg_func: str = 'mean') -> pd.DataFrame:
        """
        Resample time series data
        
        Args:
            date_col: Name of date column
            target_col: Name of target column
            freq: Resampling frequency ('D', 'W', 'M', 'H', etc.)
            agg_func: Aggregation function ('mean', 'sum', 'min', 'max')
        
        Returns:
            Resampled DataFrame
        """
        if self.data is None:
            raise ValueError("No data loaded.")
        
        df = self.data.copy()
        df = df.set_index(date_col)
        
        # Resample
        if agg_func == 'mean':
            resampled = df[[target_col]].resample(freq).mean()
        elif agg_func == 'sum':
            resampled = df[[target_col]].resample(freq).sum()
        elif agg_func == 'min':
            resampled = df[[target_col]].resample(freq).min()
        elif agg_func == 'max':
            resampled = df[[target_col]].resample(freq).max()
        else:
            raise ValueError(f"Unknown aggregation function: {agg_func}")
        
        resampled = resampled.reset_index()
        
        print(f"✓ Resampled from {len(df)} to {len(resampled)} rows ({freq}, {agg_func})")
        self.data = resampled
        return resampled
    
    def train_test_split(self,
                        date_col: str,
                        test_size: float = 0.2,
                        validation_size: Optional[float] = None) -> Union[Tuple, Tuple]:
        """
        Split data into train/test (or train/val/test)
        
        Args:
            date_col: Name of date column
            test_size: Proportion of data for testing
            validation_size: Optional proportion for validation
        
        Returns:
            Tuple of DataFrames (train, test) or (train, val, test)
        """
        if self.data is None:
            raise ValueError("No data loaded.")
        
        df = self.data.sort_values(date_col).reset_index(drop=True)
        n = len(df)
        
        if validation_size:
            # Three-way split
            test_idx = int(n * (1 - test_size))
            val_idx = int(n * (1 - test_size - validation_size))
            
            train = df[:val_idx]
            val = df[val_idx:test_idx]
            test = df[test_idx:]
            
            print(f"✓ Split: Train={len(train)}, Val={len(val)}, Test={len(test)}")
            return train, val, test
        else:
            # Two-way split
            split_idx = int(n * (1 - test_size))
            
            train = df[:split_idx]
            test = df[split_idx:]
            
            print(f"✓ Split: Train={len(train)}, Test={len(test)}")
            return train, test
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of loaded data
        
        Returns:
            Dictionary with data summary
        """
        if self.data is None:
            return {"error": "No data loaded"}
        
        summary = {
            'metadata': self.metadata,
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isna().sum().to_dict(),
            'numeric_summary': self.data.describe().to_dict()
        }
        
        return summary


def generate_sample_data(n_samples: int = 365,
                        start_date: str = '2023-01-01',
                        freq: str = 'D',
                        trend: str = 'linear',
                        seasonality: bool = True,
                        noise_level: float = 0.1) -> pd.DataFrame:
    """
    Generate sample time series data for testing
    
    Args:
        n_samples: Number of samples
        start_date: Start date
        freq: Frequency ('D', 'H', 'W', etc.)
        trend: Trend type ('linear', 'exponential', 'none')
        seasonality: Whether to add seasonality
        noise_level: Standard deviation of noise
    
    Returns:
        DataFrame with generated data
    """
    dates = pd.date_range(start=start_date, periods=n_samples, freq=freq)
    
    # Generate trend
    if trend == 'linear':
        trend_component = np.linspace(100, 200, n_samples)
    elif trend == 'exponential':
        trend_component = 100 * np.exp(np.linspace(0, 1, n_samples))
    else:
        trend_component = np.ones(n_samples) * 150
    
    # Generate seasonality
    if seasonality:
        if freq == 'D':
            # Weekly seasonality
            seasonal_component = 30 * np.sin(2 * np.pi * np.arange(n_samples) / 7)
        elif freq == 'H':
            # Daily seasonality
            seasonal_component = 30 * np.sin(2 * np.pi * np.arange(n_samples) / 24)
        else:
            seasonal_component = 30 * np.sin(2 * np.pi * np.arange(n_samples) / 30)
    else:
        seasonal_component = np.zeros(n_samples)
    
    # Add noise
    noise = np.random.normal(0, noise_level * trend_component.mean(), n_samples)
    
    # Combine components
    values = trend_component + seasonal_component + noise
    values = np.maximum(values, 0)  # Ensure non-negative
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'value': values,
        'category': np.random.choice(['A', 'B', 'C'], n_samples)
    })
    
    return df


if __name__ == "__main__":
    print("Data Loader Module - Demo\n")
    print("="*60)
    
    # Generate sample data
    print("\n1. Generating Sample Data...")
    sample_data = generate_sample_data(n_samples=365, seasonality=True)
    print(f"✓ Generated {len(sample_data)} samples")
    print(sample_data.head())
    
    # Save to CSV
    sample_data.to_csv('sample_data.csv', index=False)
    print("\n✓ Saved to sample_data.csv")
    
    # Initialize loader
    print("\n2. Loading Data...")
    loader = TimeSeriesDataLoader()
    data = loader.load_csv('sample_data.csv', date_col='date')
    
    # Validate
    print("\n3. Validating Time Series...")
    validation = loader.validate_time_series('date', 'value')
    print(f"Valid: {validation['is_valid']}")
    print(f"Date range: {validation['date_range']['start']} to {validation['date_range']['end']}")
    print(f"Target stats: {validation['target_stats']}")
    
    if validation['issues']:
        print(f"Issues: {validation['issues']}")
    
    # Split data
    print("\n4. Splitting Data...")
    train, test = loader.train_test_split('date', test_size=0.2)
    
    # Get summary
    print("\n5. Data Summary:")
    summary = loader.get_summary()
    print(f"Shape: {summary['shape']}")
    print(f"Columns: {summary['columns']}")
    
    print("\n" + "="*60)
    print("Demo complete!")