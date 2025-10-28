"""
ChronoCast - A transparent and modular time series forecasting library
Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "Aayush Chalke"

# Core modules
from chronocast.core.model_wrapper import ChronoModel
from chronocast.core.feature_engineering import (
    create_time_features, 
    create_lag_features, 
    create_rolling_features,
    create_all_features
)
from chronocast.core.evaluation import (
    evaluate_model, 
    compare_models,
    plot_forecast_comparison,
    plot_residuals,
    plot_metrics_comparison
)
from chronocast.core.explainability import ModelExplainer
from chronocast.core.visualization import InteractiveVisualizer

# Utilities
from chronocast.utils.logger import ChronoLogger, ExperimentTracker
from chronocast.utils.data_loader import TimeSeriesDataLoader, generate_sample_data

__all__ = [
    # Core
    'ChronoModel',
    'create_time_features',
    'create_lag_features',
    'create_rolling_features',
    'create_all_features',
    'evaluate_model',
    'compare_models',
    'plot_forecast_comparison',
    'plot_residuals',
    'plot_metrics_comparison',
    'ModelExplainer',
    'InteractiveVisualizer',
    # Utils
    'ChronoLogger',
    'ExperimentTracker',
    'TimeSeriesDataLoader',
    'generate_sample_data'
]