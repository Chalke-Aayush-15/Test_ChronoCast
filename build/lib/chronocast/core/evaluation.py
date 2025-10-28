"""
Evaluation Module for ChronoCast
Provides comprehensive metrics and comparison tools
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
import json
from datetime import datetime


class MetricCalculator:
    """Calculate various forecasting metrics"""
    
    @staticmethod
    def mse(y_true, y_pred) -> float:
        """Mean Squared Error"""
        return mean_squared_error(y_true, y_pred)
    
    @staticmethod
    def rmse(y_true, y_pred) -> float:
        """Root Mean Squared Error"""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mae(y_true, y_pred) -> float:
        """Mean Absolute Error"""
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def mape(y_true, y_pred) -> float:
        """Mean Absolute Percentage Error"""
        return mean_absolute_percentage_error(y_true, y_pred) * 100
    
    @staticmethod
    def r2(y_true, y_pred) -> float:
        """R-squared (Coefficient of Determination)"""
        return r2_score(y_true, y_pred)
    
    @staticmethod
    def smape(y_true, y_pred) -> float:
        """Symmetric Mean Absolute Percentage Error"""
        numerator = np.abs(y_pred - y_true)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        return np.mean(numerator / denominator) * 100
    
    @staticmethod
    def mase(y_true, y_pred, y_train) -> float:
        """Mean Absolute Scaled Error"""
        mae_forecast = np.mean(np.abs(y_true - y_pred))
        mae_naive = np.mean(np.abs(np.diff(y_train)))
        return mae_forecast / mae_naive if mae_naive != 0 else np.inf
    
    @staticmethod
    def bias(y_true, y_pred) -> float:
        """Forecast Bias (mean error)"""
        return np.mean(y_pred - y_true)
    
    @staticmethod
    def percentage_bias(y_true, y_pred) -> float:
        """Percentage Bias"""
        return (np.sum(y_pred - y_true) / np.sum(y_true)) * 100


def evaluate_model(y_true, 
                   y_pred, 
                   y_train: Optional[np.ndarray] = None,
                   metrics: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Evaluate model predictions with multiple metrics
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        y_train: Training data (needed for MASE)
        metrics: List of metrics to calculate (default: all)
    
    Returns:
        Dictionary of metric names and values
    """
    calc = MetricCalculator()
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Default metrics
    if metrics is None:
        metrics = ['mse', 'rmse', 'mae', 'mape', 'r2', 'smape', 'bias']
    
    results = {}
    
    for metric in metrics:
        try:
            if metric.lower() == 'mse':
                results['MSE'] = calc.mse(y_true, y_pred)
            elif metric.lower() == 'rmse':
                results['RMSE'] = calc.rmse(y_true, y_pred)
            elif metric.lower() == 'mae':
                results['MAE'] = calc.mae(y_true, y_pred)
            elif metric.lower() == 'mape':
                results['MAPE'] = calc.mape(y_true, y_pred)
            elif metric.lower() == 'r2':
                results['R²'] = calc.r2(y_true, y_pred)
            elif metric.lower() == 'smape':
                results['SMAPE'] = calc.smape(y_true, y_pred)
            elif metric.lower() == 'mase' and y_train is not None:
                results['MASE'] = calc.mase(y_true, y_pred, y_train)
            elif metric.lower() == 'bias':
                results['Bias'] = calc.bias(y_true, y_pred)
            elif metric.lower() == 'percentage_bias':
                results['Percentage Bias'] = calc.percentage_bias(y_true, y_pred)
        except Exception as e:
            print(f"Warning: Could not calculate {metric}: {str(e)}")
    
    return results


def compare_models(models_dict: Dict[str, Dict], 
                   metric: str = 'RMSE',
                   sort_ascending: bool = True) -> pd.DataFrame:
    """
    Compare multiple models based on their metrics
    
    Args:
        models_dict: Dictionary of {model_name: {'y_true': ..., 'y_pred': ..., 'y_train': ...}}
        metric: Primary metric to sort by
        sort_ascending: Whether to sort in ascending order
    
    Returns:
        DataFrame with comparison results
    """
    comparison_data = []
    
    for model_name, data in models_dict.items():
        metrics = evaluate_model(
            data['y_true'], 
            data['y_pred'],
            data.get('y_train')
        )
        metrics['Model'] = model_name
        comparison_data.append(metrics)
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Reorder columns to have Model first
    cols = ['Model'] + [col for col in df.columns if col != 'Model']
    df = df[cols]
    
    # Sort by specified metric
    if metric in df.columns:
        df = df.sort_values(metric, ascending=sort_ascending)
    
    return df


def plot_forecast_comparison(y_true, 
                             y_pred, 
                             dates: Optional[np.ndarray] = None,
                             model_name: str = 'Model',
                             figsize: tuple = (12, 6),
                             save_path: Optional[str] = None):
    """
    Plot actual vs predicted values
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        dates: Optional date array for x-axis
        model_name: Name of the model
        figsize: Figure size
        save_path: Path to save the plot
    """
    plt.figure(figsize=figsize)
    
    if dates is None:
        x = np.arange(len(y_true))
        xlabel = 'Time Steps'
    else:
        x = dates
        xlabel = 'Date'
    
    plt.plot(x, y_true, label='Actual', marker='o', markersize=4, linewidth=2)
    plt.plot(x, y_pred, label='Predicted', marker='x', markersize=4, linewidth=2, alpha=0.8)
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title(f'Forecast Comparison: {model_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if dates is not None:
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_residuals(y_true, 
                   y_pred,
                   model_name: str = 'Model',
                   figsize: tuple = (15, 5),
                   save_path: Optional[str] = None):
    """
    Plot residual analysis
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        model_name: Name of the model
        figsize: Figure size
        save_path: Path to save the plot
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Residual plot
    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Values', fontsize=10)
    axes[0].set_ylabel('Residuals', fontsize=10)
    axes[0].set_title('Residual Plot', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Residual distribution
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residuals', fontsize=10)
    axes[1].set_ylabel('Frequency', fontsize=10)
    axes[1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f'Residual Analysis: {model_name}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_metrics_comparison(comparison_df: pd.DataFrame,
                            metrics: Optional[List[str]] = None,
                            figsize: tuple = (12, 6),
                            save_path: Optional[str] = None):
    """
    Plot bar chart comparing model metrics
    
    Args:
        comparison_df: DataFrame from compare_models()
        metrics: List of metrics to plot (default: all numeric columns)
        figsize: Figure size
        save_path: Path to save the plot
    """
    if metrics is None:
        # Get all numeric columns except 'Model'
        metrics = [col for col in comparison_df.columns 
                  if col != 'Model' and pd.api.types.is_numeric_dtype(comparison_df[col])]
    
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        if metric in comparison_df.columns:
            ax = axes[idx]
            comparison_df.plot(x='Model', y=metric, kind='bar', ax=ax, legend=False)
            ax.set_title(metric, fontsize=11, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel(metric, fontsize=9)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
    
    # Hide extra subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Model Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def save_evaluation_report(metrics: Dict[str, float],
                           model_name: str,
                           output_path: str = 'evaluation_report.json'):
    """
    Save evaluation metrics to JSON file
    
    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model
        output_path: Path to save the report
    """
    report = {
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"Evaluation report saved to {output_path}")


def load_evaluation_report(filepath: str) -> Dict:
    """
    Load evaluation report from JSON file
    
    Args:
        filepath: Path to the report file
    
    Returns:
        Dictionary with report data
    """
    with open(filepath, 'r') as f:
        report = json.load(f)
    
    return report


if __name__ == "__main__":
    # Example usage
    print("Evaluation Module - Demo\n")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    
    y_true = np.random.randn(n_samples) * 10 + 100
    y_pred_good = y_true + np.random.randn(n_samples) * 2
    y_pred_bad = y_true + np.random.randn(n_samples) * 10
    y_train = np.random.randn(200) * 10 + 100
    
    # Evaluate single model
    print("1. Single Model Evaluation")
    print("-" * 50)
    metrics = evaluate_model(y_true, y_pred_good, y_train)
    for metric, value in metrics.items():
        print(f"{metric:20s}: {value:10.4f}")
    
    # Compare models
    print("\n2. Model Comparison")
    print("-" * 50)
    models = {
        'Good Model': {'y_true': y_true, 'y_pred': y_pred_good, 'y_train': y_train},
        'Bad Model': {'y_true': y_true, 'y_pred': y_pred_bad, 'y_train': y_train}
    }
    
    comparison = compare_models(models)
    print(comparison.to_string(index=False))
    
    # Save report
    print("\n3. Saving Evaluation Report")
    print("-" * 50)
    save_evaluation_report(metrics, 'Good Model', 'demo_report.json')
    
    # Visualizations
    print("\n4. Creating Visualizations")
    print("-" * 50)
    
    plot_forecast_comparison(y_true, y_pred_good, model_name='Good Model')
    plot_residuals(y_true, y_pred_good, model_name='Good Model')
    plot_metrics_comparison(comparison)
    
    print("\nDemo complete!")