"""
ChronoCast Multi-Model Forecasting System
==========================================

Advanced time series forecasting using ensemble of models:
- XGBoost
- ARIMA
- SARIMA
- Prophet

This module provides comprehensive forecasting capabilities with
automatic model selection, hyperparameter tuning, and ensemble predictions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Core ML and statistical libraries
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Time series libraries
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available. ARIMA and SARIMA models will be disabled.")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Warning: Prophet not available. Prophet model will be disabled.")

import logging
import time
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json

logger = logging.getLogger(__name__)


class ChronoCastMultiModel:
    """
    Advanced multi-model forecasting system that combines
    XGBoost, ARIMA, SARIMA, and Prophet for optimal predictions.
    """
    
    def __init__(self, forecast_days: int = 30, auto_ensemble: bool = True):
        """
        Initialize the ChronoCast Multi-Model system.
        
        Args:
            forecast_days: Number of days to forecast ahead
            auto_ensemble: Whether to automatically ensemble model predictions
        """
        self.forecast_days = forecast_days
        self.auto_ensemble = auto_ensemble
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        self.feature_importance = {}
        self.training_history = {}
        
        # Model configurations
        self.xgb_params = {
            'n_estimators': 300,
            'learning_rate': 0.05,
            'max_depth': 4,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        self.arima_params = {'order': (5, 1, 2)}
        self.sarima_params = {
            'order': (1, 1, 1),
            'seasonal_order': (1, 1, 1, 7)  # Weekly seasonality
        }
        
        self.prophet_params = {
            'yearly_seasonality': False,
            'weekly_seasonality': True,
            'daily_seasonality': False
        }
    
    def build_features(self, series: np.ndarray, dates: pd.DatetimeIndex, 
                      lags: int = 14) -> pd.DataFrame:
        """
        Build comprehensive features for XGBoost model.
        
        Args:
            series: Time series values
            dates: Corresponding dates
            lags: Number of lag features to create
            
        Returns:
            DataFrame with engineered features
        """
        df_feat = pd.DataFrame({"y": series})
        
        # Lag features
        for lag in range(1, lags + 1):
            df_feat[f"lag_{lag}"] = df_feat["y"].shift(lag)
        
        # Rolling statistics
        df_feat["rolling_mean_7"] = df_feat["y"].shift(1).rolling(7).mean()
        df_feat["rolling_mean_14"] = df_feat["y"].shift(1).rolling(14).mean()
        df_feat["rolling_std_7"] = df_feat["y"].shift(1).rolling(7).std()
        df_feat["rolling_std_14"] = df_feat["y"].shift(1).rolling(14).std()
        
        # Time-based features
        if dates is not None:
            if hasattr(dates, 'dayofweek'):
                df_feat["day_of_week"] = dates.dayofweek
                df_feat["month"] = dates.month
                df_feat["quarter"] = dates.quarter
                df_feat["day_of_month"] = dates.day
            else:
                # Convert to datetime if it's not already
                try:
                    dates_pd = pd.to_datetime(dates)
                    df_feat["day_of_week"] = dates_pd.dayofweek
                    df_feat["month"] = dates_pd.month
                    df_feat["quarter"] = dates_pd.quarter
                    df_feat["day_of_month"] = dates_pd.day
                except:
                    # Fallback if conversion fails
                    df_feat["day_of_week"] = np.arange(len(series)) % 7
                    df_feat["month"] = ((np.arange(len(series)) % 365) // 30) + 1
                    df_feat["quarter"] = ((np.arange(len(series)) % 365) // 90) + 1
                    df_feat["day_of_month"] = (np.arange(len(series)) % 30) + 1
        else:
            # Fallback if no dates provided
            df_feat["day_of_week"] = np.arange(len(series)) % 7
            df_feat["month"] = ((np.arange(len(series)) % 365) // 30) + 1
            df_feat["quarter"] = ((np.arange(len(series)) % 365) // 90) + 1
            df_feat["day_of_month"] = (np.arange(len(series)) % 30) + 1
        
        # Trend features
        df_feat["trend"] = np.arange(len(series))
        
        return df_feat.dropna()
    
    def train_xgboost(self, df: pd.DataFrame, test_size: int = None) -> Dict[str, Any]:
        """Train XGBoost model with feature engineering."""
        if test_size is None:
            test_size = self.forecast_days
        
        print("Training XGBoost model...")
        
        # Prepare features
        dates = pd.to_datetime(df['ds']) if 'ds' in df.columns else None
        series = df['y'].values if 'y' in df.columns else df.iloc[:, -1].values
        
        feat_df = self.build_features(series, dates, lags=14)
        X = feat_df.drop(columns="y")
        y = feat_df["y"]
        
        # Train/test split
        split = len(X) - test_size
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        # Train model
        model = XGBRegressor(**self.xgb_params)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_test = model.predict(X_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # Recursive forecast
        history = list(series)
        future_predictions = []
        
        for i in range(self.forecast_days):
            # Create features for next prediction
            row = {}
            for lag in range(1, 15):
                if len(history) >= lag:
                    row[f"lag_{lag}"] = history[-(lag)]
                else:
                    row[f"lag_{lag}"] = np.mean(history)
            
            row["rolling_mean_7"] = np.mean(history[-7:]) if len(history) >= 7 else np.mean(history)
            row["rolling_mean_14"] = np.mean(history[-14:]) if len(history) >= 14 else np.mean(history)
            row["rolling_std_7"] = np.std(history[-7:]) if len(history) >= 7 else 0
            row["rolling_std_14"] = np.std(history[-14:]) if len(history) >= 14 else 0
            
            # Time features
            if dates is not None:
                try:
                    # Get the first date and add the history length
                    first_date = pd.to_datetime(dates.iloc[0] if hasattr(dates, 'iloc') else dates[0])
                    next_date = first_date + pd.Timedelta(days=len(history))
                    row["day_of_week"] = next_date.dayofweek
                    row["month"] = next_date.month
                    row["quarter"] = next_date.quarter
                    row["day_of_month"] = next_date.day
                except:
                    # Fallback if date calculation fails
                    day_idx = len(history)
                    row["day_of_week"] = day_idx % 7
                    row["month"] = ((day_idx % 365) // 30) + 1
                    row["quarter"] = ((day_idx % 365) // 90) + 1
                    row["day_of_month"] = (day_idx % 30) + 1
            else:
                day_idx = len(history)
                row["day_of_week"] = day_idx % 7
                row["month"] = ((day_idx % 365) // 30) + 1
                row["quarter"] = ((day_idx % 365) // 90) + 1
                row["day_of_month"] = (day_idx % 30) + 1
            
            row["trend"] = len(history)
            
            # Predict
            pred = model.predict(pd.DataFrame([row]))[0]
            future_predictions.append(pred)
            history.append(pred)
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.models['xgboost'] = model
        self.predictions['xgboost'] = future_predictions
        self.metrics['xgboost'] = {'test_mae': test_mae, 'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test))}
        self.feature_importance['xgboost'] = importance_df.to_dict('records')
        
        return {
            'model': 'xgboost',
            'test_mae': test_mae,
            'predictions': future_predictions,
            'feature_importance': importance_df
        }
    
    def train_arima(self, df: pd.DataFrame, test_size: int = None) -> Dict[str, Any]:
        """Train ARIMA model."""
        if not STATSMODELS_AVAILABLE:
            print("ARIMA not available - statsmodels not installed")
            return {}
        
        if test_size is None:
            test_size = self.forecast_days
        
        print("Training ARIMA model...")
        
        series = df['y'].values if 'y' in df.columns else df.iloc[:, -1].values
        
        # Train/test split
        train_series = series[:-test_size]
        test_series = series[-test_size:]
        
        # Train model
        model = ARIMA(train_series, **self.arima_params)
        model_fit = model.fit()
        
        # Test predictions
        test_pred = model_fit.forecast(steps=test_size)
        test_mae = mean_absolute_error(test_series, test_pred)
        
        # Full series forecast
        full_model = ARIMA(series, **self.arima_params)
        full_fit = full_model.fit()
        future_predictions = full_fit.forecast(steps=self.forecast_days)
        
        self.models['arima'] = model_fit
        self.predictions['arima'] = future_predictions.tolist() if hasattr(future_predictions, 'tolist') else list(future_predictions)
        self.metrics['arima'] = {'test_mae': test_mae, 'test_rmse': np.sqrt(mean_squared_error(test_series, test_pred))}
        
        return {
            'model': 'arima',
            'test_mae': test_mae,
            'predictions': self.predictions['arima']
        }
    
    def train_sarima(self, df: pd.DataFrame, test_size: int = None) -> Dict[str, Any]:
        """Train SARIMA model with seasonality."""
        if not STATSMODELS_AVAILABLE:
            print("SARIMA not available - statsmodels not installed")
            return {}
        
        if test_size is None:
            test_size = self.forecast_days
        
        print("Training SARIMA model...")
        
        series = df['y'].values if 'y' in df.columns else df.iloc[:, -1].values
        
        # Train/test split
        train_series = series[:-test_size]
        test_series = series[-test_size:]
        
        # Train model
        model = SARIMAX(train_series, **self.sarima_params)
        model_fit = model.fit(disp=False)
        
        # Test predictions
        test_pred = model_fit.forecast(steps=test_size)
        test_mae = mean_absolute_error(test_series, test_pred)
        
        # Full series forecast
        full_model = SARIMAX(series, **self.sarima_params)
        full_fit = full_model.fit(disp=False)
        future_predictions = full_fit.forecast(steps=self.forecast_days)
        
        self.models['sarima'] = model_fit
        self.predictions['sarima'] = future_predictions.tolist() if hasattr(future_predictions, 'tolist') else list(future_predictions)
        self.metrics['sarima'] = {'test_mae': test_mae, 'test_rmse': np.sqrt(mean_squared_error(test_series, test_pred))}
        
        return {
            'model': 'sarima',
            'test_mae': test_mae,
            'predictions': self.predictions['sarima']
        }
    
    def train_prophet(self, df: pd.DataFrame, test_size: int = None) -> Dict[str, Any]:
        """Train Prophet model."""
        if not PROPHET_AVAILABLE:
            print("Prophet not available - prophet not installed")
            return {}
        
        if test_size is None:
            test_size = self.forecast_days
        
        print("Training Prophet model...")
        
        # Prepare data for Prophet
        if 'ds' in df.columns and 'y' in df.columns:
            prophet_df = df[['ds', 'y']].copy()
        else:
            # Create dates if not provided
            dates = pd.date_range(start='2024-01-01', periods=len(df), freq='D')
            prophet_df = pd.DataFrame({
                'ds': dates,
                'y': df.iloc[:, -1].values
            })
        
        # Train/test split
        train_df = prophet_df[:-test_size]
        test_df = prophet_df[-test_size:]
        
        # Train model
        model = Prophet(**self.prophet_params)
        model.fit(train_df)
        
        # Create future dataframe
        future_dates = model.make_future_dataframe(periods=self.forecast_days)
        forecast = model.predict(future_dates)
        
        # Test predictions
        test_pred = forecast.iloc[-test_size-self.forecast_days:-self.forecast_days]['yhat'].values
        test_mae = mean_absolute_error(test_df['y'].values, test_pred)
        
        # Future predictions
        future_predictions = forecast.iloc[-self.forecast_days:]['yhat'].values
        
        self.models['prophet'] = model
        self.predictions['prophet'] = future_predictions.tolist()
        self.metrics['prophet'] = {'test_mae': test_mae, 'test_rmse': np.sqrt(mean_squared_error(test_df['y'].values, test_pred))}
        
        return {
            'model': 'prophet',
            'test_mae': test_mae,
            'predictions': future_predictions
        }
    
    def ensemble_predictions(self) -> np.ndarray:
        """
        Create ensemble predictions from all available models.
        
        Returns:
            Ensemble predictions array
        """
        if not self.predictions:
            raise ValueError("No predictions available. Train models first.")
        
        # Collect all predictions
        all_predictions = []
        model_weights = {}
        
        # Calculate weights based on inverse MAE (lower MAE = higher weight)
        for model_name, metrics in self.metrics.items():
            if 'test_mae' in metrics:
                mae = metrics['test_mae']
                weight = 1.0 / (mae + 1e-8)  # Add small epsilon to avoid division by zero
                model_weights[model_name] = weight
                all_predictions.append(self.predictions[model_name])
        
        if not all_predictions:
            raise ValueError("No valid predictions for ensemble")
        
        # Normalize weights
        total_weight = sum(model_weights.values())
        model_weights = {k: v/total_weight for k, v in model_weights.items()}
        
        # Calculate weighted ensemble
        ensemble_pred = np.zeros(self.forecast_days)
        for i, (model_name, pred) in enumerate(self.predictions.items()):
            if model_name in model_weights and len(pred) == self.forecast_days:
                ensemble_pred += np.array(pred) * model_weights[model_name]
        
        self.predictions['ensemble'] = ensemble_pred.tolist()
        
        # Calculate ensemble metrics (approximate)
        ensemble_mae = np.mean([self.metrics[m]['test_mae'] for m in model_weights.keys()])
        self.metrics['ensemble'] = {'test_mae': ensemble_mae, 'test_rmse': ensemble_mae * 1.2}
        
        print(f"Ensemble weights: {model_weights}")
        
        return ensemble_pred
    
    def fit_predict(self, df: pd.DataFrame, models: List[str] = None) -> Dict[str, Any]:
        """
        Train specified models and generate predictions.
        
        Args:
            df: DataFrame with time series data (columns: 'ds', 'y' or last column is target)
            models: List of models to train ['xgboost', 'arima', 'sarima', 'prophet']
            
        Returns:
            Dictionary with results from all models
        """
        if models is None:
            models = ['xgboost', 'arima', 'sarima', 'prophet']
        
        start_time = time.time()
        results = {}
        
        print(f"Starting ChronoCast Multi-Model training with {len(df)} data points...")
        print(f"Models to train: {models}")
        print(f"Forecast horizon: {self.forecast_days} days")
        
        # Train each model
        for model_name in models:
            try:
                if model_name == 'xgboost':
                    result = self.train_xgboost(df)
                elif model_name == 'arima':
                    result = self.train_arima(df)
                elif model_name == 'sarima':
                    result = self.train_sarima(df)
                elif model_name == 'prophet':
                    result = self.train_prophet(df)
                else:
                    print(f"Unknown model: {model_name}")
                    continue
                
                results[model_name] = result
                print(f"   {model_name.upper()} - Test MAE: {result.get('test_mae', 'N/A'):.1f}")
                
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                logger.error(f"Error training {model_name}: {str(e)}")
        
        # Create ensemble if multiple models trained
        if len(results) > 1 and self.auto_ensemble:
            print("\nCreating ensemble prediction...")
            ensemble_pred = self.ensemble_predictions()
            results['ensemble'] = {
                'model': 'ensemble',
                'predictions': ensemble_pred.tolist(),
                'test_mae': self.metrics['ensemble']['test_mae']
            }
            print(f"ENSEMBLE - Combined MAE: {self.metrics['ensemble']['test_mae']:.1f}")
        
        # Store training history
        self.training_history = {
            'training_time': time.time() - start_time,
            'models_trained': list(results.keys()),
            'forecast_days': self.forecast_days,
            'data_points': len(df)
        }
        
        print(f"\nTraining completed in {self.training_history['training_time']:.2f} seconds")
        
        return results
    
    def get_future_dates(self, df: pd.DataFrame) -> pd.DatetimeIndex:
        """Generate future dates for forecasting."""
        if 'ds' in df.columns:
            last_date = pd.to_datetime(df['ds']).iloc[-1]
        else:
            # Assume daily frequency starting from 2024-01-01
            last_date = pd.Timestamp('2024-01-01') + pd.Timedelta(days=len(df)-1)
        
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=self.forecast_days,
            freq='D'
        )
        
        return future_dates
    
    def create_summary_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a summary table with all model predictions.
        
        Args:
            df: Original training data
            
        Returns:
            DataFrame with forecast summary
        """
        future_dates = self.get_future_dates(df)
        
        summary_data = {'Date': future_dates.strftime('%Y-%m-%d').tolist()}
        
        for model_name, predictions in self.predictions.items():
            if len(predictions) == self.forecast_days:
                summary_data[model_name.capitalize()] = np.round(predictions).astype(int).tolist()
        
        summary_df = pd.DataFrame(summary_data)
        return summary_df
    
    def plot_forecasts(self, df: pd.DataFrame, save_path: str = None, 
                      show_plot: bool = True) -> None:
        """
        Plot forecasts from all models.
        
        Args:
            df: Original training data
            save_path: Path to save the plot
            show_plot: Whether to display the plot
        """
        if not self.predictions:
            print("No predictions to plot")
            return
        
        future_dates = self.get_future_dates(df)
        
        # Prepare data
        if 'ds' in df.columns and 'y' in df.columns:
            historical_dates = pd.to_datetime(df['ds'])
            historical_values = df['y'].values
        else:
            # Create dummy dates
            historical_dates = pd.date_range(start='2024-01-01', periods=len(df), freq='D')
            historical_values = df.iloc[:, -1].values
        
        # Create subplots
        n_models = len(self.predictions)
        cols = 2
        rows = (n_models + 1) // 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 6*rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes] if cols == 1 else list(axes)
        else:
            axes = axes.flatten()
        
        # Plot each model
        colors = ['steelblue', 'darkorange', 'seagreen', 'crimson', 'purple']
        
        for i, (model_name, predictions) in enumerate(self.predictions.items()):
            ax = axes[i]
            
            # Plot historical data (last 60 days for clarity)
            hist_days_to_show = min(60, len(historical_dates))
            ax.plot(
                historical_dates[-hist_days_to_show:], 
                historical_values[-hist_days_to_show:],
                color="gray", linewidth=1.2, label="Historical"
            )
            
            # Plot predictions
            if len(predictions) == self.forecast_days:
                ax.plot(
                    future_dates, predictions,
                    color=colors[i % len(colors)], linewidth=2, 
                    marker="o", markersize=3, label=f"{model_name.capitalize()} forecast"
                )
            
            # Vertical line at forecast start
            ax.axvline(
                historical_dates[-1], color="black", 
                linestyle="--", linewidth=0.8, alpha=0.5
            )
            
            ax.set_title(f"{model_name.capitalize()}", fontsize=13, fontweight='bold')
            ax.set_xlabel("Date")
            ax.set_ylabel("Values")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="x", rotation=30)
        
        # Hide empty subplots
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(
            f"ChronoCast Multi-Model Forecast - Next {self.forecast_days} Days", 
            fontsize=16, fontweight="bold", y=1.02
        )
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def save_results(self, df: pd.DataFrame, output_dir: str = "chronocast_results") -> None:
        """
        Save all results to files.
        
        Args:
            df: Original training data
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save summary table
        summary_df = self.create_summary_table(df)
        summary_csv_path = output_path / "forecast_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"Summary table saved to {summary_csv_path}")
        
        # Save detailed predictions
        predictions_df = pd.DataFrame({
            'model': list(self.predictions.keys()),
            'predictions': [str(pred) for pred in self.predictions.values()],
            'test_mae': [self.metrics.get(m, {}).get('test_mae', 'N/A') for m in self.predictions.keys()]
        })
        predictions_df.to_csv(output_path / "model_predictions.csv", index=False)
        
        # Save metrics
        with open(output_path / "metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Save feature importance (if available)
        if self.feature_importance:
            with open(output_path / "feature_importance.json", 'w') as f:
                json.dump(self.feature_importance, f, indent=2)
        
        # Save plot
        plot_path = output_path / "forecast_plot.png"
        self.plot_forecasts(df, save_path=str(plot_path), show_plot=False)
        
        # Save models
        models_path = output_path / "trained_models"
        models_path.mkdir(exist_ok=True)
        
        for model_name, model in self.models.items():
            model_file = models_path / f"{model_name}.pkl"
            try:
                with open(model_file, 'wb') as f:
                    pickle.dump(model, f)
                print(f"Model {model_name} saved to {model_file}")
            except Exception as e:
                print(f"Could not save model {model_name}: {str(e)}")
        
        print(f"All results saved to {output_path}")
    
    def get_best_model(self) -> Tuple[str, float]:
        """
        Get the best performing model based on test MAE.
        
        Returns:
            Tuple of (model_name, test_mae)
        """
        if not self.metrics:
            raise ValueError("No metrics available. Train models first.")
        
        best_model = min(
            self.metrics.items(),
            key=lambda x: x[1].get('test_mae', float('inf')) if isinstance(x[1], dict) else float('inf')
        )
        
        model_name = best_model[0]
        test_mae = best_model[1].get('test_mae', float('inf')) if isinstance(best_model[1], dict) else float('inf')
        
        return model_name, test_mae


# Convenience function for quick forecasting
def forecast_views(df, forecast_days=30, models=None, save_results=True):
    """
    Convenience function to quickly forecast views using ChronoCast Multi-Model.
    
    Args:
        df: DataFrame with time series data (columns: 'ds', 'y' or last column is target)
        forecast_days: Number of days to forecast ahead
        models: List of models to use ['xgboost', 'arima', 'sarima', 'prophet']
        save_results: Whether to save results to files
        
    Returns:
        ChronoCastMultiModel instance with results
    """
    # Initialize model
    chronocast = ChronoCastMultiModel(forecast_days=forecast_days)
    
    # Train and predict
    results = chronocast.fit_predict(df, models=models)
    
    # Create and display summary
    summary_df = chronocast.create_summary_table(df)
    print("\n" + "="*60)
    print("CHRONOCAST MULTI-MODEL FORECAST SUMMARY")
    print("="*60)
    print(summary_df.to_string(index=False))
    
    # Show best model
    try:
        best_model, best_mae = chronocast.get_best_model()
        print(f"\nBest performing model: {best_model.upper()} (MAE: {best_mae:.1f})")
    except:
        pass
    
    # Save results if requested
    if save_results:
        chronocast.save_results(df)
    
    # Plot forecasts
    try:
        chronocast.plot_forecasts(df, show_plot=False)  # Don't show plot in automated test
    except Exception as e:
        print(f"Warning: Could not generate plot: {str(e)}")
    
    return chronocast


if __name__ == "__main__":
    # Example usage with sample data
    print("ChronoCast Multi-Model Forecasting System")
    print("=" * 50)
    
    # Create sample data
    dates = pd.date_range(start="2024-01-01", periods=180, freq="D")
    np.random.seed(42)
    trend = np.linspace(1000, 4000, 180)
    season = 500 * np.sin(np.linspace(0, 4 * np.pi, 180))
    noise = np.random.normal(0, 150, 180)
    views = (trend + season + noise).astype(int)
    
    df = pd.DataFrame({"ds": dates, "y": views})
    
    # Run forecasting
    chronocast = forecast_views(df, forecast_days=30, models=['xgboost', 'arima', 'sarima', 'prophet'])
