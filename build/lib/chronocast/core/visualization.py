"""
Visualization Utilities for ChronoCast
Interactive and dashboard-ready visualizations using Plotly
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional, List, Dict, Union
import warnings


class InteractiveVisualizer:
    """Create interactive visualizations for forecasting"""
    
    def __init__(self, theme: str = 'plotly_white'):
        """
        Initialize visualizer
        
        Args:
            theme: Plotly template theme
        """
        self.theme = theme
        self.color_palette = px.colors.qualitative.Set2
    
    def plot_forecast(self,
                     y_true,
                     y_pred,
                     dates: Optional[np.ndarray] = None,
                     train_data: Optional[tuple] = None,
                     confidence_interval: Optional[tuple] = None,
                     title: str = 'Forecast vs Actual',
                     save_html: Optional[str] = None) -> go.Figure:
        """
        Create interactive forecast plot
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            dates: Optional date array
            train_data: Optional tuple of (train_dates, train_values)
            confidence_interval: Optional tuple of (lower, upper) bounds
            title: Plot title
            save_html: Path to save HTML file
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Prepare x-axis
        if dates is None:
            x_test = np.arange(len(y_true))
        else:
            x_test = dates
        
        # Add training data if provided
        if train_data is not None:
            train_dates, train_values = train_data
            fig.add_trace(go.Scatter(
                x=train_dates,
                y=train_values,
                mode='lines',
                name='Training Data',
                line=dict(color='lightgray', width=1),
                opacity=0.6
            ))
        
        # Add confidence interval if provided
        if confidence_interval is not None:
            lower, upper = confidence_interval
            fig.add_trace(go.Scatter(
                x=x_test,
                y=upper,
                mode='lines',
                name='Upper Bound',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            fig.add_trace(go.Scatter(
                x=x_test,
                y=lower,
                mode='lines',
                name='Confidence Interval',
                line=dict(width=0),
                fillcolor='rgba(68, 68, 68, 0.15)',
                fill='tonexty',
                showlegend=True
            ))
        
        # Add actual values
        fig.add_trace(go.Scatter(
            x=x_test,
            y=y_true,
            mode='lines+markers',
            name='Actual',
            line=dict(color=self.color_palette[0], width=2),
            marker=dict(size=6)
        ))
        
        # Add predicted values
        fig.add_trace(go.Scatter(
            x=x_test,
            y=y_pred,
            mode='lines+markers',
            name='Predicted',
            line=dict(color=self.color_palette[1], width=2, dash='dash'),
            marker=dict(size=6, symbol='x')
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Date' if dates is not None else 'Time',
            yaxis_title='Value',
            hovermode='x unified',
            template=self.theme,
            height=500,
            legend=dict(x=0.01, y=0.99)
        )
        
        if save_html:
            fig.write_html(save_html)
            print(f"Interactive plot saved to {save_html}")
        
        return fig
    
    def plot_residuals(self,
                      y_true,
                      y_pred,
                      dates: Optional[np.ndarray] = None,
                      title: str = 'Residual Analysis',
                      save_html: Optional[str] = None) -> go.Figure:
        """
        Create interactive residual analysis plots
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            dates: Optional date array
            title: Plot title
            save_html: Path to save HTML file
        
        Returns:
            Plotly figure
        """
        residuals = y_true - y_pred
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Residuals Over Time', 'Residual Distribution',
                          'Residuals vs Predicted', 'Q-Q Plot'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Residuals over time
        x = dates if dates is not None else np.arange(len(residuals))
        fig.add_trace(
            go.Scatter(x=x, y=residuals, mode='markers', name='Residuals',
                      marker=dict(color=self.color_palette[2])),
            row=1, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
        
        # 2. Histogram
        fig.add_trace(
            go.Histogram(x=residuals, name='Distribution', nbinsx=30,
                        marker=dict(color=self.color_palette[3])),
            row=1, col=2
        )
        
        # 3. Residuals vs Predicted
        fig.add_trace(
            go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuals vs Pred',
                      marker=dict(color=self.color_palette[4])),
            row=2, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)
        
        # 4. Q-Q plot
        from scipy import stats
        (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
        fig.add_trace(
            go.Scatter(x=osm, y=osr, mode='markers', name='Q-Q',
                      marker=dict(color=self.color_palette[5])),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=osm, y=slope * osm + intercept, mode='lines',
                      name='Theoretical', line=dict(color='red', dash='dash')),
            row=2, col=2
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_xaxes(title_text="Residual Value", row=1, col=2)
        fig.update_xaxes(title_text="Predicted Value", row=2, col=1)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=2)
        
        fig.update_yaxes(title_text="Residuals", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_yaxes(title_text="Residuals", row=2, col=1)
        fig.update_yaxes(title_text="Ordered Values", row=2, col=2)
        
        fig.update_layout(
            title_text=title,
            showlegend=False,
            height=700,
            template=self.theme
        )
        
        if save_html:
            fig.write_html(save_html)
            print(f"Interactive plot saved to {save_html}")
        
        return fig
    
    def plot_model_comparison(self,
                             comparison_df: pd.DataFrame,
                             metrics: Optional[List[str]] = None,
                             title: str = 'Model Comparison',
                             save_html: Optional[str] = None) -> go.Figure:
        """
        Create interactive model comparison chart
        
        Args:
            comparison_df: DataFrame from compare_models()
            metrics: List of metrics to plot
            title: Plot title
            save_html: Path to save HTML file
        
        Returns:
            Plotly figure
        """
        if metrics is None:
            metrics = [col for col in comparison_df.columns 
                      if col != 'Model' and pd.api.types.is_numeric_dtype(comparison_df[col])]
        
        n_metrics = len(metrics)
        
        # Create subplots
        fig = make_subplots(
            rows=(n_metrics + 1) // 2,
            cols=2,
            subplot_titles=metrics
        )
        
        for idx, metric in enumerate(metrics):
            row = (idx // 2) + 1
            col = (idx % 2) + 1
            
            fig.add_trace(
                go.Bar(
                    x=comparison_df['Model'],
                    y=comparison_df[metric],
                    name=metric,
                    marker_color=self.color_palette[idx % len(self.color_palette)],
                    showlegend=False
                ),
                row=row, col=col
            )
            
            fig.update_xaxes(tickangle=-45, row=row, col=col)
            fig.update_yaxes(title_text=metric, row=row, col=col)
        
        fig.update_layout(
            title_text=title,
            height=400 * ((n_metrics + 1) // 2),
            template=self.theme
        )
        
        if save_html:
            fig.write_html(save_html)
            print(f"Interactive plot saved to {save_html}")
        
        return fig
    
    def plot_feature_importance(self,
                               importance_df: pd.DataFrame,
                               top_n: int = 20,
                               title: str = 'Feature Importance',
                               save_html: Optional[str] = None) -> go.Figure:
        """
        Create interactive feature importance plot
        
        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            top_n: Number of top features to show
            title: Plot title
            save_html: Path to save HTML file
        
        Returns:
            Plotly figure
        """
        top_features = importance_df.head(top_n).sort_values('importance')
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=top_features['feature'],
            x=top_features['importance'],
            orientation='h',
            marker=dict(
                color=top_features['importance'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Importance")
            ),
            text=top_features['importance'].round(4),
            textposition='auto'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Importance Score',
            yaxis_title='Feature',
            height=max(400, top_n * 25),
            template=self.theme
        )
        
        if save_html:
            fig.write_html(save_html)
            print(f"Interactive plot saved to {save_html}")
        
        return fig
    
    def plot_time_series_decomposition(self,
                                      data: pd.DataFrame,
                                      date_col: str,
                                      value_col: str,
                                      title: str = 'Time Series Components',
                                      save_html: Optional[str] = None) -> go.Figure:
        """
        Plot time series with trend and seasonality
        
        Args:
            data: DataFrame with date and value columns
            date_col: Name of date column
            value_col: Name of value column
            title: Plot title
            save_html: Path to save HTML file
        
        Returns:
            Plotly figure
        """
        from scipy.signal import savgol_filter
        
        dates = data[date_col].values
        values = data[value_col].values
        
        # Calculate trend using Savitzky-Golay filter
        window = min(51, len(values) // 4)
        if window % 2 == 0:
            window += 1
        trend = savgol_filter(values, window, 3)
        
        # Calculate seasonality (residual after trend removal)
        detrended = values - trend
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Original Time Series', 'Trend Component', 'Detrended (Seasonal + Noise)'),
            vertical_spacing=0.1
        )
        
        # Original series
        fig.add_trace(
            go.Scatter(x=dates, y=values, mode='lines', name='Original',
                      line=dict(color=self.color_palette[0])),
            row=1, col=1
        )
        
        # Trend
        fig.add_trace(
            go.Scatter(x=dates, y=trend, mode='lines', name='Trend',
                      line=dict(color=self.color_palette[1], width=3)),
            row=2, col=1
        )
        
        # Detrended
        fig.add_trace(
            go.Scatter(x=dates, y=detrended, mode='lines', name='Detrended',
                      line=dict(color=self.color_palette[2])),
            row=3, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=3, col=1)
        
        # Update layout
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Trend", row=2, col=1)
        fig.update_yaxes(title_text="Residual", row=3, col=1)
        
        fig.update_layout(
            title_text=title,
            showlegend=False,
            height=900,
            template=self.theme
        )
        
        if save_html:
            fig.write_html(save_html)
            print(f"Interactive plot saved to {save_html}")
        
        return fig
    
    def plot_prediction_intervals(self,
                                 y_true,
                                 y_pred,
                                 dates: Optional[np.ndarray] = None,
                                 std_errors: Optional[np.ndarray] = None,
                                 confidence_levels: List[float] = [0.68, 0.95],
                                 title: str = 'Predictions with Confidence Intervals',
                                 save_html: Optional[str] = None) -> go.Figure:
        """
        Plot predictions with multiple confidence intervals
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            dates: Optional date array
            std_errors: Standard errors for predictions
            confidence_levels: List of confidence levels (e.g., [0.68, 0.95])
            title: Plot title
            save_html: Path to save HTML file
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        x = dates if dates is not None else np.arange(len(y_true))
        
        # If no std_errors provided, estimate from residuals
        if std_errors is None:
            residuals = y_true - y_pred
            std_errors = np.std(residuals)
        
        # Add confidence intervals (from widest to narrowest)
        colors = ['rgba(68, 68, 68, 0.1)', 'rgba(68, 68, 68, 0.2)']
        for idx, conf_level in enumerate(sorted(confidence_levels, reverse=True)):
            z_score = {0.68: 1.0, 0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(conf_level, 1.96)
            
            if np.isscalar(std_errors):
                upper = y_pred + z_score * std_errors
                lower = y_pred - z_score * std_errors
            else:
                upper = y_pred + z_score * std_errors
                lower = y_pred - z_score * std_errors
            
            fig.add_trace(go.Scatter(
                x=x, y=upper, mode='lines',
                line=dict(width=0), showlegend=False, hoverinfo='skip'
            ))
            fig.add_trace(go.Scatter(
                x=x, y=lower, mode='lines',
                line=dict(width=0),
                fillcolor=colors[idx % len(colors)],
                fill='tonexty',
                name=f'{int(conf_level*100)}% CI',
                showlegend=True
            ))
        
        # Add actual and predicted
        fig.add_trace(go.Scatter(
            x=x, y=y_true, mode='markers', name='Actual',
            marker=dict(size=8, color=self.color_palette[0])
        ))
        
        fig.add_trace(go.Scatter(
            x=x, y=y_pred, mode='lines', name='Predicted',
            line=dict(color=self.color_palette[1], width=3)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date' if dates is not None else 'Time',
            yaxis_title='Value',
            hovermode='x unified',
            template=self.theme,
            height=500
        )
        
        if save_html:
            fig.write_html(save_html)
            print(f"Interactive plot saved to {save_html}")
        
        return fig
    
    def create_dashboard(self,
                        y_true,
                        y_pred,
                        dates: Optional[np.ndarray] = None,
                        metrics: Optional[Dict] = None,
                        importance_df: Optional[pd.DataFrame] = None,
                        title: str = 'Forecasting Dashboard',
                        save_html: Optional[str] = None) -> go.Figure:
        """
        Create comprehensive dashboard with all key visualizations
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            dates: Optional date array
            metrics: Dictionary of evaluation metrics
            importance_df: Feature importance DataFrame
            title: Dashboard title
            save_html: Path to save HTML file
        
        Returns:
            Plotly figure
        """
        residuals = y_true - y_pred
        x = dates if dates is not None else np.arange(len(y_true))
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Forecast vs Actual',
                'Error Distribution',
                'Residuals Over Time',
                'Actual vs Predicted Scatter',
                'Feature Importance (Top 10)',
                'Cumulative Error'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. Forecast vs Actual
        fig.add_trace(
            go.Scatter(x=x, y=y_true, mode='lines', name='Actual',
                      line=dict(color=self.color_palette[0])),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=y_pred, mode='lines', name='Predicted',
                      line=dict(color=self.color_palette[1], dash='dash')),
            row=1, col=1
        )
        
        # 2. Error Distribution
        fig.add_trace(
            go.Histogram(x=residuals, nbinsx=30, name='Residuals',
                        marker=dict(color=self.color_palette[2])),
            row=1, col=2
        )
        
        # 3. Residuals Over Time
        fig.add_trace(
            go.Scatter(x=x, y=residuals, mode='markers', name='Residuals',
                      marker=dict(size=5, color=self.color_palette[3])),
            row=2, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)
        
        # 4. Actual vs Predicted Scatter
        fig.add_trace(
            go.Scatter(x=y_pred, y=y_true, mode='markers', name='Predictions',
                      marker=dict(size=6, color=self.color_palette[4])),
            row=2, col=2
        )
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                      mode='lines', name='Perfect',
                      line=dict(color='red', dash='dash')),
            row=2, col=2
        )
        
        # 5. Feature Importance
        if importance_df is not None:
            top_10 = importance_df.head(10).sort_values('importance')
            fig.add_trace(
                go.Bar(y=top_10['feature'], x=top_10['importance'],
                      orientation='h', name='Importance',
                      marker=dict(color=self.color_palette[5])),
                row=3, col=1
            )
        
        # 6. Cumulative Error
        cumsum_error = np.cumsum(np.abs(residuals))
        fig.add_trace(
            go.Scatter(x=x, y=cumsum_error, mode='lines',
                      name='Cumulative |Error|',
                      line=dict(color=self.color_palette[6])),
            row=3, col=2
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_xaxes(title_text="Error", row=1, col=2)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_xaxes(title_text="Predicted", row=2, col=2)
        fig.update_xaxes(title_text="Importance", row=3, col=1)
        fig.update_xaxes(title_text="Time", row=3, col=2)
        
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_yaxes(title_text="Residual", row=2, col=1)
        fig.update_yaxes(title_text="Actual", row=2, col=2)
        fig.update_yaxes(title_text="Feature", row=3, col=1)
        fig.update_yaxes(title_text="Cumulative", row=3, col=2)
        
        # Add metrics as annotation if provided
        if metrics:
            metrics_text = "<br>".join([f"{k}: {v:.4f}" for k, v in list(metrics.items())[:5]])
            fig.add_annotation(
                text=metrics_text,
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                font=dict(size=10)
            )
        
        fig.update_layout(
            title_text=title,
            showlegend=True,
            height=1200,
            template=self.theme
        )
        
        if save_html:
            fig.write_html(save_html)
            print(f"Dashboard saved to {save_html}")
        
        return fig


if __name__ == "__main__":
    print("Visualization Module - Demo\n")
    print("="*60)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
    
    y_true = np.sin(np.linspace(0, 4*np.pi, n_samples)) * 50 + 200 + np.random.randn(n_samples) * 10
    y_pred = y_true + np.random.randn(n_samples) * 5
    
    # Initialize visualizer
    viz = InteractiveVisualizer()
    
    print("1. Creating Forecast Plot...")
    fig1 = viz.plot_forecast(
        y_true, y_pred, dates=dates,
        title='Sample Forecast',
        save_html='forecast_interactive.html'
    )
    
    print("\n2. Creating Residual Analysis...")
    fig2 = viz.plot_residuals(
        y_true, y_pred, dates=dates,
        save_html='residuals_interactive.html'
    )
    
    print("\n3. Creating Feature Importance Plot...")
    importance_df = pd.DataFrame({
        'feature': [f'feature_{i}' for i in range(20)],
        'importance': np.random.rand(20)
    }).sort_values('importance', ascending=False)
    
    fig3 = viz.plot_feature_importance(
        importance_df, top_n=15,
        save_html='importance_interactive.html'
    )
    
    print("\n4. Creating Prediction Intervals...")
    fig4 = viz.plot_prediction_intervals(
        y_true, y_pred, dates=dates,
        confidence_levels=[0.68, 0.95],
        save_html='intervals_interactive.html'
    )
    
    print("\n5. Creating Complete Dashboard...")
    metrics = {
        'RMSE': np.sqrt(np.mean((y_true - y_pred)**2)),
        'MAE': np.mean(np.abs(y_true - y_pred)),
        'R²': 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)
    }
    
    fig5 = viz.create_dashboard(
        y_true, y_pred, dates=dates,
        metrics=metrics,
        importance_df=importance_df,
        save_html='dashboard_interactive.html'
    )
    
    print("\n" + "="*60)
    print("Demo complete! Check the generated HTML files.")
    print("="*60)