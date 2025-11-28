"""
API Views for ChronoCast
"""

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from django.shortcuts import get_object_or_404
from django.utils import timezone
import pandas as pd
import numpy as np
import json
import traceback
from pathlib import Path

from .models import Dataset, ForecastRun, ModelComparison, ExplainabilityResult
from .serializers import (
    DatasetSerializer, DatasetCreateSerializer,
    ForecastRunSerializer, ForecastRunCreateSerializer,
    ModelComparisonSerializer, ExplainabilityResultSerializer
)

# Import ChronoCast
from chronocast import (
    ChronoModel,
    create_all_features,
    evaluate_model,
    compare_models,
    ModelExplainer,
    TimeSeriesDataLoader
)

import logging
logger = logging.getLogger(__name__)


class DatasetViewSet(viewsets.ModelViewSet):
    """
    ViewSet for Dataset operations
    """
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer
    
    def get_parser_classes(self):
        """Use different parsers based on action"""
        if self.action == 'create':
            return [MultiPartParser, FormParser]
        return super().get_parser_classes()
    
    def get_serializer_class(self):
        if self.action == 'create':
            return DatasetCreateSerializer
        return DatasetSerializer
    
    def create(self, request, *args, **kwargs):
        """Upload and validate dataset"""
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        dataset = serializer.save()
        
        # Load and validate data
        try:
            loader = TimeSeriesDataLoader()
            data = loader.load_csv(dataset.file.path)
            
            # Update metadata
            dataset.n_rows = len(data)
            dataset.n_columns = len(data.columns)
            dataset.columns = list(data.columns)
            dataset.save()
            
            logger.info(f"Dataset uploaded: {dataset.id}")
            
        except Exception as e:
            dataset.delete()
            logger.error(f"Dataset validation failed: {str(e)}")
            return Response(
                {'error': f'Dataset validation failed: {str(e)}'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        output_serializer = DatasetSerializer(dataset, context={'request': request})
        return Response(output_serializer.data, status=status.HTTP_201_CREATED)
    
    @action(detail=True, methods=['post'])
    def validate(self, request, pk=None):
        """Validate dataset for time series forecasting"""
        dataset = self.get_object()
        date_col = request.data.get('date_column')
        target_col = request.data.get('target_column')
        
        if not date_col or not target_col:
            return Response(
                {'error': 'date_column and target_column are required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            loader = TimeSeriesDataLoader()
            data = loader.load_csv(dataset.file.path, date_col=date_col)
            validation = loader.validate_time_series(date_col, target_col)
            
            # Update dataset with validated columns
            dataset.date_column = date_col
            dataset.target_column = target_col
            dataset.save()
            
            return Response(validation)
        
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
    
    @action(detail=True, methods=['get'])
    def preview(self, request, pk=None):
        """Preview first N rows of dataset"""
        dataset = self.get_object()
        n_rows = int(request.query_params.get('n_rows', 10))
        
        try:
            data = pd.read_csv(dataset.file.path, nrows=n_rows)
            return Response({
                'data': data.to_dict(orient='records'),
                'columns': list(data.columns)
            })
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )


class ForecastRunViewSet(viewsets.ModelViewSet):
    """
    ViewSet for ForecastRun operations
    """
    queryset = ForecastRun.objects.all()
    serializer_class = ForecastRunSerializer
    
    def get_serializer_class(self):
        if self.action == 'create':
            return ForecastRunCreateSerializer
        return ForecastRunSerializer
    
    def create(self, request, *args, **kwargs):
        """Create and start a forecast run"""
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        forecast_run = serializer.save()
        
        # Start forecast asynchronously (in production, use Celery)
        try:
            self._run_forecast(forecast_run.id)
        except Exception as e:
            forecast_run.status = 'failed'
            forecast_run.error_message = str(e)
            forecast_run.save()
            logger.error(f"Forecast failed: {str(e)}\n{traceback.format_exc()}")
        
        output_serializer = ForecastRunSerializer(forecast_run)
        return Response(output_serializer.data, status=status.HTTP_201_CREATED)
    
    def _run_forecast(self, run_id):
        """Execute forecast run"""
        forecast_run = ForecastRun.objects.get(id=run_id)
        dataset = forecast_run.dataset
        
        logger.info(f"Starting forecast run: {run_id}")
        
        # Update status
        forecast_run.status = 'running'
        forecast_run.started_at = timezone.now()
        forecast_run.progress = 10
        forecast_run.save()
        
        try:
            # Validate dataset columns are set
            if not dataset.date_column or not dataset.target_column:
                raise ValueError(
                    f"Dataset columns not configured. "
                    f"Please validate the dataset first by setting date_column and target_column."
                )
            
            # Load data
            loader = TimeSeriesDataLoader()
            data = loader.load_csv(
                dataset.file.path,
                date_col=dataset.date_column
            )
            
            forecast_run.progress = 20
            forecast_run.save()
            
            # Create features
            lag_periods = forecast_run.lag_periods or [1, 7, 14]
            rolling_windows = forecast_run.rolling_windows or [7, 14]
            
            featured_data = create_all_features(
                data,
                date_col=dataset.date_column,
                target_col=dataset.target_column,
                lags=lag_periods if forecast_run.use_lag_features else [],
                windows=rolling_windows if forecast_run.use_rolling_features else []
            )
            
            forecast_run.progress = 40
            forecast_run.save()
            
            # Split data
            split_idx = int(len(featured_data) * 0.8)
            train_data = featured_data[:split_idx]
            test_data = featured_data[split_idx:]
            
            feature_cols = [col for col in featured_data.columns 
                          if col not in [dataset.date_column, dataset.target_column]]
            
            X_train = train_data[feature_cols]
            y_train = train_data[dataset.target_column]
            X_test = test_data[feature_cols]
            y_test = test_data[dataset.target_column]
            
            forecast_run.n_train_samples = len(X_train)
            forecast_run.n_test_samples = len(X_test)
            forecast_run.n_features = len(feature_cols)
            forecast_run.progress = 50
            forecast_run.save()
            
            # Train model
            model = ChronoModel(
                forecast_run.model_type,
                **forecast_run.model_params
            )
            model.fit(X_train, y_train)
            
            forecast_run.training_time = model.training_history['training_time']
            forecast_run.progress = 70
            forecast_run.save()
            
            # Make predictions
            predictions = model.predict(X_test)
            
            # Evaluate
            metrics = evaluate_model(y_test, predictions, y_train)
            
            forecast_run.progress = 80
            forecast_run.save()
            
            # Get feature importance
            importance_df = model.get_feature_importance()
            if importance_df is not None:
                feature_importance = importance_df.to_dict(orient='records')
            else:
                feature_importance = None
            
            # Prepare predictions for storage
            predictions_data = []
            test_dates = test_data[dataset.date_column].values
            
            for i, (date, actual, pred) in enumerate(zip(test_dates, y_test.values, predictions)):
                predictions_data.append({
                    'index': i,
                    'date': str(date),
                    'actual': float(actual),
                    'predicted': float(pred),
                    'error': float(actual - pred)
                })
            
            # Save model
            from django.conf import settings
            model_filename = f"model_{run_id}.pkl"
            model_path = Path(settings.CHRONOCAST_MODEL_DIR) / model_filename
            model.save(str(model_path))
            
            # Update forecast run
            forecast_run.status = 'completed'
            forecast_run.completed_at = timezone.now()
            forecast_run.progress = 100
            forecast_run.metrics = metrics
            forecast_run.predictions = predictions_data
            forecast_run.feature_importance = feature_importance
            forecast_run.model_file = model_filename
            forecast_run.save()
            
            logger.info(f"Forecast run completed: {run_id}")
            
        except Exception as e:
            forecast_run.status = 'failed'
            forecast_run.error_message = str(e)
            forecast_run.completed_at = timezone.now()
            forecast_run.save()
            logger.error(f"Forecast run failed: {str(e)}\n{traceback.format_exc()}")
            raise
    
    @action(detail=True, methods=['get'])
    def metrics(self, request, pk=None):
        """Get forecast metrics"""
        forecast_run = self.get_object()
        
        if not forecast_run.metrics:
            return Response(
                {'error': 'Metrics not available yet'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        return Response(forecast_run.metrics)
    
    @action(detail=True, methods=['get'])
    def predictions(self, request, pk=None):
        """Get forecast predictions"""
        forecast_run = self.get_object()
        
        if not forecast_run.predictions:
            return Response(
                {'error': 'Predictions not available yet'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Pagination
        page_size = int(request.query_params.get('page_size', 100))
        page = int(request.query_params.get('page', 1))
        
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        predictions = forecast_run.predictions[start_idx:end_idx]
        
        return Response({
            'count': len(forecast_run.predictions),
            'page': page,
            'page_size': page_size,
            'results': predictions
        })
    
    @action(detail=True, methods=['post'])
    def generate_explainability(self, request, pk=None):
        """Generate explainability analysis"""
        forecast_run = self.get_object()
        
        if forecast_run.status != 'completed':
            return Response(
                {'error': 'Forecast must be completed first'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Load dataset
            dataset = forecast_run.dataset
            loader = TimeSeriesDataLoader()
            data = loader.load_csv(dataset.file.path, date_col=dataset.date_column)
            
            # Recreate features
            lag_periods = forecast_run.lag_periods or [1, 7, 14]
            rolling_windows = forecast_run.rolling_windows or [7, 14]
            
            featured_data = create_all_features(
                data,
                date_col=dataset.date_column,
                target_col=dataset.target_column,
                lags=lag_periods if forecast_run.use_lag_features else [],
                windows=rolling_windows if forecast_run.use_rolling_features else []
            )
            
            # Split data
            split_idx = int(len(featured_data) * 0.8)
            train_data = featured_data[:split_idx]
            test_data = featured_data[split_idx:]
            
            feature_cols = [col for col in featured_data.columns 
                          if col not in [dataset.date_column, dataset.target_column]]
            
            X_train = train_data[feature_cols]
            X_test = test_data[feature_cols]
            
            # Load model
            from django.conf import settings
            model_path = Path(settings.CHRONOCAST_MODEL_DIR) / forecast_run.model_file
            model = ChronoModel.load(str(model_path))
            
            # Create explainer
            explainer = ModelExplainer(model, X_train, feature_names=feature_cols)
            
            # Calculate SHAP values (limited samples for performance)
            max_samples = int(request.data.get('max_samples', 50))
            shap_values = explainer.calculate_shap_values(X_test, max_samples=max_samples)
            
            # Get feature importance
            importance_df = explainer.plot_feature_importance(top_n=20)
            
            # Create or update explainability result
            explainability, created = ExplainabilityResult.objects.get_or_create(
                forecast_run=forecast_run
            )
            
            explainability.feature_importance = importance_df.to_dict(orient='records') if importance_df is not None else None
            explainability.explainer_type = explainer.explainer_type if hasattr(explainer, 'explainer_type') else 'unknown'
            explainability.n_samples_explained = max_samples
            
            if shap_values is not None:
                # Store SHAP values (limited for performance)
                explainability.shap_values = shap_values[:10].tolist()  # Store first 10
                if hasattr(explainer.explainer, 'expected_value'):
                    expected_value = explainer.explainer.expected_value
                    explainability.shap_base_value = float(expected_value) if np.isscalar(expected_value) else float(expected_value[0])
            
            explainability.save()
            
            serializer = ExplainabilityResultSerializer(explainability)
            return Response(serializer.data)
            
        except Exception as e:
            logger.error(f"Explainability generation failed: {str(e)}\n{traceback.format_exc()}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ModelComparisonViewSet(viewsets.ModelViewSet):
    """
    ViewSet for ModelComparison operations
    """
    queryset = ModelComparison.objects.all()
    serializer_class = ModelComparisonSerializer
    
    @action(detail=False, methods=['post'])
    def create_comparison(self, request):
        """Create a model comparison"""
        dataset_id = request.data.get('dataset_id')
        run_ids = request.data.get('forecast_run_ids', [])
        name = request.data.get('name', 'Model Comparison')
        
        if not dataset_id or not run_ids:
            return Response(
                {'error': 'dataset_id and forecast_run_ids are required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            dataset = Dataset.objects.get(id=dataset_id)
            forecast_runs = ForecastRun.objects.filter(id__in=run_ids, dataset=dataset)
            
            if len(forecast_runs) < 2:
                return Response(
                    {'error': 'At least 2 forecast runs are required for comparison'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Create comparison
            comparison = ModelComparison.objects.create(
                dataset=dataset,
                name=name,
                description=request.data.get('description', '')
            )
            comparison.forecast_runs.set(forecast_runs)
            
            # Prepare comparison data
            comparison_data = []
            for run in forecast_runs:
                if run.metrics:
                    comparison_data.append({
                        'run_id': str(run.id),
                        'model_type': run.model_type,
                        'metrics': run.metrics,
                        'training_time': run.training_time
                    })
            
            # Find best model
            if comparison_data:
                # Sort by RMSE (lower is better)
                best = min(comparison_data, key=lambda x: x['metrics'].get('RMSE', float('inf')))
                comparison.best_model = best['model_type']
                comparison.best_metric_value = best['metrics']['RMSE']
            
            comparison.comparison_data = comparison_data
            comparison.save()
            
            serializer = ModelComparisonSerializer(comparison)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
            
        except Dataset.DoesNotExist:
            return Response(
                {'error': 'Dataset not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            logger.error(f"Comparison creation failed: {str(e)}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=True, methods=['get'])
    def chart_data(self, request, pk=None):
        """Get formatted data for comparison charts"""
        comparison = self.get_object()
        
        if not comparison.comparison_data:
            return Response(
                {'error': 'No comparison data available'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Format for frontend charts
        chart_data = {
            'models': [],
            'metrics': {},
            'training_times': {}
        }
        
        for item in comparison.comparison_data:
            model_name = item['model_type']
            chart_data['models'].append(model_name)
            
            for metric_name, value in item['metrics'].items():
                if metric_name not in chart_data['metrics']:
                    chart_data['metrics'][metric_name] = []
                chart_data['metrics'][metric_name].append(value)
            
            chart_data['training_times'][model_name] = item.get('training_time', 0)
        
        return Response(chart_data)


class ExplainabilityViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for Explainability results
    """
    queryset = ExplainabilityResult.objects.all()
    serializer_class = ExplainabilityResultSerializer
    
    @action(detail=True, methods=['get'])
    def feature_contributions(self, request, pk=None):
        """Get feature contributions for a specific instance"""
        explainability = self.get_object()
        instance_idx = int(request.query_params.get('instance_idx', 0))
        
        if not explainability.shap_values:
            return Response(
                {'error': 'SHAP values not available'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if instance_idx >= len(explainability.shap_values):
            return Response(
                {'error': f'Instance index out of range. Max: {len(explainability.shap_values)-1}'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Get SHAP values for instance
        shap_vals = explainability.shap_values[instance_idx]
        
        # Get feature names from forecast run
        forecast_run = explainability.forecast_run
        if forecast_run.feature_importance:
            feature_names = [f['feature'] for f in forecast_run.feature_importance]
        else:
            feature_names = [f'feature_{i}' for i in range(len(shap_vals))]
        
        # Create contributions list
        contributions = []
        for fname, sval in zip(feature_names, shap_vals):
            contributions.append({
                'feature': fname,
                'shap_value': float(sval),
                'abs_shap_value': abs(float(sval))
            })
        
        # Sort by absolute SHAP value
        contributions.sort(key=lambda x: x['abs_shap_value'], reverse=True)
        
        return Response({
            'instance_idx': instance_idx,
            'base_value': explainability.shap_base_value,
            'contributions': contributions[:20]  # Top 20
        })