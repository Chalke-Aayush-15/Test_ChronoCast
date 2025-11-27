from rest_framework import serializers
from .models import Dataset, ForecastRun, ModelComparison, ExplainabilityResult

class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = '__all__'

class DatasetCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = ['name', 'description', 'file']

class ForecastRunSerializer(serializers.ModelSerializer):
    dataset_name = serializers.CharField(source='dataset.name', read_only=True)

    class Meta:
        model = ForecastRun
        fields = '__all__'

class ForecastRunCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = ForecastRun
        fields = [
            'dataset', 'model_type', 'model_params',
            'use_time_features', 'use_lag_features', 'lag_periods',
            'use_rolling_features', 'rolling_windows'
        ]

class ModelComparisonSerializer(serializers.ModelSerializer):
    dataset_name = serializers.CharField(source='dataset.name', read_only=True)
    forecast_runs_details = ForecastRunSerializer(source='forecast_runs', many=True, read_only=True)

    class Meta:
        model = ModelComparison
        fields = '__all__'

class ExplainabilityResultSerializer(serializers.ModelSerializer):
    forecast_run_id = serializers.CharField(source='forecast_run.id', read_only=True)
    model_type = serializers.CharField(source='forecast_run.model_type', read_only=True)

    class Meta:
        model = ExplainabilityResult
        fields = '__all__'
