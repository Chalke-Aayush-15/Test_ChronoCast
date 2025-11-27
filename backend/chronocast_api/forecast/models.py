"""
Django models for ChronoCast API
"""

from django.db import models
from django.core.validators import FileExtensionValidator
import uuid


class Dataset(models.Model):
    """Store uploaded datasets"""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    file = models.FileField(
        upload_to='datasets/',
        validators=[FileExtensionValidator(allowed_extensions=['csv', 'xlsx'])]
    )
    
    # Metadata
    n_rows = models.IntegerField(null=True, blank=True)
    n_columns = models.IntegerField(null=True, blank=True)
    columns = models.JSONField(null=True, blank=True)
    date_column = models.CharField(max_length=100, null=True, blank=True)
    target_column = models.CharField(max_length=100, null=True, blank=True)
    
    # Timestamps
    uploaded_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return f"{self.name} ({self.id})"


class ForecastRun(models.Model):
    """Store forecast run information"""
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    MODEL_CHOICES = [
        ('linear', 'Linear Regression'),
        ('ridge', 'Ridge Regression'),
        ('lasso', 'Lasso Regression'),
        ('rf', 'Random Forest'),
        ('dt', 'Decision Tree'),
        ('gbm', 'Gradient Boosting'),
        ('xgb', 'XGBoost'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='forecast_runs')
    
    # Configuration
    model_type = models.CharField(max_length=20, choices=MODEL_CHOICES)
    model_params = models.JSONField(default=dict)
    
    # Feature engineering config
    use_time_features = models.BooleanField(default=True)
    use_lag_features = models.BooleanField(default=True)
    lag_periods = models.JSONField(default=list)
    use_rolling_features = models.BooleanField(default=True)
    rolling_windows = models.JSONField(default=list)
    
    # Status
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    progress = models.IntegerField(default=0)
    error_message = models.TextField(blank=True, null=True)
    
    # Results
    metrics = models.JSONField(null=True, blank=True)
    predictions = models.JSONField(null=True, blank=True)
    feature_importance = models.JSONField(null=True, blank=True)
    
    # Model storage
    model_file = models.CharField(max_length=500, blank=True, null=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    # Training metadata
    training_time = models.FloatField(null=True, blank=True)
    n_train_samples = models.IntegerField(null=True, blank=True)
    n_test_samples = models.IntegerField(null=True, blank=True)
    n_features = models.IntegerField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.model_type} on {self.dataset.name} ({self.status})"


class ModelComparison(models.Model):
    """Store model comparison results"""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='comparisons')
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    
    # Runs included in comparison
    forecast_runs = models.ManyToManyField(ForecastRun, related_name='comparisons')
    
    # Comparison results
    comparison_data = models.JSONField(null=True, blank=True)
    best_model = models.CharField(max_length=100, null=True, blank=True)
    best_metric_value = models.FloatField(null=True, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Comparison: {self.name}"


class ExplainabilityResult(models.Model):
    """Store explainability analysis results"""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    forecast_run = models.OneToOneField(
        ForecastRun, 
        on_delete=models.CASCADE, 
        related_name='explainability'
    )
    
    # Feature importance
    feature_importance = models.JSONField(null=True, blank=True)
    
    # SHAP values (stored as compressed JSON)
    shap_values = models.JSONField(null=True, blank=True)
    shap_base_value = models.FloatField(null=True, blank=True)
    
    # Visualizations (store paths or base64)
    importance_plot = models.TextField(null=True, blank=True)
    shap_summary_plot = models.TextField(null=True, blank=True)
    shap_waterfall_plot = models.TextField(null=True, blank=True)
    
    # Metadata
    explainer_type = models.CharField(max_length=50, null=True, blank=True)
    n_samples_explained = models.IntegerField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Explainability for {self.forecast_run.id}"