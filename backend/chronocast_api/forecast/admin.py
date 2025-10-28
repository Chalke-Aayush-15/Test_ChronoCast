"""
Django Admin Configuration for ChronoCast
"""

from django.contrib import admin
from django.utils.html import format_html
from .models import Dataset, ForecastRun, ModelComparison, ExplainabilityResult


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    """Admin for Dataset model"""
    
    list_display = ['name', 'n_rows', 'n_columns', 'date_column', 'target_column', 'uploaded_at', 'file_link']
    list_filter = ['uploaded_at']
    search_fields = ['name', 'description']
    readonly_fields = ['id', 'uploaded_at', 'updated_at', 'n_rows', 'n_columns', 'columns']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('id', 'name', 'description', 'file')
        }),
        ('Metadata', {
            'fields': ('n_rows', 'n_columns', 'columns', 'date_column', 'target_column')
        }),
        ('Timestamps', {
            'fields': ('uploaded_at', 'updated_at')
        }),
    )
    
    def file_link(self, obj):
        if obj.file:
            return format_html('<a href="{}" target="_blank">Download</a>', obj.file.url)
        return '-'
    file_link.short_description = 'File'


@admin.register(ForecastRun)
class ForecastRunAdmin(admin.ModelAdmin):
    """Admin for ForecastRun model"""
    
    list_display = ['id', 'dataset', 'model_type', 'status', 'progress_bar', 'created_at', 'duration']
    list_filter = ['status', 'model_type', 'created_at']
    search_fields = ['id', 'dataset__name']
    readonly_fields = [
        'id', 'status', 'progress', 'error_message',
        'metrics', 'predictions', 'feature_importance',
        'created_at', 'started_at', 'completed_at',
        'training_time', 'n_train_samples', 'n_test_samples', 'n_features'
    ]
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('id', 'dataset', 'model_type', 'model_params')
        }),
        ('Feature Engineering', {
            'fields': (
                'use_time_features', 'use_lag_features', 'lag_periods',
                'use_rolling_features', 'rolling_windows'
            )
        }),
        ('Status', {
            'fields': ('status', 'progress', 'error_message')
        }),
        ('Results', {
            'fields': ('metrics', 'feature_importance', 'model_file'),
            'classes': ('collapse',)
        }),
        ('Training Metadata', {
            'fields': (
                'training_time', 'n_train_samples', 'n_test_samples', 'n_features'
            )
        }),
        ('Timestamps', {
            'fields': ('created_at', 'started_at', 'completed_at')
        }),
    )
    
    def progress_bar(self, obj):
        if obj.status == 'completed':
            color = 'green'
        elif obj.status == 'failed':
            color = 'red'
        elif obj.status == 'running':
            color = 'blue'
        else:
            color = 'gray'
        
        return format_html(
            '<div style="width:100px; height:20px; border:1px solid #ccc;">'
            '<div style="width:{}px; height:20px; background-color:{};"></div>'
            '</div>',
            obj.progress,
            color
        )
    progress_bar.short_description = 'Progress'
    
    def duration(self, obj):
        if obj.started_at and obj.completed_at:
            delta = obj.completed_at - obj.started_at
            return f"{delta.total_seconds():.2f}s"
        return '-'
    duration.short_description = 'Duration'
    
    actions = ['delete_with_models']
    
    def delete_with_models(self, request, queryset):
        """Delete forecast runs and their model files"""
        from pathlib import Path
        from django.conf import settings
        
        count = 0
        for run in queryset:
            if run.model_file:
                model_path = Path(settings.CHRONOCAST_MODEL_DIR) / run.model_file
                if model_path.exists():
                    model_path.unlink()
            run.delete()
            count += 1
        
        self.message_user(request, f"Deleted {count} forecast runs and their models.")
    delete_with_models.short_description = "Delete selected runs and their models"


@admin.register(ModelComparison)
class ModelComparisonAdmin(admin.ModelAdmin):
    """Admin for ModelComparison model"""
    
    list_display = ['name', 'dataset', 'run_count', 'best_model', 'best_metric_value', 'created_at']
    list_filter = ['created_at']
    search_fields = ['name', 'description', 'dataset__name']
    readonly_fields = ['id', 'created_at', 'updated_at', 'comparison_data', 'best_model', 'best_metric_value']
    filter_horizontal = ['forecast_runs']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('id', 'dataset', 'name', 'description')
        }),
        ('Forecast Runs', {
            'fields': ('forecast_runs',)
        }),
        ('Results', {
            'fields': ('comparison_data', 'best_model', 'best_metric_value')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at')
        }),
    )
    
    def run_count(self, obj):
        return obj.forecast_runs.count()
    run_count.short_description = 'Runs'


@admin.register(ExplainabilityResult)
class ExplainabilityResultAdmin(admin.ModelAdmin):
    """Admin for ExplainabilityResult model"""
    
    list_display = ['id', 'forecast_run', 'explainer_type', 'n_samples_explained', 'created_at', 'has_shap']
    list_filter = ['explainer_type', 'created_at']
    search_fields = ['forecast_run__id', 'forecast_run__dataset__name']
    readonly_fields = ['id', 'created_at', 'feature_importance', 'shap_values', 'shap_base_value']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('id', 'forecast_run', 'explainer_type', 'n_samples_explained')
        }),
        ('Feature Importance', {
            'fields': ('feature_importance',)
        }),
        ('SHAP Analysis', {
            'fields': ('shap_values', 'shap_base_value'),
            'classes': ('collapse',)
        }),
        ('Visualizations', {
            'fields': ('importance_plot', 'shap_summary_plot', 'shap_waterfall_plot'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at',)
        }),
    )
    
    def has_shap(self, obj):
        return '✓' if obj.shap_values else '✗'
    has_shap.short_description = 'SHAP Available'


# Customize admin site
admin.site.site_header = "ChronoCast Administration"
admin.site.site_title = "ChronoCast Admin"
admin.site.index_title = "Welcome to ChronoCast Admin"