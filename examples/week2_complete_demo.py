"""
ChronoCast Week 2 Complete Demo
Demonstrates all Week 2 features: Explainability + Visualization + Logging
"""

import numpy as np
import pandas as pd
from datetime import datetime

# Import all ChronoCast modules
from chronocast import (
    # Core
    ChronoModel,
    create_all_features,
    evaluate_model,
    compare_models,
    ModelExplainer,
    InteractiveVisualizer,
    # Utils
    ChronoLogger,
    ExperimentTracker,
    TimeSeriesDataLoader,
    generate_sample_data
)

print("="*70)
print(" "*10 + "ChronoCast Week 2 - Complete Integration Demo")
print("="*70)

# ============================================================
# STEP 1: DATA LOADING & VALIDATION
# ============================================================
print("\n" + "="*70)
print("STEP 1: Data Loading & Validation")
print("="*70)

# Generate sample data
print("\n  Generating sample blog engagement data...")
data = generate_sample_data(
    n_samples=365,
    start_date='2023-01-01',
    trend='linear',
    seasonality=True,
    noise_level=0.15
)

# Rename columns for clarity
data = data.rename(columns={'value': 'views'})

# Load with data loader
loader = TimeSeriesDataLoader()
loader.data = data

# Validate
print("\n  Validating data...")
validation = loader.validate_time_series('date', 'views')
print(f"  ✓ Data valid: {validation['is_valid']}")
print(f"  ✓ Date range: {validation['date_range']['start'].date()} to {validation['date_range']['end'].date()}")
print(f"  ✓ Samples: {validation['date_range']['n_periods']}")

# ============================================================
# STEP 2: INITIALIZE LOGGING
# ============================================================
print("\n" + "="*70)
print("STEP 2: Initialize Logging & Experiment Tracking")
print("="*70)

logger = ChronoLogger(log_dir='week2_logs')
experiment = ExperimentTracker('week2_demo', log_dir='week2_experiments')

print("  ✓ Logger initialized")
print("  ✓ Experiment tracker initialized")

# ============================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================
print("\n" + "="*70)
print("STEP 3: Feature Engineering")
print("="*70)

print("\n  Creating features...")
featured_data = create_all_features(
    data,
    date_col='date',
    target_col='views',
    categorical_cols=['category'],
    lags=[1, 7, 14],
    windows=[7, 14]
)

print(f"  ✓ Created {featured_data.shape[1]} features")

# ============================================================
# STEP 4: TRAIN MULTIPLE MODELS WITH LOGGING
# ============================================================
print("\n" + "="*70)
print("STEP 4: Training Multiple Models with Logging")
print("="*70)

# Split data
train, test = loader.train_test_split('date', test_size=0.2)

# Update featured_data split
split_idx = int(len(featured_data) * 0.8)
train_featured = featured_data[:split_idx]
test_featured = featured_data[split_idx:]

feature_cols = [col for col in featured_data.columns 
                if col not in ['date', 'views', 'category']]

X_train = train_featured[feature_cols]
y_train = train_featured['views']
X_test = test_featured[feature_cols]
y_test = test_featured['views']

# Models to train
models_config = {
    'Random Forest': {'model_type': 'rf', 'params': {'n_estimators': 100, 'max_depth': 10}},
    'XGBoost': {'model_type': 'xgb', 'params': {'n_estimators': 100, 'learning_rate': 0.1}},
    'Gradient Boosting': {'model_type': 'gbm', 'params': {'n_estimators': 100, 'learning_rate': 0.1}}
}

results = {}

for name, config in models_config.items():
    print(f"\n  Training {name}...")
    
    # Start logging
    run_id = logger.start_training(name, config['params'])
    logger.log_training_data(len(X_train), len(feature_cols), feature_cols)
    
    try:
        # Train model
        model = ChronoModel(config['model_type'], **config['params'])
        model.fit(X_train, y_train)
        
        # Predict
        predictions = model.predict(X_test)
        
        # Evaluate
        metrics = evaluate_model(y_test, predictions, y_train)
        
        # End logging
        logger.end_training(model.training_history['training_time'], metrics)
        
        # Log experiment
        experiment.log_experiment(name, config['params'], metrics)
        
        # Store results
        results[name] = {
            'model': model,
            'predictions': predictions,
            'metrics': metrics
        }
        
        print(f"    ✓ RMSE: {metrics['RMSE']:.2f}, R²: {metrics['R²']:.4f}")
    
    except Exception as e:
        logger.log_error(f"Training failed for {name}", e)
        print(f"    ✗ Training failed: {str(e)}")

# ============================================================
# STEP 5: MODEL COMPARISON
# ============================================================
print("\n" + "="*70)
print("STEP 5: Model Comparison")
print("="*70)

# Compare models
model_comparison_data = {
    name: {'y_true': y_test, 'y_pred': res['predictions'], 'y_train': y_train}
    for name, res in results.items()
}

comparison_df = compare_models(model_comparison_data, metric='RMSE')
print("\n" + comparison_df.to_string(index=False))

best_model_name = comparison_df.iloc[0]['Model']
print(f"\n  🏆 Best Model: {best_model_name}")

# ============================================================
# STEP 6: EXPLAINABILITY ANALYSIS
# ============================================================
print("\n" + "="*70)
print("STEP 6: Explainability Analysis")
print("="*70)

best_model = results[best_model_name]['model']
best_predictions = results[best_model_name]['predictions']

# Initialize explainer
print("\n  Initializing explainer...")
explainer = ModelExplainer(best_model, X_train, feature_names=feature_cols)

# Feature importance
print("\n  Generating feature importance...")
importance_df = explainer.plot_feature_importance(
    top_n=15,
    save_path='week2_feature_importance.png'
)

if importance_df is not None:
    print(f"  ✓ Top feature: {importance_df.iloc[0]['feature']}")

# SHAP analysis (if available)
try:
    print("\n  Calculating SHAP values...")
    shap_values = explainer.calculate_shap_values(X_test, max_samples=50)
    
    if shap_values is not None:
        print("  ✓ SHAP values calculated")
        
        explainer.plot_shap_summary(
            X_test,
            max_samples=50,
            save_path='week2_shap_summary.png'
        )
        
        explainer.plot_shap_waterfall(
            X_test,
            instance_idx=0,
            save_path='week2_shap_waterfall.png'
        )
except Exception as e:
    print(f"  ⚠️  SHAP analysis skipped: {str(e)}")

# Save explainability log
explainer.save_explainability_log(X_test, 'week2_explainability_log.json')

# ============================================================
# STEP 7: INTERACTIVE VISUALIZATIONS
# ============================================================
print("\n" + "="*70)
print("STEP 7: Creating Interactive Visualizations")
print("="*70)

viz = InteractiveVisualizer()

# 1. Interactive forecast plot
print("\n  Creating interactive forecast plot...")
fig1 = viz.plot_forecast(
    y_test,
    best_predictions,
    dates=test_featured['date'].values,
    title=f'Interactive Forecast - {best_model_name}',
    save_html='week2_forecast_interactive.html'
)

# 2. Interactive residual analysis
print("  Creating interactive residual analysis...")
fig2 = viz.plot_residuals(
    y_test,
    best_predictions,
    dates=test_featured['date'].values,
    save_html='week2_residuals_interactive.html'
)

# 3. Model comparison chart
print("  Creating model comparison chart...")
fig3 = viz.plot_model_comparison(
    comparison_df,
    metrics=['RMSE', 'MAE', 'R²'],
    save_html='week2_comparison_interactive.html'
)

# 4. Feature importance (interactive)
if importance_df is not None:
    print("  Creating interactive feature importance...")
    fig4 = viz.plot_feature_importance(
        importance_df,
        top_n=20,
        save_html='week2_importance_interactive.html'
    )

# 5. Complete dashboard
print("  Creating complete dashboard...")
fig5 = viz.create_dashboard(
    y_test,
    best_predictions,
    dates=test_featured['date'].values,
    metrics=results[best_model_name]['metrics'],
    importance_df=importance_df,
    title=f'ChronoCast Dashboard - {best_model_name}',
    save_html='week2_dashboard.html'
)

# ============================================================
# STEP 8: EXPORT LOGS & REPORTS
# ============================================================
print("\n" + "="*70)
print("STEP 8: Exporting Logs & Reports")
print("="*70)

# Export training history
logger.export_logs('week2_training_history.json')
logger.generate_report('week2_training_report.txt')

# Export experiment comparison
exp_comparison = experiment.compare_experiments()
exp_comparison.to_csv('week2_experiment_comparison.csv', index=False)
print("  ✓ Experiment comparison saved")

# Get best experiment
best_exp = experiment.get_best_experiment('RMSE', minimize=True)
print(f"\n  Best Experiment: {best_exp['model_name']}")
print(f"  Best RMSE: {best_exp['metrics']['RMSE']:.2f}")

# ============================================================
# STEP 9: FINAL SUMMARY
# ============================================================
print("\n" + "="*70)
print("STEP 9: Week 2 Integration Summary")
print("="*70)

print("\n  📊 Models Trained:")
for name, res in results.items():
    metrics = res['metrics']
    print(f"    • {name}: RMSE={metrics['RMSE']:.2f}, R²={metrics['R²']:.4f}")

print(f"\n  🏆 Best Model: {best_model_name}")
print(f"    RMSE: {results[best_model_name]['metrics']['RMSE']:.2f}")
print(f"    MAE: {results[best_model_name]['metrics']['MAE']:.2f}")
print(f"    R²: {results[best_model_name]['metrics']['R²']:.4f}")

print("\n  📁 Generated Files:")
files = [
    "Feature Importance (PNG)",
    "SHAP Summary (PNG)",
    "SHAP Waterfall (PNG)",
    "Explainability Log (JSON)",
    "Interactive Forecast (HTML)",
    "Interactive Residuals (HTML)",
    "Model Comparison (HTML)",
    "Feature Importance Interactive (HTML)",
    "Complete Dashboard (HTML)",
    "Training History (JSON)",
    "Training Report (TXT)",
    "Experiment Comparison (CSV)"
]

for file in files:
    print(f"    ✓ {file}")

print("\n  🎯 Week 2 Features Demonstrated:")
print("    ✓ SHAP-based explainability")
print("    ✓ Interactive Plotly visualizations")
print("    ✓ Comprehensive logging system")
print("    ✓ Experiment tracking")
print("    ✓ Data validation utilities")
print("    ✓ Model transparency")
print("    ✓ Feature importance analysis")
print("    ✓ Complete dashboard creation")

print("\n" + "="*70)
print("  Week 2 Demo Complete! 🎉")
print("  All explainability & visualization features integrated!")
print("="*70 + "\n")