"""
ChronoCast Complete Pipeline - End-to-End Demo
Demonstrates: Feature Engineering → Model Training → Evaluation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Import ChronoCast modules
from chronocast.core.feature_engineering import create_all_features
from chronocast.core.model_wrapper import ChronoModel
from chronocast.core.evaluation import (
    evaluate_model,
    compare_models,
    plot_forecast_comparison,
    plot_residuals,
    plot_metrics_comparison,
    save_evaluation_report
)

print("="*70)
print(" "*15 + "ChronoCast Complete Pipeline Demo")
print("="*70)

# ============================================================
# STEP 1: CREATE REALISTIC TIME SERIES DATA
# ============================================================
print("\n" + "="*70)
print("STEP 1: Creating Realistic Blog Engagement Dataset")
print("="*70)

np.random.seed(42)

# Generate 18 months of daily data
start_date = datetime(2023, 1, 1)
n_days = 540
dates = [start_date + timedelta(days=i) for i in range(n_days)]

# Simulate realistic blog views with:
# 1. Long-term growth trend
# 2. Weekly seasonality (more views on weekdays)
# 3. Monthly patterns
# 4. Random noise

# Base trend (growing from 200 to 800 views)
trend = np.linspace(200, 800, n_days)

# Weekly seasonality (peak on weekdays)
day_of_week = np.array([d.weekday() for d in dates])
weekly_pattern = np.where(day_of_week < 5, 100, -50)  # +100 on weekdays, -50 on weekends

# Monthly seasonality (higher at month start/end)
day_of_month = np.array([d.day for d in dates])
monthly_pattern = 50 * np.sin(2 * np.pi * day_of_month / 30)

# Random noise
noise = np.random.normal(0, 40, n_days)

# Combine all components
views = trend + weekly_pattern + monthly_pattern + noise
views = np.maximum(views, 50).astype(int)  # Ensure positive values

# Create categories with different baseline views
categories = np.random.choice(['tech', 'lifestyle', 'business'], n_days, p=[0.4, 0.3, 0.3])
category_boost = np.where(categories == 'tech', 50, 
                          np.where(categories == 'lifestyle', 20, 30))
views += category_boost.astype(int)

# Create DataFrame
data = pd.DataFrame({
    'date': dates,
    'views': views,
    'category': categories
})

print(f"\n✓ Generated {len(data)} days of blog view data")
print(f"  Date range: {data['date'].min().date()} to {data['date'].max().date()}")
print(f"  Views range: {data['views'].min()} to {data['views'].max()}")
print(f"  Average views: {data['views'].mean():.0f}")
print(f"  Categories: {data['category'].value_counts().to_dict()}")

# ============================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================
print("\n" + "="*70)
print("STEP 2: Feature Engineering")
print("="*70)

print("\n  Creating time-based and domain features...")

featured_data = create_all_features(
    data,
    date_col='date',
    target_col='views',
    categorical_cols=['category'],
    lags=[1, 7, 14, 30],
    windows=[7, 14, 30]
)

print(f"\n✓ Feature engineering complete")
print(f"  Original features: {data.shape[1]}")
print(f"  After feature engineering: {featured_data.shape[1]}")
print(f"  New features added: {featured_data.shape[1] - data.shape[1]}")
print(f"  Usable samples: {len(featured_data)} (after dropping NaN)")

# Show some feature names
feature_cols = [col for col in featured_data.columns 
                if col not in ['date', 'views', 'category']]
print(f"\n  Sample features: {feature_cols[:10]}")

# ============================================================
# STEP 3: TRAIN/TEST SPLIT
# ============================================================
print("\n" + "="*70)
print("STEP 3: Preparing Train/Test Split")
print("="*70)

# Use time-based split (last 20% as test)
split_idx = int(len(featured_data) * 0.8)

train_data = featured_data[:split_idx]
test_data = featured_data[split_idx:]

X_train = train_data[feature_cols]
y_train = train_data['views']
X_test = test_data[feature_cols]
y_test = test_data['views']

print(f"\n✓ Data split complete")
print(f"  Training samples: {len(X_train)} ({len(X_train)/len(featured_data)*100:.1f}%)")
print(f"  Testing samples: {len(X_test)} ({len(X_test)/len(featured_data)*100:.1f}%)")
print(f"  Number of features: {len(feature_cols)}")
print(f"  Test period: {test_data['date'].min().date()} to {test_data['date'].max().date()}")

# ============================================================
# STEP 4: TRAIN MULTIPLE MODELS
# ============================================================
print("\n" + "="*70)
print("STEP 4: Training Multiple Models")
print("="*70)

models = {
    'Linear Regression': ChronoModel('linear'),
    'Ridge Regression': ChronoModel('ridge', alpha=10.0),
    'Random Forest': ChronoModel('rf', n_estimators=100, max_depth=15, random_state=42),
    'Gradient Boosting': ChronoModel('gbm', n_estimators=100, learning_rate=0.1, random_state=42),
    'XGBoost': ChronoModel('xgb', n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n  Training {name}...")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Store results
    results[name] = {
        'model': model,
        'train_pred': train_pred,
        'test_pred': test_pred
    }
    
    print(f"    ✓ Training time: {model.training_history['training_time']:.3f}s")

print(f"\n✓ All {len(models)} models trained successfully")

# ============================================================
# STEP 5: EVALUATE MODELS
# ============================================================
print("\n" + "="*70)
print("STEP 5: Model Evaluation")
print("="*70)

print("\n  Calculating metrics for all models...")

# Prepare data for comparison
model_comparison = {}
for name, result in results.items():
    model_comparison[name] = {
        'y_true': y_test,
        'y_pred': result['test_pred'],
        'y_train': y_train
    }

# Compare models
comparison_df = compare_models(model_comparison, metric='RMSE', sort_ascending=True)

print("\n✓ Model Comparison Results:")
print("\n" + comparison_df.to_string(index=False))

best_model_name = comparison_df.iloc[0]['Model']
best_rmse = comparison_df.iloc[0]['RMSE']

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Best RMSE: {best_rmse:.2f}")

# ============================================================
# STEP 6: DETAILED ANALYSIS OF BEST MODEL
# ============================================================
print("\n" + "="*70)
print("STEP 6: Detailed Analysis of Best Model")
print("="*70)

best_model = results[best_model_name]['model']
best_predictions = results[best_model_name]['test_pred']

# Detailed metrics
detailed_metrics = evaluate_model(y_test, best_predictions, y_train)

print(f"\n  Detailed Metrics for {best_model_name}:")
print("  " + "-"*50)
for metric, value in detailed_metrics.items():
    print(f"  {metric:20s}: {value:12.4f}")

# Feature importance (if available)
importance = best_model.get_feature_importance()
if importance is not None:
    print(f"\n  Top 10 Most Important Features:")
    print("  " + "-"*50)
    top_features = importance.head(10)
    for idx, row in top_features.iterrows():
        print(f"  {row['feature']:30s}: {row['importance']:8.4f}")
else:
    print(f"\n  ℹ {best_model_name} does not support feature importance")

# ============================================================
# STEP 7: SAVE EVALUATION REPORT
# ============================================================
print("\n" + "="*70)
print("STEP 7: Saving Evaluation Report")
print("="*70)

report_path = f'{best_model_name.replace(" ", "_")}_report.json'
save_evaluation_report(detailed_metrics, best_model_name, report_path)
print(f"\n✓ Report saved to: {report_path}")

# Save best model
model_path = f'{best_model_name.replace(" ", "_")}_model.pkl'
best_model.save(model_path)
print(f"✓ Model saved to: {model_path}")

# ============================================================
# STEP 8: VISUALIZATIONS
# ============================================================
print("\n" + "="*70)
print("STEP 8: Creating Visualizations")
print("="*70)

# 1. Forecast Comparison
print("\n  Creating forecast comparison plot...")
plot_forecast_comparison(
    y_test, 
    best_predictions,
    dates=test_data['date'].values,
    model_name=best_model_name,
    save_path='forecast_comparison.png'
)

# 2. Residual Analysis
print("  Creating residual analysis plot...")
plot_residuals(
    y_test,
    best_predictions,
    model_name=best_model_name,
    save_path='residual_analysis.png'
)

# 3. Model Comparison
print("  Creating model comparison plot...")
plot_metrics_comparison(
    comparison_df,
    metrics=['RMSE', 'MAE', 'R²'],
    save_path='model_comparison.png'
)

# 4. Custom visualization: Actual vs Predicted over time
print("  Creating detailed time series plot...")
fig, axes = plt.subplots(2, 1, figsize=(15, 10))

# Plot 1: Full forecast comparison
axes[0].plot(test_data['date'].values, y_test.values, 
            label='Actual', linewidth=2, marker='o', markersize=3)
axes[0].plot(test_data['date'].values, best_predictions, 
            label=f'Predicted ({best_model_name})', linewidth=2, marker='x', markersize=3, alpha=0.8)
axes[0].fill_between(test_data['date'].values, 
                     best_predictions - best_rmse, 
                     best_predictions + best_rmse,
                     alpha=0.2, label=f'±{best_rmse:.0f} RMSE')
axes[0].set_xlabel('Date', fontsize=12)
axes[0].set_ylabel('Views', fontsize=12)
axes[0].set_title('Blog Views Forecast - Complete Test Period', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(axis='x', rotation=45)

# Plot 2: Error over time
errors = y_test.values - best_predictions
axes[1].plot(test_data['date'].values, errors, 
            color='red', linewidth=1, marker='o', markersize=3, alpha=0.6)
axes[1].axhline(y=0, color='black', linestyle='--', linewidth=2)
axes[1].axhline(y=best_rmse, color='gray', linestyle=':', linewidth=1, label=f'+RMSE ({best_rmse:.0f})')
axes[1].axhline(y=-best_rmse, color='gray', linestyle=':', linewidth=1, label=f'-RMSE ({best_rmse:.0f})')
axes[1].set_xlabel('Date', fontsize=12)
axes[1].set_ylabel('Prediction Error', fontsize=12)
axes[1].set_title('Prediction Error Over Time', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('detailed_forecast_analysis.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved to: detailed_forecast_analysis.png")

# ============================================================
# STEP 9: SUMMARY STATISTICS
# ============================================================
print("\n" + "="*70)
print("STEP 9: Summary Statistics")
print("="*70)

print(f"\n  Test Set Statistics:")
print(f"    Actual views - Mean: {y_test.mean():.1f}, Std: {y_test.std():.1f}")
print(f"    Predicted views - Mean: {best_predictions.mean():.1f}, Std: {best_predictions.std():.1f}")
print(f"    Correlation: {np.corrcoef(y_test, best_predictions)[0, 1]:.4f}")

# Calculate percentage of predictions within acceptable error
acceptable_error = 0.15  # 15%
within_error = np.abs((y_test.values - best_predictions) / y_test.values) <= acceptable_error
accuracy_rate = (within_error.sum() / len(y_test)) * 100

print(f"\n  Forecast Accuracy:")
print(f"    Predictions within ±{acceptable_error*100:.0f}%: {accuracy_rate:.1f}%")
print(f"    Max error: {np.abs(errors).max():.1f} views")
print(f"    Min error: {np.abs(errors).min():.1f} views")

# ============================================================
# STEP 10: CONCLUSION
# ============================================================
print("\n" + "="*70)
print("STEP 10: Pipeline Complete! 🎉")
print("="*70)

print(f"\n  Summary:")
print(f"    • Processed {len(data)} days of blog data")
print(f"    • Created {len(feature_cols)} features")
print(f"    • Trained {len(models)} different models")
print(f"    • Best model: {best_model_name}")
print(f"    • Test RMSE: {best_rmse:.2f}")
print(f"    • Test R²: {detailed_metrics['R²']:.4f}")

print(f"\n  Generated Files:")
print(f"    • {report_path}")
print(f"    • {model_path}")
print(f"    • forecast_comparison.png")
print(f"    • residual_analysis.png")
print(f"    • model_comparison.png")
print(f"    • detailed_forecast_analysis.png")

print("\n" + "="*70)
print("  ChronoCast pipeline executed successfully!")
print("  Next steps: Deploy this model or try with your own data!")
print("="*70 + "\n")