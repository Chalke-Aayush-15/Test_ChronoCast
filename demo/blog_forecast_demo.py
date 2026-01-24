"""
ChronoCast Demo Project: AI Blog Forecasting System
Complete end-to-end demonstration
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Import ChronoCast
from chronocast import (
    ChronoModel,
    create_all_features,
    evaluate_model,
    compare_models,
    ModelExplainer,
    InteractiveVisualizer,
    ChronoLogger,
    TimeSeriesDataLoader,
    generate_sample_data
)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)

print("="*80)
print(" "*20 + "ChronoCast Demo: AI Blog Forecasting")
print("="*80)

# ============================================================
# STEP 1: GENERATE REALISTIC BLOG DATA
# ============================================================
print("\n" + "="*80)
print("STEP 1: Generating Realistic Blog Engagement Data")
print("="*80)

np.random.seed(42)

# 18 months of daily blog data
n_days = 540
start_date = datetime(2023, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(n_days)]

# Simulate realistic patterns
# 1. Long-term growth (startup to established blog)
base_growth = np.linspace(100, 1000, n_days)

# 2. Weekly seasonality (weekday vs weekend traffic)
day_of_week = np.array([d.weekday() for d in dates])
weekly_pattern = np.where(day_of_week < 5, 150, -80)  # Higher on weekdays

# 3. Monthly pattern (month-end spikes)
day_of_month = np.array([d.day for d in dates])
monthly_pattern = 80 * np.sin(2 * np.pi * day_of_month / 30)

# 4. Seasonal trends (higher in Q4, lower in summer)
month = np.array([d.month for d in dates])
seasonal = np.where((month >= 10) | (month <= 2), 100, 
                    np.where((month >= 6) & (month <= 8), -50, 0))

# 5. Random events (viral posts, holidays)
random_spikes = np.random.choice([0, 0, 0, 0, 200], n_days)  # 20% chance of spike

# 6. Noise
noise = np.random.normal(0, 50, n_days)

# Combine all components
views = base_growth + weekly_pattern + monthly_pattern + seasonal + random_spikes + noise
views = np.maximum(views, 50).astype(int)  # Ensure positive

# Content categories with different engagement
categories = np.random.choice(
    ['tech', 'lifestyle', 'business', 'tutorial', 'news'],
    n_days,
    p=[0.3, 0.2, 0.2, 0.2, 0.1]
)

# Category impact
category_boost = pd.Series(categories).map({
    'tech': 100,
    'tutorial': 150,
    'business': 80,
    'lifestyle': 60,
    'news': 120
}).values

views += category_boost.astype(int)

# Create DataFrame
data = pd.DataFrame({
    'date': dates,
    'views': views,
    'category': categories,
    'day_name': [d.strftime('%A') for d in dates],
    'is_viral': random_spikes > 0
})

print(f"\n✓ Generated {len(data)} days of blog data")
print(f"  Date range: {data['date'].min().date()} to {data['date'].max().date()}")
print(f"  Views: {data['views'].min():,} - {data['views'].max():,}")
print(f"  Average: {data['views'].mean():.0f} views/day")
print(f"  Total views: {data['views'].sum():,}")

# Statistics by category
print(f"\n  Views by Category:")
for cat in data['category'].unique():
    avg = data[data['category'] == cat]['views'].mean()
    print(f"    {cat:12s}: {avg:6.0f} avg views")

# Save dataset
data.to_csv('demo/blog_data.csv', index=False)
print(f"\n✓ Dataset saved to 'demo/blog_data.csv'")

# ============================================================
# STEP 2: EXPLORATORY DATA ANALYSIS
# ============================================================
print("\n" + "="*80)
print("STEP 2: Exploratory Data Analysis")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: Time series
axes[0, 0].plot(data['date'], data['views'], linewidth=1, alpha=0.7)
axes[0, 0].set_title('Blog Views Over Time', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Views')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Distribution
axes[0, 1].hist(data['views'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Views Distribution', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Views')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Weekly pattern
weekly_avg = data.groupby('day_name')['views'].mean().reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
)
axes[1, 0].bar(range(7), weekly_avg.values, color='steelblue', edgecolor='black')
axes[1, 0].set_xticks(range(7))
axes[1, 0].set_xticklabels(weekly_avg.index, rotation=45)
axes[1, 0].set_title('Average Views by Day of Week', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Average Views')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Plot 4: Category comparison
category_avg = data.groupby('category')['views'].mean().sort_values(ascending=False)
axes[1, 1].barh(category_avg.index, category_avg.values, color='coral', edgecolor='black')
axes[1, 1].set_title('Average Views by Category', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Average Views')
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('demo/01_exploratory_analysis.png', dpi=150, bbox_inches='tight')
print("\n✓ EDA plot saved to 'demo/01_exploratory_analysis.png'")

# ============================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================
print("\n" + "="*80)
print("STEP 3: Feature Engineering")
print("="*80)

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
print(f"  After engineering: {featured_data.shape[1]}")
print(f"  New features: {featured_data.shape[1] - data.shape[1]}")
print(f"  Usable samples: {len(featured_data)}")

# ============================================================
# STEP 4: TRAIN MULTIPLE MODELS
# ============================================================
print("\n" + "="*80)
print("STEP 4: Training Multiple Models")
print("="*80)

# Initialize logger
logger = ChronoLogger(log_dir='demo/logs')

# Split data
split_idx = int(len(featured_data) * 0.8)
train_data = featured_data[:split_idx]
test_data = featured_data[split_idx:]

feature_cols = [col for col in featured_data.columns 
                if col not in ['date', 'views', 'category', 'day_name', 'is_viral']]

X_train = train_data[feature_cols]
y_train = train_data['views']
X_test = test_data[feature_cols]
y_test = test_data['views']

print(f"\n  Training samples: {len(X_train)}")
print(f"  Test samples: {len(X_test)}")
print(f"  Features: {len(feature_cols)}")

# Models to compare
models_config = {
    'Linear Regression': {'type': 'linear', 'params': {}},
    'Ridge Regression': {'type': 'ridge', 'params': {'alpha': 10.0}},
    'Random Forest': {'type': 'rf', 'params': {'n_estimators': 100, 'max_depth': 15}},
    'Gradient Boosting': {'type': 'gbm', 'params': {'n_estimators': 100, 'learning_rate': 0.1}},
    'XGBoost': {'type': 'xgb', 'params': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6}},
}

results = {}

for name, config in models_config.items():
    print(f"\n  Training {name}...")
    
    # Start logging
    run_id = logger.start_training(name, config['params'])
    logger.log_training_data(len(X_train), len(feature_cols), feature_cols)
    
    # Train
    model = ChronoModel(config['type'], **config['params'])
    model.fit(X_train, y_train)
    
    # Predict
    predictions = model.predict(X_test)
    
    # Evaluate
    metrics = evaluate_model(y_test, predictions, y_train)
    
    # End logging
    logger.end_training(model.training_history['training_time'], metrics)
    
    # Store
    results[name] = {
        'model': model,
        'predictions': predictions,
        'metrics': metrics
    }
    
    print(f"    ✓ RMSE: {metrics['RMSE']:.2f}, R²: {metrics['R²']:.4f}, Time: {model.training_history['training_time']:.2f}s")

# ============================================================
# STEP 5: MODEL COMPARISON
# ============================================================
print("\n" + "="*80)
print("STEP 5: Model Comparison")
print("="*80)

# Compare models
model_comparison = {
    name: {'y_true': y_test, 'y_pred': res['predictions'], 'y_train': y_train}
    for name, res in results.items()
}

comparison_df = compare_models(model_comparison, metric='RMSE')
print("\n" + comparison_df.to_string(index=False))

best_model_name = comparison_df.iloc[0]['Model']
best_model = results[best_model_name]['model']
best_predictions = results[best_model_name]['predictions']

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   RMSE: {results[best_model_name]['metrics']['RMSE']:.2f}")
print(f"   R²: {results[best_model_name]['metrics']['R²']:.4f}")

# ============================================================
# STEP 6: VISUALIZE RESULTS
# ============================================================
print("\n" + "="*80)
print("STEP 6: Creating Visualizations")
print("="*80)

# Create comprehensive visualization
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Forecast comparison
ax1 = fig.add_subplot(gs[0, :2])
test_dates = test_data['date'].values
ax1.plot(test_dates, y_test.values, label='Actual', linewidth=2, marker='o', markersize=3)
ax1.plot(test_dates, best_predictions, label=f'Predicted ({best_model_name})', 
         linewidth=2, marker='x', markersize=3, linestyle='--', alpha=0.8)
ax1.fill_between(test_dates, 
                 best_predictions - results[best_model_name]['metrics']['RMSE'],
                 best_predictions + results[best_model_name]['metrics']['RMSE'],
                 alpha=0.2, label='±RMSE')
ax1.set_title('Blog Views Forecast', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Views')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# 2. Model comparison
ax2 = fig.add_subplot(gs[0, 2])
comparison_df_sorted = comparison_df.sort_values('RMSE', ascending=False)
ax2.barh(comparison_df_sorted['Model'], comparison_df_sorted['RMSE'], color='steelblue')
ax2.set_title('Model RMSE Comparison', fontsize=12, fontweight='bold')
ax2.set_xlabel('RMSE')
ax2.grid(True, alpha=0.3, axis='x')

# 3. Error distribution
ax3 = fig.add_subplot(gs[1, 0])
errors = y_test.values - best_predictions
ax3.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='coral')
ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax3.set_title('Prediction Error Distribution', fontsize=12, fontweight='bold')
ax3.set_xlabel('Error')
ax3.set_ylabel('Frequency')
ax3.grid(True, alpha=0.3)

# 4. Actual vs Predicted scatter
ax4 = fig.add_subplot(gs[1, 1])
ax4.scatter(best_predictions, y_test.values, alpha=0.6, s=30)
min_val = min(y_test.min(), best_predictions.min())
max_val = max(y_test.max(), best_predictions.max())
ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax4.set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
ax4.set_xlabel('Predicted Views')
ax4.set_ylabel('Actual Views')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Feature importance
ax5 = fig.add_subplot(gs[1, 2])
importance_df = best_model.get_feature_importance()
if importance_df is not None:
    top_10 = importance_df.head(10)
    ax5.barh(range(len(top_10)), top_10['importance'].values, color='green', alpha=0.7)
    ax5.set_yticks(range(len(top_10)))
    ax5.set_yticklabels(top_10['feature'].values, fontsize=8)
    ax5.invert_yaxis()
    ax5.set_title('Top 10 Feature Importance', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Importance')
    ax5.grid(True, alpha=0.3, axis='x')

# 6. Residuals over time
ax6 = fig.add_subplot(gs[2, :])
ax6.scatter(test_dates, errors, alpha=0.6, s=20, c=np.abs(errors), cmap='RdYlGn_r')
ax6.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax6.fill_between(test_dates, 
                 -results[best_model_name]['metrics']['RMSE'],
                 results[best_model_name]['metrics']['RMSE'],
                 alpha=0.2, color='gray')
ax6.set_title('Prediction Residuals Over Time', fontsize=12, fontweight='bold')
ax6.set_xlabel('Date')
ax6.set_ylabel('Residual')
ax6.grid(True, alpha=0.3)
ax6.tick_params(axis='x', rotation=45)

plt.savefig('demo/02_complete_forecast_analysis.png', dpi=150, bbox_inches='tight')
print("\n✓ Complete analysis saved to 'demo/02_complete_forecast_analysis.png'")

# ============================================================
# STEP 7: EXPLAINABILITY
# ============================================================
print("\n" + "="*80)
print("STEP 7: Model Explainability")
print("="*80)

print("\n  Generating SHAP analysis...")
explainer = ModelExplainer(best_model, X_train, feature_names=feature_cols)

# Feature importance
print("\n  Creating feature importance plot...")
importance_df = explainer.plot_feature_importance(
    top_n=15,
    save_path='demo/03_feature_importance.png'
)

# SHAP analysis
try:
    print("\n  Calculating SHAP values...")
    shap_values = explainer.calculate_shap_values(X_test, max_samples=50)
    
    if shap_values is not None:
        print("  Creating SHAP summary plot...")
        explainer.plot_shap_summary(X_test, max_samples=50, save_path='demo/04_shap_summary.png')
        
        print("  Creating SHAP waterfall for instance 0...")
        explainer.plot_shap_waterfall(X_test, instance_idx=0, save_path='demo/05_shap_waterfall.png')
        
        print("\n✓ SHAP analysis complete")
except Exception as e:
    print(f"\n⚠️  SHAP analysis skipped: {str(e)}")

# Save explainability log
explainer.save_explainability_log(X_test, 'demo/explainability_log.json')

# ============================================================
# STEP 8: GENERATE REPORT
# ============================================================
print("\n" + "="*80)
print("STEP 8: Generating Final Report")
print("="*80)

logger.export_logs('demo/training_history.json')
logger.generate_report('demo/training_report.txt')

# Create summary report
with open('demo/DEMO_SUMMARY.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write(" "*20 + "ChronoCast Blog Forecasting Demo - Summary\n")
    f.write("="*80 + "\n\n")
    
    f.write("Dataset Information:\n")
    f.write("-" *80 + "\n")
    f.write(f"  Total days: {len(data)}\n")
    f.write(f"  Date range: {data['date'].min().date()} to {data['date'].max().date()}\n")
    f.write(f"  Total views: {data['views'].sum():,}\n")
    f.write(f"  Average views/day: {data['views'].mean():.0f}\n")
    f.write(f"  Categories: {', '.join(data['category'].unique())}\n\n")
    
    f.write("Model Performance:\n")
    f.write("-"*80 + "\n")
    for name, res in results.items():
        metrics = res['metrics']
        f.write(f"\n{name}:\n")
        f.write(f"  RMSE: {metrics['RMSE']:.2f}\n")
        f.write(f"  MAE: {metrics['MAE']:.2f}\n")
        f.write(f"  R²: {metrics['R²']:.4f}\n")
        f.write(f"  MAPE: {metrics['MAPE']:.2f}%\n")
    
    f.write(f"\n\nBest Model: {best_model_name}\n")
    f.write("-"*80 + "\n")
    f.write(f"  Test RMSE: {results[best_model_name]['metrics']['RMSE']:.2f}\n")
    f.write(f"  Test R²: {results[best_model_name]['metrics']['R²']:.4f}\n")
    f.write(f"  Training time: {best_model.training_history['training_time']:.2f}s\n")
    
    if importance_df is not None:
        f.write(f"\n\nTop 5 Important Features:\n")
        f.write("-"*80 + "\n")
        for idx, row in importance_df.head(5).iterrows():
            f.write(f"  {row['feature']:30s}: {row['importance']:.4f}\n")
    
    f.write("\n\nGenerated Files:\n")
    f.write("-"*80 + "\n")
    f.write("  • blog_data.csv - Original dataset\n")
    f.write("  • 01_exploratory_analysis.png - EDA visualizations\n")
    f.write("  • 02_complete_forecast_analysis.png - Model results\n")
    f.write("  • 03_feature_importance.png - Feature importance\n")
    f.write("  • 04_shap_summary.png - SHAP analysis\n")
    f.write("  • 05_shap_waterfall.png - SHAP waterfall\n")
    f.write("  • training_history.json - Training logs\n")
    f.write("  • training_report.txt - Detailed report\n")
    f.write("  • explainability_log.json - Explainability metadata\n")

print("\n✓ Summary report saved to 'demo/DEMO_SUMMARY.txt'")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*80)
print("Demo Complete! 🎉")
print("="*80)

print(f"\n  📊 Dataset: {len(data)} days of blog engagement data")
print(f"  🤖 Models trained: {len(results)}")
print(f"  🏆 Best model: {best_model_name}")
print(f"  📈 Best RMSE: {results[best_model_name]['metrics']['RMSE']:.2f}")
print(f"  📉 Best R²: {results[best_model_name]['metrics']['R²']:.4f}")

print(f"\n  Generated Files:")
print(f"    • Dataset: blog_data.csv")
print(f"    • Visualizations: 5 PNG files")
print(f"    • Reports: 3 files")
print(f"    • Logs: Training history & explainability")

print(f"\n  Check the 'demo/' directory for all outputs!")
print("="*80 + "\n")