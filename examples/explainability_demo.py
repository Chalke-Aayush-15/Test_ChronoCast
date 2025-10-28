"""
ChronoCast Explainability Demo
Demonstrates model interpretability features with SHAP
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Import ChronoCast modules
from chronocast.core.feature_engineering import create_all_features
from chronocast.core.model_wrapper import ChronoModel
from chronocast.core.evaluation import evaluate_model
from chronocast.core.explainability import ModelExplainer, SHAP_AVAILABLE

print("="*70)
print(" "*15 + "ChronoCast Explainability Demo")
print("="*70)

if not SHAP_AVAILABLE:
    print("\n⚠️  WARNING: SHAP not installed!")
    print("   Install with: pip install shap")
    print("   Some features will be limited.\n")

# ============================================================
# STEP 1: CREATE TIME SERIES DATA
# ============================================================
print("\n" + "="*70)
print("STEP 1: Creating Blog Engagement Dataset")
print("="*70)

np.random.seed(42)

# Generate 12 months of daily data
start_date = datetime(2023, 1, 1)
n_days = 365
dates = [start_date + timedelta(days=i) for i in range(n_days)]

# Realistic blog views with multiple components
trend = np.linspace(200, 700, n_days)
day_of_week = np.array([d.weekday() for d in dates])
weekly_pattern = np.where(day_of_week < 5, 80, -40)
noise = np.random.normal(0, 30, n_days)
views = trend + weekly_pattern + noise
views = np.maximum(views, 50).astype(int)

# Categories with different impact
categories = np.random.choice(['tech', 'lifestyle', 'business'], n_days, p=[0.4, 0.3, 0.3])
category_boost = np.where(categories == 'tech', 50, 
                          np.where(categories == 'lifestyle', 20, 30))
views += category_boost.astype(int)

data = pd.DataFrame({
    'date': dates,
    'views': views,
    'category': categories
})

print(f"\n✓ Generated {len(data)} days of data")
print(f"  Date range: {data['date'].min().date()} to {data['date'].max().date()}")
print(f"  Views: {data['views'].min()} - {data['views'].max()} (avg: {data['views'].mean():.0f})")

# ============================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================
print("\n" + "="*70)
print("STEP 2: Feature Engineering")
print("="*70)

featured_data = create_all_features(
    data,
    date_col='date',
    target_col='views',
    categorical_cols=['category'],
    lags=[1, 7, 14],
    windows=[7, 14]
)

print(f"\n✓ Created {featured_data.shape[1]} features from {data.shape[1]} original columns")

# ============================================================
# STEP 3: TRAIN MODEL
# ============================================================
print("\n" + "="*70)
print("STEP 3: Training Model")
print("="*70)

# Split data
split_idx = int(len(featured_data) * 0.8)
train_data = featured_data[:split_idx]
test_data = featured_data[split_idx:]

feature_cols = [col for col in featured_data.columns 
                if col not in ['date', 'views', 'category']]

X_train = train_data[feature_cols]
y_train = train_data['views']
X_test = test_data[feature_cols]
y_test = test_data['views']

print(f"\n  Training samples: {len(X_train)}")
print(f"  Test samples: {len(X_test)}")
print(f"  Features: {len(feature_cols)}")

# Train XGBoost model
print("\n  Training XGBoost model...")
model = ChronoModel('xgb', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
metrics = evaluate_model(y_test, predictions)

print(f"\n  ✓ Model trained successfully")
print(f"    RMSE: {metrics['RMSE']:.2f}")
print(f"    R²: {metrics['R²']:.4f}")

# ============================================================
# STEP 4: INITIALIZE EXPLAINER
# ============================================================
print("\n" + "="*70)
print("STEP 4: Initializing Model Explainer")
print("="*70)

explainer = ModelExplainer(
    model=model,
    X_train=X_train,
    feature_names=feature_cols
)

print(f"\n✓ Explainer initialized")
print(f"  Model type: {type(model.model).__name__}")
if hasattr(explainer, 'explainer_type'):
    print(f"  Explainer type: {explainer.explainer_type}")

# ============================================================
# STEP 5: FEATURE IMPORTANCE
# ============================================================
print("\n" + "="*70)
print("STEP 5: Feature Importance Analysis")
print("="*70)

print("\n  Calculating feature importance...")
importance_df = explainer.plot_feature_importance(
    top_n=15,
    figsize=(12, 8),
    save_path='feature_importance.png'
)

if importance_df is not None:
    print("\n  Top 10 Most Important Features:")
    print("  " + "-"*60)
    for idx, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']:30s}: {row['importance']:8.4f}")
    
    # Analyze feature types
    lag_features = importance_df[importance_df['feature'].str.contains('lag')].shape[0]
    rolling_features = importance_df[importance_df['feature'].str.contains('rolling')].shape[0]
    time_features = importance_df[importance_df['feature'].str.contains('month|day|week')].shape[0]
    
    print(f"\n  Feature Type Distribution (Top 15):")
    print(f"    Lag features: {lag_features}")
    print(f"    Rolling features: {rolling_features}")
    print(f"    Time features: {time_features}")

# ============================================================
# STEP 6: SHAP ANALYSIS
# ============================================================
if SHAP_AVAILABLE:
    print("\n" + "="*70)
    print("STEP 6: SHAP Analysis")
    print("="*70)
    
    print("\n  6.1 Calculating SHAP values...")
    shap_values = explainer.calculate_shap_values(X_test, max_samples=50)
    
    if shap_values is not None:
        print(f"  ✓ SHAP values calculated for {shap_values.shape[0]} samples")
        
        # SHAP Summary Plot
        print("\n  6.2 Creating SHAP summary plot...")
        explainer.plot_shap_summary(
            X_test,
            max_samples=50,
            figsize=(12, 8),
            save_path='shap_summary.png'
        )
        
        # SHAP Waterfall for specific instance
        print("\n  6.3 Creating SHAP waterfall plot (Instance 0)...")
        explainer.plot_shap_waterfall(
            X_test,
            instance_idx=0,
            figsize=(12, 8),
            save_path='shap_waterfall_0.png'
        )
        
        # Feature contributions
        print("\n  6.4 Analyzing feature contributions for instance 0:")
        contributions = explainer.get_feature_contributions(X_test, instance_idx=0)
        
        if contributions is not None:
            print("\n  Top 10 Contributing Features:")
            print("  " + "-"*60)
            print(f"  {'Feature':<25} {'Value':>10} {'SHAP':>12}")
            print("  " + "-"*60)
            for idx, row in contributions.head(10).iterrows():
                print(f"  {row['feature']:<25} {row['value']:>10.2f} {row['shap_value']:>12.4f}")
            
            # Analyze prediction
            actual = y_test.iloc[0]
            predicted = predictions[0]
            print(f"\n  Instance 0 Analysis:")
            print(f"    Actual views: {actual:.0f}")
            print(f"    Predicted views: {predicted:.0f}")
            print(f"    Error: {actual - predicted:.0f}")
        
        # Additional waterfall plots
        print("\n  6.5 Creating additional waterfall plots...")
        for idx in [5, 10]:
            if idx < len(X_test):
                explainer.plot_shap_waterfall(
                    X_test,
                    instance_idx=idx,
                    figsize=(12, 8),
                    save_path=f'shap_waterfall_{idx}.png'
                )
                print(f"    ✓ Created waterfall plot for instance {idx}")
else:
    print("\n" + "="*70)
    print("STEP 6: SHAP Analysis - SKIPPED")
    print("="*70)
    print("\n  ⚠️  SHAP not available. Install with: pip install shap")

# ============================================================
# STEP 7: SAVE EXPLAINABILITY LOG
# ============================================================
print("\n" + "="*70)
print("STEP 7: Saving Explainability Log")
print("="*70)

explainer.save_explainability_log(
    X_test,
    output_path='explainability_log.json'
)

print("\n✓ Explainability log saved to: explainability_log.json")

# ============================================================
# STEP 8: INTERPRETABILITY INSIGHTS
# ============================================================
print("\n" + "="*70)
print("STEP 8: Key Interpretability Insights")
print("="*70)

print("\n  📊 Model Transparency Summary:")
print("  " + "-"*60)

# Get feature importance if available
if importance_df is not None:
    top_feature = importance_df.iloc[0]
    print(f"\n  1. Most Important Feature: {top_feature['feature']}")
    print(f"     Importance score: {top_feature['importance']:.4f}")
    
    # Check feature types
    if 'lag' in top_feature['feature']:
        print(f"     Type: Historical lag feature")
        print(f"     Insight: Past values are strong predictors")
    elif 'rolling' in top_feature['feature']:
        print(f"     Type: Rolling statistics")
        print(f"     Insight: Short-term trends matter")
    elif 'day_of_week' in top_feature['feature'] or 'is_weekend' in top_feature['feature']:
        print(f"     Type: Time-based pattern")
        print(f"     Insight: Weekly seasonality is significant")

# Performance metrics
print(f"\n  2. Model Performance:")
print(f"     RMSE: {metrics['RMSE']:.2f}")
print(f"     MAE: {metrics['MAE']:.2f}")
print(f"     R²: {metrics['R²']:.4f}")
print(f"     MAPE: {metrics.get('MAPE', 0):.2f}%")

# Explainability coverage
if SHAP_AVAILABLE and shap_values is not None:
    mean_shap = np.mean(np.abs(shap_values))
    max_shap = np.max(np.abs(shap_values))
    print(f"\n  3. Explainability Metrics:")
    print(f"     Mean |SHAP value|: {mean_shap:.4f}")
    print(f"     Max |SHAP value|: {max_shap:.4f}")
    print(f"     Feature contributions analyzed: ✓")

# ============================================================
# STEP 9: COMPARISON VISUALIZATION
# ============================================================
print("\n" + "="*70)
print("STEP 9: Creating Comparison Visualizations")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Actual vs Predicted with feature importance overlay
ax1 = axes[0, 0]
ax1.scatter(y_test.values, predictions, alpha=0.6, s=50)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', linewidth=2, label='Perfect prediction')
ax1.set_xlabel('Actual Views', fontsize=11)
ax1.set_ylabel('Predicted Views', fontsize=11)
ax1.set_title('Prediction Accuracy', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Forecast over time
ax2 = axes[0, 1]
test_dates = test_data['date'].values
ax2.plot(test_dates, y_test.values, label='Actual', linewidth=2, marker='o', markersize=3)
ax2.plot(test_dates, predictions, label='Predicted', linewidth=2, marker='x', markersize=3, alpha=0.7)
ax2.set_xlabel('Date', fontsize=11)
ax2.set_ylabel('Views', fontsize=11)
ax2.set_title('Time Series Forecast', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

# 3. Feature Importance (if available)
ax3 = axes[1, 0]
if importance_df is not None:
    top_10 = importance_df.head(10)
    ax3.barh(range(len(top_10)), top_10['importance'].values)
    ax3.set_yticks(range(len(top_10)))
    ax3.set_yticklabels(top_10['feature'].values, fontsize=9)
    ax3.set_xlabel('Importance', fontsize=11)
    ax3.set_title('Top 10 Feature Importance', fontsize=12, fontweight='bold')
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3, axis='x')
else:
    ax3.text(0.5, 0.5, 'Feature importance\nnot available', 
             ha='center', va='center', fontsize=12)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')

# 4. Error distribution
ax4 = axes[1, 1]
errors = y_test.values - predictions
ax4.hist(errors, bins=30, edgecolor='black', alpha=0.7)
ax4.axvline(x=0, color='r', linestyle='--', linewidth=2)
ax4.set_xlabel('Prediction Error', fontsize=11)
ax4.set_ylabel('Frequency', fontsize=11)
ax4.set_title('Error Distribution', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('explainability_summary.png', dpi=150, bbox_inches='tight')
print("\n✓ Summary visualization saved to: explainability_summary.png")

# ============================================================
# STEP 10: FINAL SUMMARY
# ============================================================
print("\n" + "="*70)
print("STEP 10: Explainability Analysis Complete! 🎉")
print("="*70)

print("\n  Generated Files:")
print("  " + "-"*60)
print("  ✓ feature_importance.png - Feature importance chart")
if SHAP_AVAILABLE:
    print("  ✓ shap_summary.png - SHAP summary plot")
    print("  ✓ shap_waterfall_*.png - Individual prediction explanations")
print("  ✓ explainability_log.json - Detailed explainability log")
print("  ✓ explainability_summary.png - Complete summary dashboard")

print("\n  Key Takeaways:")
print("  " + "-"*60)
print("  1. Model predictions are transparent and interpretable")
print("  2. Feature importance identifies key predictors")
if SHAP_AVAILABLE:
    print("  3. SHAP values explain individual predictions")
    print("  4. Waterfall plots show contribution of each feature")
print(f"  5. Model achieves R² of {metrics['R²']:.4f}")

print("\n" + "="*70)
print("  Explainability demo completed successfully!")
print("  ChronoCast provides full model transparency! 🔍")
print("="*70 + "\n")