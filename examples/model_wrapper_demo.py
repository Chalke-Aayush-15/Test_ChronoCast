"""
ChronoCast Model Wrapper - Demo Script
Shows how to use different models and compare them
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Import ChronoCast modules
from chronocast.core.feature_engineering import create_all_features
from chronocast.core.model_wrapper import ChronoModel, model_registry

# Set random seed for reproducibility
np.random.seed(42)

print("="*60)
print("ChronoCast Model Wrapper Demo")
print("="*60)

# ============================================================
# 1. CREATE SAMPLE TIME SERIES DATA
# ============================================================
print("\n1. Creating Sample Blog Views Data...")

# Generate dates
start_date = datetime(2023, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(365)]

# Simulate blog views with trend and seasonality
trend = np.linspace(100, 500, 365)
seasonality = 100 * np.sin(np.arange(365) * 2 * np.pi / 7)  # Weekly pattern
noise = np.random.normal(0, 50, 365)
views = trend + seasonality + noise + 200

# Create DataFrame
data = pd.DataFrame({
    'date': dates,
    'views': views.astype(int),
    'category': np.random.choice(['tech', 'lifestyle', 'business'], 365)
})

print(f"Generated {len(data)} days of blog view data")
print(f"Date range: {data['date'].min()} to {data['date'].max()}")
print(f"Views range: {data['views'].min()} to {data['views'].max()}")

# ============================================================
# 2. CREATE FEATURES
# ============================================================
print("\n2. Creating Features...")

featured_data = create_all_features(
    data,
    date_col='date',
    target_col='views',
    categorical_cols=['category'],
    lags=[1, 7, 14],
    windows=[7, 14]
)

print(f"Original features: {data.shape[1]}")
print(f"After feature engineering: {featured_data.shape[1]}")
print(f"New feature count: {featured_data.shape[1] - data.shape[1]}")

# ============================================================
# 3. PREPARE TRAIN/TEST SPLIT
# ============================================================
print("\n3. Splitting Data (80/20 train/test)...")

# Use time-based split (last 20% as test)
split_idx = int(len(featured_data) * 0.8)

train_data = featured_data[:split_idx]
test_data = featured_data[split_idx:]

# Separate features and target
feature_cols = [col for col in featured_data.columns 
                if col not in ['date', 'views', 'category']]

X_train = train_data[feature_cols]
y_train = train_data['views']
X_test = test_data[feature_cols]
y_test = test_data['views']

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Number of features: {len(feature_cols)}")

# ============================================================
# 4. TRAIN MULTIPLE MODELS
# ============================================================
print("\n4. Training Multiple Models...")

models_to_test = {
    'Linear Regression': ChronoModel('linear'),
    'Ridge Regression': ChronoModel('ridge', alpha=1.0),
    'Random Forest': ChronoModel('rf', n_estimators=100, max_depth=10),
    'Gradient Boosting': ChronoModel('gbm', n_estimators=100, learning_rate=0.1),
    'XGBoost': ChronoModel('xgb', n_estimators=100, learning_rate=0.1)
}

results = {}

for name, model in models_to_test.items():
    print(f"\n  Training {name}...")
    model.fit(X_train, y_train)
    
    # Get predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_mae = mean_absolute_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    results[name] = {
        'model': model,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'predictions': test_pred,
        'training_time': model.training_history['training_time']
    }
    
    print(f"    Train RMSE: {train_rmse:.2f}")
    print(f"    Test RMSE: {test_rmse:.2f}")
    print(f"    Test MAE: {test_mae:.2f}")
    print(f"    Test R²: {test_r2:.3f}")
    print(f"    Training time: {model.training_history['training_time']:.3f}s")

# ============================================================
# 5. COMPARE MODELS
# ============================================================
print("\n5. Model Comparison Summary")
print("="*60)

comparison_df = pd.DataFrame({
    'Model': results.keys(),
    'Test RMSE': [r['test_rmse'] for r in results.values()],
    'Test MAE': [r['test_mae'] for r in results.values()],
    'Test R²': [r['test_r2'] for r in results.values()],
    'Training Time (s)': [r['training_time'] for r in results.values()]
})

comparison_df = comparison_df.sort_values('Test RMSE')
print(comparison_df.to_string(index=False))

best_model_name = comparison_df.iloc[0]['Model']
print(f"\n✅ Best Model: {best_model_name}")

# ============================================================
# 6. FEATURE IMPORTANCE (for tree models)
# ============================================================
print("\n6. Feature Importance Analysis...")

best_model = results[best_model_name]['model']
importance = best_model.get_feature_importance()

if importance is not None:
    print("\nTop 10 Most Important Features:")
    print(importance.head(10).to_string(index=False))
else:
    print(f"\n{best_model_name} does not support feature importance")

# ============================================================
# 7. SAVE BEST MODEL
# ============================================================
print("\n7. Saving Best Model...")

model_path = 'best_model.pkl'
best_model.save(model_path)
print(f"Model saved to {model_path}")

# Load and verify
loaded_model = ChronoModel.load(model_path)
print(f"Model loaded successfully: {loaded_model}")

# Verify predictions match
loaded_predictions = loaded_model.predict(X_test)
assert np.allclose(results[best_model_name]['predictions'], loaded_predictions)
print("✅ Model save/load verified!")

# ============================================================
# 8. VISUALIZE PREDICTIONS
# ============================================================
print("\n8. Visualizing Predictions...")

plt.figure(figsize=(15, 5))

# Plot actual vs predicted for best model
plt.subplot(1, 2, 1)
plt.plot(test_data['date'].values, y_test.values, label='Actual', marker='o', markersize=3)
plt.plot(test_data['date'].values, results[best_model_name]['predictions'], 
         label='Predicted', marker='x', markersize=3)
plt.xlabel('Date')
plt.ylabel('Views')
plt.title(f'Forecast: {best_model_name}')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Plot residuals
plt.subplot(1, 2, 2)
residuals = y_test.values - results[best_model_name]['predictions']
plt.scatter(results[best_model_name]['predictions'], residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Views')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=100, bbox_inches='tight')
print("Plot saved to 'model_comparison.png'")

# ============================================================
# 9. MODEL REGISTRY DEMO
# ============================================================
print("\n9. Model Registry Demo...")

print("\nAvailable models:")
print(model_registry.list_models())

# Register custom model example
print("\nRegistering custom model...")
from sklearn.linear_model import ElasticNet

model_registry.register('elasticnet', ElasticNet)
print("Available models after registration:")
print(model_registry.list_models())

# Use custom model
custom_model = ChronoModel('elasticnet', alpha=0.5, l1_ratio=0.5)
custom_model.fit(X_train, y_train)
print(f"Custom model trained: {custom_model}")

print("\n" + "="*60)
print("Demo Complete!")
print("="*60)