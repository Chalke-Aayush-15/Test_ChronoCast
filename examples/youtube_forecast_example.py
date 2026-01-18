"""
Complete Example: Fetch YouTube Video Data and Forecast Views
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Import ChronoCast
from chronocast import (
    ChronoModel,
    create_all_features,
    evaluate_model,
    ModelExplainer,
    InteractiveVisualizer
)

# Import data fetcher
from chronocast.utils.data_fetchers import YouTubeDataFetcher

print("="*80)
print("YouTube Video Forecast - Complete Example")
print("="*80)

# ============================================================
# STEP 1: FETCH REAL YOUTUBE DATA
# ============================================================
print("\n" + "="*80)
print("STEP 1: Fetching YouTube Video Data")
print("="*80)

# IMPORTANT: Set your YouTube API key
# Get it from: https://console.cloud.google.com/
api_key = os.getenv('YOUTUBE_API_KEY', 'YOUR_API_KEY_HERE')

if api_key == 'YOUR_API_KEY_HERE':
    print("\n⚠️  WARNING: Using simulated data")
    print("   To use real YouTube data:")
    print("   1. Get API key: https://console.cloud.google.com/")
    print("   2. Set: export YOUTUBE_API_KEY='your_key'")
    print("   3. Run this script again")
    
    # SIMULATE DATA for demonstration
    print("\n  Generating simulated YouTube video data...")
    
    dates = pd.date_range('2024-01-01', periods=180, freq='D')
    
    # Simulate viral video growth pattern
    days = np.arange(len(dates))
    views = 1000 * (1 + np.exp(days / 30))  # Exponential growth
    views += np.random.normal(0, views * 0.1)  # Add noise
    views = views.astype(int)
    
    data = pd.DataFrame({
        'date': dates,
        'views': views,
        'video_id': 'simulated_video'
    })
    
    print(f"\n✓ Generated {len(data)} days of simulated data")

else:
    print(f"\n  Using YouTube API key: {api_key[:10]}...")
    
    # Initialize fetcher
    fetcher = YouTubeDataFetcher(api_key)
    
    # Example: Popular video URL
    video_url = input("\nEnter YouTube video URL (or press Enter for example): ").strip()
    if not video_url:
        video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Example
    
    print(f"\n  Fetching data for: {video_url}")
    
    try:
        # Get current stats
        current_stats = fetcher.get_video_stats(video_url)
        print(f"\n  Current Stats:")
        print(f"    Title: {current_stats['title'].iloc[0]}")
        print(f"    Views: {current_stats['views'].iloc[0]:,}")
        print(f"    Likes: {current_stats['likes'].iloc[0]:,}")
        
        # For forecasting, we need historical data
        # Since YouTube API doesn't provide historical views,
        # we'll simulate growth pattern based on publish date
        
        published_date = pd.to_datetime(current_stats['published_at'].iloc[0])
        current_date = datetime.now()
        days_since_publish = (current_date - published_date).days
        
        print(f"\n  Video published: {published_date.date()}")
        print(f"  Days since publish: {days_since_publish}")
        
        # Simulate historical data based on current views
        print("\n  Generating historical data based on growth pattern...")
        
        dates = pd.date_range(published_date, current_date, freq='D')
        current_views = current_stats['views'].iloc[0]
        
        # Simulate exponential growth
        views = current_views * (np.arange(len(dates)) / len(dates)) ** 2
        views += np.random.normal(0, views * 0.05)  # Add realistic noise
        views = np.maximum(views, 0).astype(int)
        
        data = pd.DataFrame({
            'date': dates,
            'views': views,
            'video_id': current_stats['video_id'].iloc[0]
        })
        
        print(f"\n✓ Generated {len(data)} days of historical data")
        
    except Exception as e:
        print(f"\n❌ Error fetching YouTube data: {e}")
        print("   Using simulated data instead...")
        
        # Fallback to simulated data
        dates = pd.date_range('2024-01-01', periods=180, freq='D')
        days = np.arange(len(dates))
        views = 1000 * (1 + np.exp(days / 30))
        views += np.random.normal(0, views * 0.1)
        views = views.astype(int)
        
        data = pd.DataFrame({
            'date': dates,
            'views': views,
            'video_id': 'simulated_video'
        })

# Display data summary
print(f"\n  Data Summary:")
print(f"    Date range: {data['date'].min().date()} to {data['date'].max().date()}")
print(f"    Views range: {data['views'].min():,} to {data['views'].max():,}")
print(f"    Total views: {data['views'].sum():,}")
print(f"    Average daily views: {data['views'].mean():.0f}")

# ============================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================
print("\n" + "="*80)
print("STEP 2: Feature Engineering")
print("="*80)

print("\n  Creating time-based features...")
featured_data = create_all_features(
    data,
    date_col='date',
    target_col='views',
    lags=[1, 7, 14],
    windows=[7, 14, 30]
)

print(f"\n✓ Feature engineering complete")
print(f"  Original features: {data.shape[1]}")
print(f"  After engineering: {featured_data.shape[1]}")
print(f"  Usable samples: {len(featured_data)}")

# ============================================================
# STEP 3: TRAIN MODEL
# ============================================================
print("\n" + "="*80)
print("STEP 3: Training Forecasting Model")
print("="*80)

# Split data (80/20)
split_idx = int(len(featured_data) * 0.8)
train_data = featured_data[:split_idx]
test_data = featured_data[split_idx:]

feature_cols = [col for col in featured_data.columns 
                if col not in ['date', 'views', 'video_id']]

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

# Make predictions
predictions = model.predict(X_test)

# Evaluate
metrics = evaluate_model(y_test, predictions, y_train)

print(f"\n✓ Model trained successfully")
print(f"  Training time: {model.training_history['training_time']:.2f}s")
print(f"\n  Evaluation Metrics:")
print(f"    RMSE: {metrics['RMSE']:.2f}")
print(f"    MAE: {metrics['MAE']:.2f}")
print(f"    R²: {metrics['R²']:.4f}")
print(f"    MAPE: {metrics['MAPE']:.2f}%")

# ============================================================
# STEP 4: VISUALIZE RESULTS
# ============================================================
print("\n" + "="*80)
print("STEP 4: Visualizing Forecast Results")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 1. Forecast
ax1 = axes[0, 0]
test_dates = test_data['date'].values
ax1.plot(test_dates, y_test.values, label='Actual Views', linewidth=2, marker='o', markersize=4)
ax1.plot(test_dates, predictions, label='Predicted Views', linewidth=2, 
         marker='x', markersize=4, linestyle='--', alpha=0.8)
ax1.fill_between(test_dates, 
                 predictions - metrics['RMSE'],
                 predictions + metrics['RMSE'],
                 alpha=0.2, label='±RMSE')
ax1.set_title('YouTube Views Forecast', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Views')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# 2. Full timeline
ax2 = axes[0, 1]
ax2.plot(data['date'], data['views'], label='Historical', linewidth=1, alpha=0.7)
ax2.plot(test_dates, predictions, label='Forecast', linewidth=2, linestyle='--', color='red')
ax2.set_title('Complete Timeline', fontsize=14, fontweight='bold')
ax2.set_xlabel('Date')
ax2.set_ylabel('Views')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

# 3. Error distribution
ax3 = axes[1, 0]
errors = y_test.values - predictions
ax3.hist(errors, bins=20, edgecolor='black', alpha=0.7, color='coral')
ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax3.set_title('Prediction Error Distribution', fontsize=14, fontweight='bold')
ax3.set_xlabel('Error')
ax3.set_ylabel('Frequency')
ax3.grid(True, alpha=0.3)

# 4. Feature importance
ax4 = axes[1, 1]
importance_df = model.get_feature_importance()
if importance_df is not None:
    top_10 = importance_df.head(10)
    ax4.barh(range(len(top_10)), top_10['importance'].values, color='green', alpha=0.7)
    ax4.set_yticks(range(len(top_10)))
    ax4.set_yticklabels(top_10['feature'].values, fontsize=9)
    ax4.invert_yaxis()
    ax4.set_title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Importance')
    ax4.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('youtube_forecast_results.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualization saved to 'youtube_forecast_results.png'")

# ============================================================
# STEP 5: FUTURE PREDICTIONS
# ============================================================
print("\n" + "="*80)
print("STEP 5: Predicting Future Views")
print("="*80)

# Predict next 30 days
print("\n  Generating 30-day forecast...")

future_dates = pd.date_range(data['date'].max() + timedelta(days=1), periods=30, freq='D')
last_known_views = data['views'].iloc[-1]

# Simple approach: Use last known values and create features
# In production, you'd want a more sophisticated approach
future_predictions = []

for i in range(30):
    # Use growth rate from recent data
    recent_growth = (data['views'].iloc[-7:].diff().mean())
    next_prediction = last_known_views + recent_growth * (i + 1)
    future_predictions.append(max(0, int(next_prediction)))

future_df = pd.DataFrame({
    'date': future_dates,
    'predicted_views': future_predictions
})

print(f"\n  30-Day Forecast:")
print(f"    Start date: {future_df['date'].min().date()}")
print(f"    End date: {future_df['date'].max().date()}")
print(f"    Predicted views range: {future_df['predicted_views'].min():,} to {future_df['predicted_views'].max():,}")
print(f"\n  First 7 days:")
print(future_df.head(7).to_string(index=False))

# ============================================================
# STEP 6: EXPORT RESULTS
# ============================================================
print("\n" + "="*80)
print("STEP 6: Exporting Results")
print("="*80)

# Save forecast
forecast_results = pd.DataFrame({
    'date': test_dates,
    'actual_views': y_test.values,
    'predicted_views': predictions.astype(int),
    'error': (y_test.values - predictions).astype(int)
})

forecast_results.to_csv('youtube_forecast.csv', index=False)
print("\n✓ Forecast saved to 'youtube_forecast.csv'")

# Save future predictions
future_df.to_csv('youtube_future_forecast.csv', index=False)
print("✓ Future forecast saved to 'youtube_future_forecast.csv'")

# Save summary
with open('youtube_forecast_summary.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("YouTube Video Forecast - Summary\n")
    f.write("="*80 + "\n\n")
    
    f.write("Data Information:\n")
    f.write("-"*80 + "\n")
    f.write(f"  Date range: {data['date'].min().date()} to {data['date'].max().date()}\n")
    f.write(f"  Total days: {len(data)}\n")
    f.write(f"  Total views: {data['views'].sum():,}\n")
    f.write(f"  Average daily views: {data['views'].mean():.0f}\n\n")
    
    f.write("Model Performance:\n")
    f.write("-"*80 + "\n")
    f.write(f"  RMSE: {metrics['RMSE']:.2f}\n")
    f.write(f"  MAE: {metrics['MAE']:.2f}\n")
    f.write(f"  R²: {metrics['R²']:.4f}\n")
    f.write(f"  MAPE: {metrics['MAPE']:.2f}%\n\n")
    
    f.write("30-Day Forecast:\n")
    f.write("-"*80 + "\n")
    f.write(f"  Forecast period: {future_df['date'].min().date()} to {future_df['date'].max().date()}\n")
    f.write(f"  Expected views range: {future_df['predicted_views'].min():,} to {future_df['predicted_views'].max():,}\n")

print("✓ Summary saved to 'youtube_forecast_summary.txt'")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*80)
print("YouTube Forecast Complete! 🎉")
print("="*80)

print(f"\n  📊 Analyzed {len(data)} days of video data")
print(f"  🤖 Trained XGBoost model (R² = {metrics['R²']:.4f})")
print(f"  📈 Generated 30-day forecast")
print(f"  💾 Exported 3 files:")
print(f"     • youtube_forecast_results.png")
print(f"     • youtube_forecast.csv")
print(f"     • youtube_future_forecast.csv")
print(f"     • youtube_forecast_summary.txt")

print("\n" + "="*80 + "\n")