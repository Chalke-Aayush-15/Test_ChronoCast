# 📘 ChronoCast Complete Guide

## Comprehensive Documentation for Time Series Forecasting Platform

---

## 📑 Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Library](#core-library)
5. [Backend API](#backend-api)
6. [Frontend Dashboard](#frontend-dashboard)
7. [Demo Project](#demo-project)
8. [Deployment](#deployment)
9. [API Reference](#api-reference)
10. [Troubleshooting](#troubleshooting)

---

## 🎯 Introduction

ChronoCast is a complete time series forecasting platform featuring:

- **Python Library** - 7 ML algorithms with explainability
- **REST API** - Django backend with PostgreSQL
- **Web Dashboard** - React frontend with interactive charts
- **Full Transparency** - SHAP-based explainability

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    React Frontend                        │
│              (Interactive Dashboard)                     │
└────────────────────┬────────────────────────────────────┘
                     │ REST API
┌────────────────────┴────────────────────────────────────┐
│                  Django Backend                          │
│            (API + Database + Storage)                    │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────┐
│                ChronoCast Library                        │
│     (Feature Engineering + Models + Explainability)     │
└─────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### Prerequisites

- Python 3.8+
- Node.js 16+
- PostgreSQL 12+ (optional, can use SQLite)

### 1. Install Python Library

```bash
# Clone repository
git clone https://github.com/yourusername/chronocast.git
cd chronocast

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install library
pip install -e .

# Or install from PyPI (when published)
pip install chronocast
```

### 2. Setup Backend

```bash
cd backend
chmod +x setup.sh
./setup.sh

# Or manual setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py createsuperuser
```

### 3. Setup Frontend

```bash
cd frontend
npm install
```

### 4. Start Services

```bash
# Terminal 1: Backend
cd backend
source venv/bin/activate
python manage.py runserver

# Terminal 2: Frontend
cd frontend
npm run dev
```

**Access:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000/api
- API Docs: http://localhost:8000/swagger
- Admin: http://localhost:8000/admin

---

## 🚀 Quick Start

### Python Library

```python
from chronocast import ChronoModel, create_all_features, evaluate_model
import pandas as pd

# Load data
df = pd.read_csv('your_data.csv')

# Create features
df_featured = create_all_features(
    df, 
    date_col='date', 
    target_col='views',
    lags=[1, 7, 14],
    windows=[7, 14]
)

# Split data
train = df_featured[:int(len(df_featured)*0.8)]
test = df_featured[int(len(df_featured)*0.8):]

# Train model
model = ChronoModel('xgb', n_estimators=100)
model.fit(train[features], train['views'])

# Predict & evaluate
predictions = model.predict(test[features])
metrics = evaluate_model(test['views'], predictions)
print(metrics)
```

### Web Dashboard

1. **Upload Dataset**
   - Go to http://localhost:3000/upload
   - Drag & drop CSV file
   - Select date and target columns

2. **Create Forecast**
   - Select dataset
   - Choose model (XGBoost recommended)
   - Configure parameters
   - Click "Start Forecast"

3. **View Results**
   - Interactive charts
   - Evaluation metrics
   - Feature importance
   - Generate SHAP analysis

4. **Compare Models**
   - Select multiple forecasts
   - View side-by-side comparison
   - Export results

---

## 📚 Core Library

### Feature Engineering

```python
from chronocast import (
    create_time_features,
    create_lag_features,
    create_rolling_features,
    create_all_features
)

# Time features
df = create_time_features(df, 'date')
# Adds: year, month, day, day_of_week, is_weekend, etc.

# Lag features
df = create_lag_features(df, 'views', lags=[1, 7, 14])
# Adds: views_lag_1, views_lag_7, views_lag_14

# Rolling features
df = create_rolling_features(df, 'views', windows=[7, 14])
# Adds: rolling_mean_7, rolling_std_7, etc.

# All features at once
df = create_all_features(df, 'date', 'views', lags=[1,7], windows=[7])
```

### Model Training

```python
from chronocast import ChronoModel

# Available models
models = ['linear', 'ridge', 'lasso', 'rf', 'dt', 'gbm', 'xgb']

# Initialize model
model = ChronoModel('xgb', n_estimators=100, max_depth=5)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Get feature importance
importance = model.get_feature_importance()

# Save/load model
model.save('model.pkl')
loaded_model = ChronoModel.load('model.pkl')
```

### Evaluation

```python
from chronocast import evaluate_model, compare_models

# Single model evaluation
metrics = evaluate_model(y_true, y_pred, y_train)
# Returns: RMSE, MAE, MAPE, R², SMAPE, Bias, MASE

# Multiple model comparison
models_dict = {
    'XGBoost': {'y_true': y_test, 'y_pred': pred_xgb},
    'Random Forest': {'y_true': y_test, 'y_pred': pred_rf}
}
comparison = compare_models(models_dict)
print(comparison)
```

### Explainability

```python
from chronocast import ModelExplainer

# Initialize explainer
explainer = ModelExplainer(model, X_train, feature_names)

# Feature importance
explainer.plot_feature_importance(top_n=15)

# SHAP analysis
explainer.plot_shap_summary(X_test, max_samples=50)
explainer.plot_shap_waterfall(X_test, instance_idx=0)

# Get contributions
contributions = explainer.get_feature_contributions(X_test, instance_idx=0)

# Save log
explainer.save_explainability_log(X_test, 'explainability.json')
```

### Visualization

```python
from chronocast import InteractiveVisualizer

viz = InteractiveVisualizer()

# Forecast plot
viz.plot_forecast(y_test, predictions, dates, save_html='forecast.html')

# Residual analysis
viz.plot_residuals(y_test, predictions, save_html='residuals.html')

# Complete dashboard
viz.create_dashboard(
    y_test, predictions, dates,
    metrics=metrics,
    importance_df=importance_df,
    save_html='dashboard.html'
)
```

### Logging

```python
from chronocast import ChronoLogger, ExperimentTracker

# Initialize logger
logger = ChronoLogger(log_dir='logs')

# Log training
run_id = logger.start_training('XGBoost', params)
logger.log_training_data(n_samples, n_features)
logger.end_training(training_time, metrics)

# Experiment tracking
tracker = ExperimentTracker('experiment_name')
tracker.log_experiment('XGBoost', params, metrics)
best = tracker.get_best_experiment('RMSE')
```

---

## 🔌 Backend API

### Datasets

```bash
# Upload dataset
POST /api/datasets/
Content-Type: multipart/form-data
{
  "name": "Dataset Name",
  "file": <file>
}

# List datasets
GET /api/datasets/

# Validate dataset
POST /api/datasets/{id}/validate/
{
  "date_column": "date",
  "target_column": "views"
}

# Preview dataset
GET /api/datasets/{id}/preview/?n_rows=10
```

### Forecast Runs

```bash
# Create forecast
POST /api/forecast-runs/
{
  "dataset": "uuid",
  "model_type": "xgb",
  "model_params": {"n_estimators": 100},
  "lag_periods": [1, 7, 14],
  "rolling_windows": [7, 14]
}

# Get forecast status
GET /api/forecast-runs/{id}/

# Get metrics
GET /api/forecast-runs/{id}/metrics/

# Get predictions
GET /api/forecast-runs/{id}/predictions/?page=1&page_size=100

# Generate explainability
POST /api/forecast-runs/{id}/generate_explainability/
{
  "max_samples": 50
}
```

### Model Comparison

```bash
# Create comparison
POST /api/comparisons/create_comparison/
{
  "dataset_id": "uuid",
  "forecast_run_ids": ["uuid1", "uuid2"],
  "name": "Comparison Name"
}

# Get comparison data
GET /api/comparisons/{id}/chart_data/
```

---

## 🎨 Frontend Dashboard

### Components

**Pages:**
- HomePage - Landing page
- UploadPage - Dataset upload
- ForecastPage - Model configuration
- ResultsPage - Results dashboard
- ComparePage - Model comparison

**Features:**
- Drag & drop file upload
- Real-time progress tracking
- Interactive Plotly charts
- Responsive design
- Error handling

### Customization

```javascript
// src/services/api.js
const API_BASE_URL = process.env.VITE_API_URL || 'http://localhost:8000/api';

// tailwind.config.js
theme: {
  extend: {
    colors: {
      primary: { /* your colors */ }
    }
  }
}
```

---

## 🎬 Demo Project

Run the complete demo:

```bash
cd demo
python blog_forecast_demo.py
```

**Generates:**
- Realistic blog data (18 months)
- 5 trained models
- Comprehensive visualizations
- SHAP analysis
- Complete report

**Output files:**
- `blog_data.csv` - Dataset
- `01_exploratory_analysis.png` - EDA
- `02_complete_forecast_analysis.png` - Results
- `03_feature_importance.png` - Feature importance
- `04_shap_summary.png` - SHAP analysis
- `05_shap_waterfall.png` - SHAP waterfall
- `DEMO_SUMMARY.txt` - Summary report

---

## 🚀 Deployment

### Backend (Django)

**Option 1: Heroku**
```bash
# Install Heroku CLI
heroku create chronocast-api
heroku addons:create heroku-postgresql:hobby-dev
git push heroku main
heroku run python manage.py migrate
```

**Option 2: AWS EC2**
```bash
# Install dependencies
sudo apt-get update
sudo apt-get install python3-pip postgresql nginx

# Setup application
git clone your-repo
cd backend
pip install -r requirements.txt
gunicorn chronocast_api.wsgi:application

# Configure nginx
sudo nano /etc/nginx/sites-available/chronocast
```

**Option 3: Docker**
```dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "chronocast_api.wsgi:application", "--bind", "0.0.0.0:8000"]
```

### Frontend (React)

**Option 1: Vercel**
```bash
npm install -g vercel
vercel
```

**Option 2: Netlify**
```bash
npm run build
netlify deploy --prod --dir=dist
```

**Option 3: AWS S3 + CloudFront**
```bash
npm run build
aws s3 sync dist/ s3://your-bucket
aws cloudfront create-invalidation --distribution-id XXX --paths "/*"
```

### Environment Variables

**Backend (.env)**
```env
SECRET_KEY=your-production-secret-key
DEBUG=False
ALLOWED_HOSTS=yourdomain.com
DATABASE_URL=postgresql://user:pass@host:5432/db
```

**Frontend (.env.production)**
```env
VITE_API_URL=https://api.yourdomain.com/api
```

---

## 📖 API Reference

### Python Library

**ChronoModel**
- `__init__(model_type, **params)` - Initialize model
- `fit(X, y)` - Train model
- `predict(X)` - Make predictions
- `get_feature_importance()` - Get feature importance
- `save(path)` - Save model
- `load(path)` - Load model (classmethod)

**Feature Engineering**
- `create_time_features(df, date_col)` - Time features
- `create_lag_features(df, target_col, lags)` - Lag features
- `create_rolling_features(df, target_col, windows)` - Rolling features
- `create_all_features(df, date_col, target_col, ...)` - All features

**Evaluation**
- `evaluate_model(y_true, y_pred, y_train)` - Calculate metrics
- `compare_models(models_dict, metric, sort_ascending)` - Compare models

**Explainability**
- `ModelExplainer(model, X_train, feature_names)` - Initialize
- `calculate_shap_values(X_test, max_samples)` - SHAP values
- `plot_feature_importance(top_n)` - Feature importance plot
- `plot_shap_summary(X_test)` - SHAP summary plot

### REST API Endpoints

**Complete endpoint list:**
- POST `/api/datasets/` - Upload dataset
- GET `/api/datasets/` - List datasets
- POST `/api/datasets/{id}/validate/` - Validate
- POST `/api/forecast-runs/` - Create forecast
- GET `/api/forecast-runs/{id}/` - Get forecast
- GET `/api/forecast-runs/{id}/metrics/` - Get metrics
- POST `/api/comparisons/create_comparison/` - Compare models

Full API documentation: http://localhost:8000/swagger/

---

## 🐛 Troubleshooting

### Common Issues

**1. Import Error: ChronoCast**
```bash
# Solution: Install in editable mode
pip install -e .
```

**2. Database Connection Error**
```bash
# Solution: Use SQLite for development
echo "USE_SQLITE=True" >> backend/.env
```

**3. CORS Error**
```python
# Solution: Update Django settings.py
CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
]
```

**4. SHAP Installation Error**
```bash
# Solution: Install with specific version
pip install shap==0.41.0
```

**5. Frontend Build Error**
```bash
# Solution: Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

### Performance Tips

1. **Large Datasets**: Use sampling for SHAP analysis
2. **Slow Training**: Use fewer estimators or simpler models
3. **Memory Issues**: Reduce batch sizes or use pagination
4. **API Timeouts**: Use Celery for async tasks

---

## 📊 Best Practices

### Model Selection
- Start with XGBoost for best performance
- Use Linear/Ridge for interpretability
- Try Random Forest for robustness

### Feature Engineering
- Always include lag features (1, 7, 14 days)
- Use rolling windows (7, 14, 30 days)
- Enable time features for seasonality

### Evaluation
- Use multiple metrics (RMSE, MAE, R²)
- Check residual plots
- Validate on holdout set

### Deployment
- Use environment variables
- Enable logging
- Set up monitoring
- Regular backups

---

## 🎓 Resources

- **Documentation**: `/docs`
- **Examples**: `/examples`
- **Demo**: `/demo`
- **Tests**: `/tests`
- **API Docs**: http://localhost:8000/swagger/

---

## 📝 License

MIT License - See LICENSE file

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Add tests
5. Submit pull request

---

## 📞 Support

- GitHub Issues: https://github.com/yourusername/chronocast/issues
- Email: support@chronocast.io
- Documentation: https://docs.chronocast.io

---

**ChronoCast v0.1.0 - Complete Documentation**

Built with ❤️ for transparent time series forecasting