# 🚀 ChronoCast Quick Start Guide

## Week 1 Complete: Day 1-7 ✅

This guide will help you get started with ChronoCast after completing Week 1 of development.

---

## 📥 Installation

### 1. Clone and Setup

```bash
# Navigate to your project directory
cd chronocast

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install ChronoCast in development mode
pip install -e .
```

### 2. Verify Installation

```bash
# Run tests
pytest tests/ -v

# Should see all tests passing ✓
```

---

## 🎯 Quick Example

### Complete Pipeline in 10 Lines

```python
from chronocast import ChronoModel, create_all_features, evaluate_model
import pandas as pd

# 1. Load your data
df = pd.read_csv('your_data.csv')

# 2. Create features
df_featured = create_all_features(df, date_col='date', target_col='views')

# 3. Split data
train = df_featured[:int(len(df_featured)*0.8)]
test = df_featured[int(len(df_featured)*0.8):]

feature_cols = [c for c in df_featured.columns if c not in ['date', 'views']]
X_train, y_train = train[feature_cols], train['views']
X_test, y_test = test[feature_cols], test['views']

# 4. Train model
model = ChronoModel('xgb')
model.fit(X_train, y_train)

# 5. Evaluate
predictions = model.predict(X_test)
metrics = evaluate_model(y_test, predictions)
print(metrics)
```

---

## 📚 Run Example Scripts

### 1. Feature Engineering Demo

```bash
cd chronocast/core
python feature_engineering.py
```

**What it does:** Demonstrates time features, lag features, and rolling statistics.

### 2. Model Wrapper Demo

```bash
cd examples
python model_wrapper_demo.py
```

**What it does:** 
- Trains 5 different models
- Compares performance
- Shows feature importance
- Saves best model

### 3. Complete Pipeline Demo

```bash
cd examples
python complete_pipeline.py
```

**What it does:**
- End-to-end workflow
- Creates realistic blog data
- Trains multiple models
- Generates visualizations
- Saves reports

**Expected output:**
- Model comparison report (JSON)
- Saved model (.pkl)
- 4 visualization plots (PNG)

---

## 🧪 Testing

### Run All Tests

```bash
# From project root
pytest tests/ -v
```

### Run Specific Test Module

```bash
# Test feature engineering
pytest tests/test_feature_engineering.py -v

# Test model wrapper
pytest tests/test_model_wrapper.py -v

# Test evaluation
pytest tests/test_evaluation.py -v
```

### Check Test Coverage

```bash
pytest tests/ --cov=chronocast --cov-report=html
```

---

## 📂 Project Structure Overview

```
chronocast/
├── chronocast/              # Main package
│   ├── __init__.py         # Package initialization
│   ├── core/               # Core modules
│   │   ├── feature_engineering.py  # ✅ Day 2-3
│   │   ├── model_wrapper.py        # ✅ Day 4-5
│   │   └── evaluation.py           # ✅ Day 6-7
│   └── utils/              # Utility modules (Week 2)
│       ├── logger.py
│       └── data_loader.py
├── examples/               # Example scripts
│   ├── model_wrapper_demo.py
│   └── complete_pipeline.py
├── tests/                  # Unit tests
│   ├── test_feature_engineering.py
│   ├── test_model_wrapper.py
│   └── test_evaluation.py
├── docs/                   # Documentation (Week 4)
├── setup.py               # Package setup
├── requirements.txt       # Dependencies
└── README.md             # Project README
```

---

## 🔑 Key Features Implemented (Week 1)

### ✅ Feature Engineering
- Time-based features (day, month, weekday, cyclical encoding)
- Lag features (1, 7, 14, 30 days)
- Rolling statistics (mean, std, min, max)
- Categorical encoding (one-hot)

### ✅ Model Wrapper
- 7 model types: Linear, Ridge, Lasso, RF, DT, GBM, XGBoost
- Unified interface: fit, predict, save, load
- Feature importance extraction
- Training history tracking
- Custom model registration

### ✅ Evaluation
- 9 metrics: MSE, RMSE, MAE, MAPE, R², SMAPE, MASE, Bias, % Bias
- Model comparison tools
- Visualization functions (forecast, residuals, metrics)
- Report generation (JSON)

---

## 🎓 Usage Patterns

### Pattern 1: Quick Forecast

```python
from chronocast import ChronoModel
import pandas as pd

# Load and prepare data
df = pd.read_csv('data.csv')
X_train, y_train = df[features], df['target']

# Train and predict
model = ChronoModel('xgb', n_estimators=100)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Pattern 2: Model Comparison

```python
from chronocast.core.evaluation import compare_models

models_data = {
    'XGBoost': {'y_true': y_test, 'y_pred': pred_xgb},
    'Random Forest': {'y_true': y_test, 'y_pred': pred_rf}
}

comparison = compare_models(models_data, metric='RMSE')
print(comparison)
```

### Pattern 3: Full Pipeline

See `examples/complete_pipeline.py` for comprehensive example.

---

## 🐛 Troubleshooting

### Import Errors

```bash
# Make sure ChronoCast is installed
pip install -e .

# Verify installation
python -c "import chronocast; print(chronocast.__version__)"
```

### Test Failures

```bash
# Clear cache and rerun
pytest --cache-clear tests/ -v

# Run specific failing test
pytest tests/test_feature_engineering.py::test_name -v
```

### Missing Dependencies

```bash
# Reinstall all requirements
pip install -r requirements.txt --upgrade
```

---

## 📊 Next Steps (Week 2)

After completing Week 1, you'll add:

1. **Explainability Module** (Day 8-9)
   - SHAP integration
   - Feature importance plots
   - Explainability logs

2. **Visualization Utilities** (Day 10-11)
   - Interactive Plotly charts
   - Dashboard-ready visualizations

3. **Logging System** (Day 12)
   - Training logs
   - Hyperparameter tracking
   - JSON logging

4. **End-to-End Integration** (Day 13-14)
   - Complete workflow validation
   - Baseline comparisons (Prophet, ARIMA)

---

## 💡 Tips

1. **Start Simple**: Begin with Linear or Ridge models before trying ensemble methods
2. **Feature Selection**: Not all created features may be useful - use feature importance
3. **Cross-Validation**: For production, consider time-series cross-validation
4. **Save Models**: Always save your trained models with descriptive names
5. **Monitor Metrics**: Track multiple metrics, not just RMSE

---

## 📞 Support

- Check `examples/` directory for working code
- Run tests to verify your setup
- See `README.md` for architecture details

---

**Week 1 Status: Complete! ✅**

Ready to move to Week 2? 🚀