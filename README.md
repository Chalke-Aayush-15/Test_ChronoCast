# 🗓️ ChronoCast

**A Transparent and Modular Time Series Forecasting Library**

## 🎯 Project Goals

ChronoCast is designed to make time series forecasting:
- **Transparent**: Full explainability with SHAP values and feature importance
- **Modular**: Easy to swap models and customize pipelines
- **Practical**: Built for real-world business forecasting needs

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│              ChronoCast Library                      │
├─────────────────────────────────────────────────────┤
│  Feature Engineering → Model Wrapper → Evaluation   │
│           ↓                                          │
│  Explainability ← Visualization ← Logging           │
└─────────────────────────────────────────────────────┘
```

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/chalke-Aayush-15/chronocast.git
cd chronocast

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## 🚀 Quick Start

```python
from chronocast import ChronoModel, create_time_features, evaluate_model
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# Create time features
df = create_time_features(df, date_col='date')

# Train model
model = ChronoModel(model_type='xgb')
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate
metrics = evaluate_model(y_test, predictions)
print(metrics)
```

## 📚 Features

- ✅ Multiple ML models (Linear, Random Forest, XGBoost)
- ✅ Automated feature engineering for time series
- ✅ Model comparison and evaluation
- ✅ SHAP-based explainability
- ✅ Interactive visualizations
- ✅ Comprehensive logging

## 🗂️ Project Structure

```
chronocast/
├── chronocast/
│   ├── __init__.py
│   ├── core/
│   │   ├── feature_engineering.py
│   │   ├── model_wrapper.py
│   │   ├── evaluation.py
│   │   └── explainability.py
│   └── utils/
│       ├── logger.py
│       └── data_loader.py
├── examples/
├── tests/
├── docs/
├── setup.py
└── README.md
```

## 📈 Roadmap

- [x] Week 1: Core library foundation
- [ ] Week 2: Explainability & visualization
- [ ] Week 3: Django + React dashboard
- [ ] Week 4: Documentation & deployment

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

Aayush Chalke - [GitHub](https://github.com/chalke-Aayush-15/)