# ⚡ PJM AEP Energy Consumption Forecasting

A Streamlit application for time-series forecasting of hourly energy consumption
from the PJM AEP region. Built with SARIMAX, LSTM, and XGBoost models.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## 📊 Features

| Tab | Description |
|---|---|
| ⚡ Pre-trained Best Model | Instant predictions from saved models — no retraining needed |
| 📊 Exploratory Analysis | Full series, year/week zooms, seasonality patterns, decomposition |
| 🔬 Stationarity Tests | ADF & KPSS tests, ACF/PACF plots |
| 🤖 Model Training | Interactive training for SARIMAX, LSTM, XGBoost with sidebar hyperparams |
| 🏆 Model Comparison | RMSE/MAE/MAPE bar charts, metrics table, forecast overlay |

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Mesutssmn/energy_consumption_forecasting.git
cd pjm-energy-forecasting

# 2. Create virtual environment and install deps
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 3. (Optional) Train and save best models
python train_best_model.py    # interactive model picker

# 4. Launch the app
streamlit run app.py
```

## 🗂️ Project Structure

```
├── app.py                  # Streamlit application
├── train_best_model.py     # One-time training script (CLI + interactive)
├── requirements.txt
├── models/                 # Saved model files (pre-trained)
│   ├── lstm_model.keras
│   ├── xgb_model.json
│   ├── scaler.pkl
│   ├── lstm_pred.npy
│   ├── xgb_pred.npy
│   └── metadata.json
└── archive/
    └── AEP_hourly.csv      # PJM AEP hourly energy data
```

## 🤖 Models

| Model | RMSE (MW) | MAPE | Notes |
|---|---|---|---|
| LSTM | ~780 MW | ~3.9% | 2-layer LSTM, lookback 60 days |
| XGBoost | ~816 MW | ~4.1% | Calendar + lag + rolling features |
| SARIMAX | ~5200 MW | ~34% | Daily aggregation, (2,1,1)(1,1,1,7) |

## 📦 Training Script Usage

```bash
python train_best_model.py                      # interactive menu
python train_best_model.py --models lstm        # LSTM only
python train_best_model.py --models xgb         # XGBoost only
python train_best_model.py --models lstm xgb    # both (recommended)
python train_best_model.py --models all         # all three
```

## 📄 Data

[PJM Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)
by Rob Mulla on Kaggle. Download and place `AEP_hourly.csv` in `archive/`.

## 🛠️ Tech Stack

`Python` · `Streamlit` · `TensorFlow/Keras` · `XGBoost` · `statsmodels` · `Plotly` · `pandas`
