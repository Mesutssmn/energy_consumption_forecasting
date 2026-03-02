# ⚡ PJM AEP Energy Consumption Forecasting

<div align="center">

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://energy-consumption-forecasting-1.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-FF6F00?logo=tensorflow&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**An interactive time-series forecasting app for hourly energy consumption in the PJM AEP region.**  
Combines statistical models, deep learning, and gradient boosting in a single Streamlit dashboard.

🔗 **Live Demo → [energy-consumption-forecasting-1.streamlit.app](https://energy-consumption-forecasting-1.streamlit.app/)**

</div>

---

## 📸 App Tabs

| Tab | Description |
|---|---|
| **⚡ Pre-trained Best Model** | Instant forecasts from saved LSTM + XGBoost — no retraining needed |
| **📊 Exploratory Analysis** | Full series view, year/week zooms, hour/day/month seasonality, STL decomposition |
| **🔬 Stationarity Tests** | ADF & KPSS test results, ACF/PACF plots with confidence bands |
| **🤖 Model Training** | Train SARIMAX, LSTM, XGBoost with tunable sidebar hyperparameters |
| **🏆 Model Comparison** | RMSE/MAE/MAPE bar charts, metrics table, multi-model forecast overlay |

---

## 🤖 Models & Results

> **Test set:** Jan 2018 – Aug 2018 · **Training data:** Jan 2004 – Dec 2017

| Model | RMSE | MAE | MAPE | Notes |
|---|---|---|---|---|
| **LSTM** 🏆 | ~780 MW | ~605 MW | ~3.9% | 2-layer LSTM, 60-day lookback, early stopping |
| **XGBoost** | ~816 MW | ~633 MW | ~4.1% | Calendar + lag + rolling features, `hist` method |
| **SARIMAX** | ~5200 MW | ~4900 MW | ~34% | SARIMA(2,1,1)(1,1,1,7), daily aggregation |

### What the metrics mean

| Metric | Formula | Meaning |
|---|---|---|
| **RMSE** | √(mean(actual − forecast)²) | Average error in MW; penalises large mistakes |
| **MAE** | mean(\|actual − forecast\|) | Average absolute error in MW |
| **MAPE** | mean(\|actual − forecast\| / actual) × 100 | Error as a percentage — easy to interpret |

---

## 🔬 Stationarity Tests — What & Why

Before fitting any model, we check whether the series is **stationary** (constant mean/variance over time). Non-stationary data causes spurious model fits.

### ADF Test (Augmented Dickey-Fuller)

Tests whether the series has a **unit root** (non-stationary).

| Hypothesis | Meaning |
|---|---|
| H₀ (null) | Series **has** a unit root → non-stationary |
| H₁ (alternative) | Series **is** stationary |

- **p < 0.05** → reject H₀ → ✅ stationary  
- **p ≥ 0.05** → fail to reject H₀ → ❌ non-stationary

### KPSS Test (Kwiatkowski–Phillips–Schmidt–Shin)

Tests the **opposite** — directly checks for stationarity.

| Hypothesis | Meaning |
|---|---|
| H₀ (null) | Series **is** stationary |
| H₁ (alternative) | Series is non-stationary |

- **p > 0.05** → fail to reject H₀ → ✅ stationary  
- **p ≤ 0.05** → reject H₀ → ❌ non-stationary

> **Why use both?** They have opposite nulls — so they complement each other. If both agree you have high confidence. If they disagree, the series may need fractional differencing.

### ACF — Autocorrelation Function

Shows **how correlated the series is with its own past values** at each lag k:

```
ACF(k) = correlation( Yₜ, Yₜ₋ₖ )
```

- Slow decay → non-stationary series
- Spikes at lag 7, 24, 168 → weekly and daily seasonality  
- Used to select the **MA(q)** order in ARIMA

### PACF — Partial Autocorrelation Function

Shows the **"pure" effect** of each lag after removing the influence of all shorter lags:

```
PACF(k) = correlation( Yₜ, Yₜ₋ₖ | Yₜ₋₁, …, Yₜ₋ₖ₊₁ )
```

- Sharp cutoff at lag p → suggests AR(p) model  
- Used to select the **AR(p)** order in ARIMA

### ARIMA Order Selection Cheat Sheet

| ACF | PACF | Suggested model |
|---|---|---|
| Cuts off at lag q | Decays slowly | MA(q) |
| Decays slowly | Cuts off at lag p | AR(p) |
| Both decay slowly | Both decay slowly | ARMA(p,q) |

---

## 🏗️ Project Structure

```
├── app.py                   # Streamlit application (5 tabs)
├── train_best_model.py      # Offline training script (CLI + interactive picker)
├── requirements.txt
├── README.md
├── models/                  # Pre-trained artefacts (included in repo)
│   ├── lstm_model.keras
│   ├── xgb_model.json
│   ├── scaler.pkl
│   ├── lstm_pred.npy
│   ├── xgb_pred.npy
│   └── metadata.json
└── archive/
    └── AEP_hourly.csv       # PJM AEP hourly energy data (2004–2018)
```

---

## 🚀 Run Locally

```bash
# 1. Clone & install
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd pjm-energy-forecasting
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt

# 2. (Optional) Retrain models
python train_best_model.py    # interactive menu, or:
python train_best_model.py --models lstm xgb   # CLI

# 3. Launch
streamlit run app.py
```

### Training script options

```bash
python train_best_model.py --models lstm          # LSTM only
python train_best_model.py --models xgb           # XGBoost only
python train_best_model.py --models lstm xgb      # both ✅ recommended
python train_best_model.py --models all           # all three
```

---

## ⚙️ XGBoost Feature Engineering

| Group | Features |
|---|---|
| **Calendar** | `dayofweek`, `month`, `quarter`, `year`, `dayofyear`, `is_weekend`, `weekofyear` |
| **Lag features** | `lag_1`, `lag_7`, `lag_14`, `lag_30`, `lag_365` |
| **Rolling 7d & 30d** | `mean`, `std`, `min`, `max` — shifted by 1 day (no data leakage) |

---

## 🛠️ Tech Stack

| Layer | Libraries |
|---|---|
| **App** | `streamlit`, `plotly` |
| **Deep Learning** | `tensorflow` / `keras` |
| **Gradient Boosting** | `xgboost` |
| **Statistical** | `statsmodels` (SARIMAX, ADF, KPSS, ACF/PACF) |
| **Data** | `pandas`, `numpy`, `scikit-learn`, `joblib` |

---

## 📦 Data Source

[PJM Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) — Rob Mulla on Kaggle.  
Download and place `AEP_hourly.csv` in `archive/`.

---

## 📄 License

MIT © 2024
