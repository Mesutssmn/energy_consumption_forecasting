"""
train_best_model.py
───────────────────
One-time training script: train one or more models and save them to models/.

Usage examples:
    python train_best_model.py                      # interactive model picker
    python train_best_model.py --models lstm
    python train_best_model.py --models xgb
    python train_best_model.py --models lstm xgb
    python train_best_model.py --models lstm xgb sarimax
    python train_best_model.py --models all

Outputs in models/:
    lstm_model.keras      — LSTM weights
    xgb_model.json        — XGBoost booster
    sarimax_result.pkl    — SARIMAX fitted result
    scaler.pkl            — MinMaxScaler (LSTM)
    metadata.json         — params, metrics, best model
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ── Model selection ───────────────────────────────────────────────────────────
AVAILABLE = ["lstm", "xgb", "sarimax"]

parser = argparse.ArgumentParser(description="Train PJM AEP forecasting models")
parser.add_argument(
    "--models", nargs="+",
    choices=AVAILABLE + ["all"],
    metavar="MODEL",
    help=f"Models to train: {AVAILABLE} or 'all' (default: interactive)",
)
args = parser.parse_args()

if args.models is None:
    # Interactive picker when no CLI args given
    print("\n┌─────────────────────────────────────────────┐")
    print("\u2502  ⚡ PJM AEP Model Trainer                   \u2502")
    print("\u251c─────────────────────────────────────────────┤")
    print("\u2502  [1] LSTM only                              \u2502")
    print("\u2502  [2] XGBoost only                           \u2502")
    print("\u2502  [3] SARIMAX only                           \u2502")
    print("\u2502  [4] LSTM + XGBoost          (recommended)  \u2502")
    print("\u2502  [5] LSTM + XGBoost + SARIMAX               \u2502")
    print("\u2502  [6] All models                             \u2502")
    print("\u2514─────────────────────────────────────────────┘")
    choice_map = {
        "1": ["lstm"],
        "2": ["xgb"],
        "3": ["sarimax"],
        "4": ["lstm", "xgb"],
        "5": ["lstm", "xgb", "sarimax"],
        "6": AVAILABLE,
    }
    while True:
        c = input("\nEnter choice [1-6]: ").strip()
        if c in choice_map:
            selected = choice_map[c]
            break
        print("  Invalid choice, try again.")
elif "all" in args.models:
    selected = AVAILABLE
else:
    selected = list(dict.fromkeys(args.models))   # dedup, preserve order

print(f"\n🔧 Models to train: {', '.join(m.upper() for m in selected)}\n")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH  = "archive/AEP_hourly.csv"
SPLIT_DATE = "2017-12-31"
LOOKBACK   = 60
LSTM_UNITS = 64
EPOCHS     = 100
MODELS_DIR = "models"
RANDOM_SEED = 42
os.makedirs(MODELS_DIR, exist_ok=True)

# XGBoost config
XGB = dict(
    n_estimators        = 1500,       # more trees (early stopping prevents overfit)
    learning_rate       = 0.03,       # smaller lr + more trees = better generalisation
    max_depth           = 5,          # shallower trees → less overfit for tabular time-series
    min_child_weight    = 5,          # higher = more conservative splits
    subsample           = 0.75,       # row sampling per tree
    colsample_bytree    = 0.75,       # feature sampling per tree
    colsample_bylevel   = 0.8,        # feature sampling per depth level
    colsample_bynode    = 0.8,        # feature sampling per split node
    gamma               = 0.1,        # min loss reduction to make a split (regularisation)
    reg_alpha           = 0.2,        # L1 regularisation (sparse feature weights)
    reg_lambda          = 1.5,        # L2 regularisation (weight shrinkage)
    max_delta_step      = 1,          # clip gradient updates → more stable on energy data
    tree_method         = "hist",     # histogram-based algorithm: fast + memory-efficient
    grow_policy         = "depthwise",# standard depth-first growth
    early_stopping_rounds = 75,       # more patience for low lr
    eval_metric         = "rmse",
    n_jobs              = -1,          # use all CPU cores
    random_state        = RANDOM_SEED,
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def eval_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}


# ── Load & Clean Data ─────────────────────────────────────────────────────────
print("📂 Loading data…")
df = pd.read_csv(DATA_PATH, parse_dates=["Datetime"], index_col="Datetime")
df.columns = ["energy_mw"]
df = df.sort_index()
df = df[~df.index.duplicated(keep="first")]
full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="h")
df = df.reindex(full_range)
df["energy_mw"] = df["energy_mw"].interpolate(method="time")

daily    = df["energy_mw"].resample("D").mean()
split_ts = pd.Timestamp(SPLIT_DATE)
train    = daily[daily.index <= split_ts]
test     = daily[daily.index >  split_ts]

print(f"  Train: {train.index.min().date()} → {train.index.max().date()} ({len(train)} days)")
print(f"  Test : {test.index.min().date()} → {test.index.max().date()} ({len(test)} days)")


if "lstm" in selected:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM as LSTMLayer, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    scaler = MinMaxScaler()
    tr_sc  = scaler.fit_transform(train.values.reshape(-1, 1))
    te_sc  = scaler.transform(test.values.reshape(-1, 1))

    def create_sequences(data, lookback):
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i - lookback:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X_tr, y_tr = create_sequences(tr_sc, LOOKBACK)
    combined   = np.concatenate([tr_sc[-LOOKBACK:], te_sc])
    X_te, _    = create_sequences(combined, LOOKBACK)
    X_tr = X_tr.reshape(*X_tr.shape, 1)
    X_te = X_te.reshape(*X_te.shape, 1)

    lstm_model = Sequential([
        LSTMLayer(LSTM_UNITS, return_sequences=True, input_shape=(LOOKBACK, 1)),
        Dropout(0.2),
        LSTMLayer(LSTM_UNITS // 2),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    lstm_model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    cbs = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1),
    ]
    lstm_model.fit(X_tr, y_tr, epochs=EPOCHS, batch_size=32,
                   validation_split=0.1, callbacks=cbs, verbose=1)

    lstm_pred_sc = lstm_model.predict(X_te, verbose=0)
    lstm_pred    = scaler.inverse_transform(lstm_pred_sc).flatten()
    lstm_metrics = eval_metrics(test.values, lstm_pred)
    np.save(os.path.join(MODELS_DIR, "lstm_pred.npy"), lstm_pred)

    print(f"\n📊 LSTM — RMSE: {lstm_metrics['RMSE']:,.1f} MW | "
          f"MAE: {lstm_metrics['MAE']:,.1f} MW | MAPE: {lstm_metrics['MAPE']:.2f}%")
    lstm_model.save(os.path.join(MODELS_DIR, "lstm_model.keras"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    print("✅ LSTM saved → models/lstm_model.keras, models/scaler.pkl")


# ── SARIMAX ───────────────────────────────────────────────────────────────────
if "sarimax" in selected:
    print("\n📈 Training SARIMAX(2,1,1)(1,1,1,7)…")
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    def make_exog(series):
        d = pd.DataFrame(index=series.index)
        d["month"]      = series.index.month
        d["dayofweek"]  = series.index.dayofweek
        d["is_weekend"] = (series.index.dayofweek >= 5).astype(int)
        d["quarter"]    = series.index.quarter
        return d

    exog_train = make_exog(train)
    exog_test  = make_exog(test)

    sarimax_model = SARIMAX(
        train, exog=exog_train,
        order=(2, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    sarimax_result = sarimax_model.fit(disp=False, maxiter=200)
    sarimax_fc     = sarimax_result.forecast(steps=len(test), exog=exog_test)
    sarimax_pred   = np.maximum(sarimax_fc.values, 0)
    sarimax_metrics = eval_metrics(test.values, sarimax_pred)

    print(f"\n📊 SARIMAX — RMSE: {sarimax_metrics['RMSE']:,.1f} MW | "
          f"MAE: {sarimax_metrics['MAE']:,.1f} MW | MAPE: {sarimax_metrics['MAPE']:.2f}%")

    joblib.dump(sarimax_result, os.path.join(MODELS_DIR, "sarimax_result.pkl"))
    np.save(os.path.join(MODELS_DIR, "sarimax_pred.npy"), sarimax_pred)
    print("✅ SARIMAX saved → models/sarimax_result.pkl")


# ── XGBoost ───────────────────────────────────────────────────────────────────
if "xgb" in selected:
    print("\n🌲 Training XGBoost…")
    from xgboost import XGBRegressor

    def make_features(series):
        """Rich feature set for XGBoost time-series forecasting."""
        d = pd.DataFrame({"energy_mw": series})

        # ── Calendar features ─────────────────────────────────────────────────────
        d["dayofweek"]  = d.index.dayofweek          # 0=Mon … 6=Sun
        d["month"]      = d.index.month              # 1–12
        d["quarter"]    = d.index.quarter            # 1–4
        d["year"]       = d.index.year
        d["dayofyear"]  = d.index.dayofyear          # 1–365/6
        d["is_weekend"] = (d.index.dayofweek >= 5).astype(int)
        d["weekofyear"] = d.index.isocalendar().week.astype(int)
        # Linear trend (days since start) — captures long-term drift
        d["trend"]      = (d.index - d.index.min()).days

        # ── Fourier features — encode cyclical patterns without ordinal bias ───────
        # Weekly cycle (period = 7 days)
        for k in [1, 2]:                             # 1st and 2nd harmonics
            d[f"sin_week_{k}"] = np.sin(2 * np.pi * k * d["dayofweek"] / 7)
            d[f"cos_week_{k}"] = np.cos(2 * np.pi * k * d["dayofweek"] / 7)
        # Yearly cycle (period ≈ 365.25 days)
        for k in [1, 2, 3]:                          # 3 harmonics for richer annual shape
            d[f"sin_year_{k}"] = np.sin(2 * np.pi * k * d["dayofyear"] / 365.25)
            d[f"cos_year_{k}"] = np.cos(2 * np.pi * k * d["dayofyear"] / 365.25)

        # ── Lag features ─────────────────────────────────────────────────────────
        # Short-term: yesterday, 2–6 days ago (capture recent momentum)
        for lag in [1, 2, 3, 4, 5, 6]:
            d[f"lag_{lag}"]   = d["energy_mw"].shift(lag)
        # Medium-term: same day last week / fortnight / month
        for lag in [7, 14, 21, 28]:
            d[f"lag_{lag}"]   = d["energy_mw"].shift(lag)
        # Long-term: same day ~1 year ago (seasonal baseline)
        d["lag_365"]           = d["energy_mw"].shift(365)
        # Lag differences (rate of change)
        d["lag_diff_1"]        = d["lag_1"] - d["lag_2"]    # 1-day delta
        d["lag_diff_7"]        = d["lag_1"] - d["lag_7"]    # week-over-week delta

        # ── Rolling-window statistics (shifted by 1 to avoid data leakage) ────────
        src = d["energy_mw"].shift(1)                # always use yesterday's value as base
        for w in [7, 14, 30, 90]:
            d[f"roll_mean_{w}"] = src.rolling(w).mean()
            d[f"roll_std_{w}"]  = src.rolling(w).std()
            d[f"roll_min_{w}"]  = src.rolling(w).min()
            d[f"roll_max_{w}"]  = src.rolling(w).max()
            d[f"roll_range_{w}"] = d[f"roll_max_{w}"] - d[f"roll_min_{w}"]

        # ── Exponentially-weighted moving averages (recent days weighted more) ─────
        for span in [7, 14, 30]:
            d[f"ewma_{span}"]   = src.ewm(span=span, adjust=False).mean()

        return d.dropna()

    full_feat  = make_features(daily)
    feat_cols  = [c for c in full_feat.columns if c != "energy_mw"]
    xgb_train  = full_feat[full_feat.index <= split_ts]
    xgb_test   = full_feat[full_feat.index >  split_ts]

    xgb_model = XGBRegressor(**XGB)
    xgb_model.fit(
        xgb_train[feat_cols], xgb_train["energy_mw"],
        eval_set=[(xgb_test[feat_cols], xgb_test["energy_mw"])],
        verbose=100,
    )
    print(f"  Best round: {xgb_model.best_iteration} / {XGB['n_estimators']}")
    xgb_pred    = xgb_model.predict(xgb_test[feat_cols])
    xgb_metrics = eval_metrics(xgb_test["energy_mw"].values, xgb_pred)
    np.save(os.path.join(MODELS_DIR, "xgb_pred.npy"), xgb_pred)

    print(f"\n📊 XGBoost — RMSE: {xgb_metrics['RMSE']:,.1f} MW | "
          f"MAE: {xgb_metrics['MAE']:,.1f} MW | MAPE: {xgb_metrics['MAPE']:.2f}%")

    xgb_model.save_model(os.path.join(MODELS_DIR, "xgb_model.json"))
    print("✅ XGBoost saved → models/xgb_model.json")


# ── Collect results & pick best ───────────────────────────────────────────────
all_metrics = {}
if "lstm"    in selected: all_metrics["LSTM"]    = lstm_metrics
if "sarimax" in selected: all_metrics["SARIMAX"] = sarimax_metrics
if "xgb"     in selected: all_metrics["XGBoost"] = xgb_metrics

best = min(all_metrics, key=lambda m: all_metrics[m]["RMSE"]) if all_metrics else "N/A"

# Accumulate saved predictions
saved_preds = {}
if "lstm"    in selected: saved_preds["lstm"]    = lstm_pred
if "sarimax" in selected: saved_preds["sarimax"] = sarimax_pred
if "xgb"     in selected: saved_preds["xgb"]     = xgb_pred

# Save/merge metadata (keep existing entries for models NOT retrained)
meta_path = os.path.join(MODELS_DIR, "metadata.json")
existing  = {}
if os.path.exists(meta_path):
    with open(meta_path) as f:
        existing = json.load(f)

# Merge with existing metadata so untrained models aren't wiped
existing_metrics = existing.get("metrics", {})
existing_metrics.update(all_metrics)               # overwrite only retrained ones

# XGBoost test index (varies by lag dropna)
if "xgb" in selected:
    xgb_idx = xgb_test.index.strftime("%Y-%m-%d").tolist()
    with open(os.path.join(MODELS_DIR, "xgb_test_index.json"), "w") as fj:
        json.dump(xgb_idx, fj)

metadata = {
    **existing,                              # carry over untouched fields
    "split_date":  SPLIT_DATE,
    "lookback":    LOOKBACK,
    "lstm_units":  LSTM_UNITS,
    "epochs":      EPOCHS,
    "xgb_params":  {k: v for k, v in XGB.items()
                    if k not in ("early_stopping_rounds", "eval_metric")},
    "feat_cols":   feat_cols if "xgb" in selected else existing.get("feat_cols", []),
    "best_model":  best,
    "models_trained": selected,
    "metrics":     existing_metrics,
}
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n{'='*52}")
print(f"  Models trained : {', '.join(m.upper() for m in selected)}")
for name, m in all_metrics.items():
    flag = " 🏆" if name == best else ""
    print(f"  {name:<10} RMSE {m['RMSE']:>8,.1f} MW  MAPE {m['MAPE']:>5.2f}%{flag}")
print(f"  Best model     : {best}")
print(f"  Saved to       : {MODELS_DIR}/")
print(f"{'='*52}")
print(f"\nRun: streamlit run app.py")

