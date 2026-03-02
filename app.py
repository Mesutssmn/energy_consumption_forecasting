"""
PJM AEP Energy Consumption Forecasting — Streamlit App
Comparing SARIMAX vs LSTM vs XGBoost
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PJM AEP Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background: #0f1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2030, #252839);
        border: 1px solid #2d3150;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-label { color: #8b92b8; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; }
    .metric-value { color: #e8eaf6; font-size: 1.6rem; font-weight: 700; margin-top: 4px; }
    .metric-sub   { color: #5c6bc0; font-size: 0.75rem; margin-top: 2px; }
    .section-header {
        background: linear-gradient(90deg, #3949ab, #1a237e);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        color: white;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #1e2030;
        border-radius: 8px;
        color: #8b92b8;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3949ab, #5c6bc0) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ PJM AEP Forecasting")
    st.caption("PJM Hourly Energy Consumption — AEP Region (Ohio)")
    st.divider()

    st.markdown("### 📂 Dataset")
    data_path = st.text_input("CSV path", value="archive/AEP_hourly.csv")
    split_date = st.text_input("Train/Test split date", value="2017-12-31")

    st.divider()
    st.markdown("### 🔧 Models to Run")
    run_sarimax = st.checkbox("SARIMAX", value=True)
    run_lstm    = st.checkbox("LSTM",    value=True)
    run_xgb     = st.checkbox("XGBoost", value=True)

    st.divider()
    st.markdown("### ⚙️ SARIMAX Parameters")
    sar_p = st.slider("p (AR)",  0, 3, 2)
    sar_d = st.slider("d (diff)",0, 2, 1)
    sar_q = st.slider("q (MA)",  0, 3, 1)
    sar_P = st.slider("P (seasonal AR)",  0, 2, 1)
    sar_D = st.slider("D (seasonal diff)",0, 2, 1)
    sar_Q = st.slider("Q (seasonal MA)",  0, 2, 1)
    sar_s = st.select_slider("s (season period)", [7, 12, 14, 30], value=7)

    st.divider()
    st.markdown("### ⚙️ LSTM Settings")
    lookback   = st.slider("Lookback window (days)", 30, 120, 60, step=10)
    lstm_units = st.slider("LSTM units", 32, 128, 64, step=16)
    epochs     = st.slider("Max epochs", 20, 200, 100, step=10)

    st.divider()
    st.markdown("### ⚙️ XGBoost Settings")
    n_estimators      = st.slider("n_estimators",     200, 2000, 1000, step=100)
    learning_rate     = st.select_slider("learning_rate",
                            [0.005, 0.01, 0.02, 0.05, 0.1], value=0.05)
    max_depth         = st.slider("max_depth",         3, 10, 6)
    min_child_weight  = st.slider("min_child_weight",  1, 20, 3)
    subsample         = st.slider("subsample",         0.5, 1.0, 0.80, step=0.05)
    colsample_bytree  = st.slider("colsample_bytree",  0.5, 1.0, 0.80, step=0.05)
    colsample_bylevel = st.slider("colsample_bylevel", 0.6, 1.0, 0.80, step=0.05)
    gamma             = st.select_slider("gamma (min split gain)",
                            [0.0, 0.05, 0.1, 0.2, 0.5], value=0.0)
    reg_alpha         = st.select_slider("reg_alpha (L1)",
                            [0.0, 0.1, 0.2, 0.5, 1.0], value=0.1)
    reg_lambda        = st.select_slider("reg_lambda (L2)",
                            [0.5, 1.0, 1.5, 2.0, 3.0], value=1.0)

    st.divider()
    run_btn      = st.button("🚀 Run All Models", type="primary", use_container_width=True)
    load_pretrained = st.button("⚡ Load Pre-trained Model", use_container_width=True)

# ── Data Loading ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading and cleaning data…")
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Datetime"], index_col="Datetime")
    df.columns = ["energy_mw"]
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]          # fix duplicate timestamps
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="h")
    df = df.reindex(full_range)
    df["energy_mw"] = df["energy_mw"].interpolate(method="time")
    return df

# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown("# ⚡ PJM AEP Energy Consumption Forecasting")
st.markdown("**Comparing SARIMAX · LSTM · XGBoost** on the PJM Hourly Energy dataset (AEP Region, Ohio)")
st.divider()

# Load data
try:
    df = load_data(data_path)
except FileNotFoundError:
    st.error(f"❌ File not found: `{data_path}`. Place `AEP_hourly.csv` in the `archive/` folder.")
    st.stop()

# Top-level KPIs
daily = df["energy_mw"].resample("D").mean()
split_ts = pd.Timestamp(split_date)
train = daily[daily.index <= split_ts]
test  = daily[daily.index >  split_ts]

k1, k2, k3, k4, k5 = st.columns(5)
def kpi(col, label, value, sub=""):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

kpi(k1, "Total Hours",   f"{len(df):,}",                "hourly records")
kpi(k2, "Date Range",    f"{df.index.min().year}–{df.index.max().year}", "full span")
kpi(k3, "Avg Energy",    f"{df['energy_mw'].mean():,.0f} MW", "mean hourly")
kpi(k4, "Train Days",    f"{len(train):,}",             "days in train")
kpi(k5, "Test Days",     f"{len(test):,}",              "days in test")

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────────────────────────
tab_pretrained, tab_eda, tab_stat, tab_models, tab_compare = st.tabs([
    "⚡ Pre-trained Best Model",
    "📊 Exploratory Analysis",
    "🔬 Stationarity Tests",
    "🤖 Model Training",
    "🏆 Model Comparison",
])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 0: PRE-TRAINED BEST MODEL
# ──────────────────────────────────────────────────────────────────────────────
with tab_pretrained:
    import os, json as _json
    MODELS_DIR = "models"
    meta_path  = os.path.join(MODELS_DIR, "metadata.json")

    if not os.path.exists(meta_path):
        st.warning("⚠️ No pre-trained models found. Run the one-time training script first:")
        st.code("python train_best_model.py", language="bash")
        st.info("This trains LSTM + XGBoost, picks the best, and saves everything to `models/`.")
    else:
        with open(meta_path) as f:
            meta = _json.load(f)

        best_name = meta["best_model"]
        metrics   = meta["metrics"]
        split_ts_pt = pd.Timestamp(meta["split_date"])
        test_pt     = daily[daily.index > split_ts_pt]

        st.markdown(f"### ⚡ Pre-trained Best Model: **{best_name}**")
        st.caption(f"Trained on data up to {meta['split_date']} · "
                   f"Lookback: {meta['lookback']}d · "
                   f"LSTM units: {meta['lstm_units']}")

        # Metric comparison cards
        c1, c2 = st.columns(2)
        for col, mname, color in [(c1, "LSTM", "#43a047"), (c2, "XGBoost", "#ab47bc")]:
            if mname in metrics:
                m = metrics[mname]
                suffix = " 🏆" if mname == best_name else ""
                col.markdown(f"""
                <div class="metric-card" style="border-color:{color}">
                    <div class="metric-label">{mname}{suffix}</div>
                    <div class="metric-value" style="color:{color}">{m['RMSE']:,.0f} MW</div>
                    <div class="metric-sub">RMSE &nbsp;|&nbsp; MAE {m['MAE']:,.0f} MW &nbsp;|&nbsp; MAPE {m['MAPE']:.2f}%</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        @st.cache_resource(show_spinner="Loading best model from disk…")
        def load_best_model(models_dir, best):
            if best == "LSTM":
                import tensorflow as tf
                model  = tf.keras.models.load_model(os.path.join(models_dir, "lstm_model.keras"))
                scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
                return model, scaler
            else:
                from xgboost import XGBRegressor
                model = XGBRegressor()
                model.load_model(os.path.join(models_dir, "xgb_model.json"))
                return model, None

        @st.cache_data(show_spinner="Generating predictions…")
        def get_pretrained_preds(models_dir, best, _train, _test, lb, feat_cols):
            lstm_path = os.path.join(models_dir, "lstm_pred.npy")
            xgb_path  = os.path.join(models_dir, "xgb_pred.npy")
            preds = {}
            if os.path.exists(lstm_path):
                preds["LSTM"] = np.load(lstm_path)
            if os.path.exists(xgb_path):
                preds["XGBoost"] = np.load(xgb_path)
            return preds

        feat_cols_pt = meta.get("feat_cols", [])
        preds_pt = get_pretrained_preds(MODELS_DIR, best_name, train,
                                        test_pt, meta["lookback"], feat_cols_pt)

        # Charts
        colors_pt = {"LSTM": "#43a047", "XGBoost": "#ab47bc"}
        for mname, pred_arr in preds_pt.items():
            st.markdown(f"#### {mname} Forecast")
            # Align length
            n = min(len(test_pt), len(pred_arr))
            t_idx = test_pt.index[:n]
            # Full view
            fig_pt = make_subplots(rows=2, cols=1,
                subplot_titles=[f"{mname} — Full Test Period",
                                f"{mname} — First 60 Days"])
            fig_pt.add_trace(go.Scatter(x=train[-90:].index, y=train[-90:].values,
                name="Train (last 90d)", line=dict(color="#5c6bc0")), row=1, col=1)
            fig_pt.add_trace(go.Scatter(x=t_idx, y=test_pt.values[:n],
                name="Actual", line=dict(color="white")), row=1, col=1)
            fig_pt.add_trace(go.Scatter(x=t_idx, y=pred_arr[:n],
                name="Forecast", line=dict(color=colors_pt[mname], dash="dash")), row=1, col=1)
            # Zoom 60d
            n60 = min(60, n)
            fig_pt.add_trace(go.Scatter(x=t_idx[:n60], y=test_pt.values[:n60],
                name="Actual", line=dict(color="white"), showlegend=False), row=2, col=1)
            fig_pt.add_trace(go.Scatter(x=t_idx[:n60], y=pred_arr[:n60],
                name="Forecast", line=dict(color=colors_pt[mname], dash="dash"),
                showlegend=False), row=2, col=1)
            fig_pt.update_layout(template="plotly_dark", height=580,
                margin=dict(l=40, r=20, t=50, b=20), hovermode="x unified")
            st.plotly_chart(fig_pt, use_container_width=True)

        # Overlay
        if len(preds_pt) > 1:
            st.markdown("#### 📊 All Models Overlay")
            fig_ov_pt = go.Figure()
            fig_ov_pt.add_trace(go.Scatter(x=test_pt.index, y=test_pt.values,
                name="Actual", line=dict(color="white", width=1.5)))
            for mname, pred_arr in preds_pt.items():
                n = min(len(test_pt), len(pred_arr))
                fig_ov_pt.add_trace(go.Scatter(
                    x=test_pt.index[:n], y=pred_arr[:n],
                    name=mname, line=dict(color=colors_pt[mname], dash="dash")))
            fig_ov_pt.update_layout(
                template="plotly_dark", height=400, hovermode="x unified",
                xaxis_title="Date", yaxis_title="Energy (MW)",
                margin=dict(l=40, r=20, t=20, b=40))
            st.plotly_chart(fig_ov_pt, use_container_width=True)

        st.success(f"✅ Loaded from `{MODELS_DIR}/` — no retraining needed!")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1: EDA
# ──────────────────────────────────────────────────────────────────────────────
with tab_eda:
    st.markdown("### 📈 Full Hourly Series")

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=df.index, y=df["energy_mw"],
        mode="lines", name="Energy (MW)",
        line=dict(color="#5c6bc0", width=0.6),
    ))
    fig.update_layout(
        template="plotly_dark", height=300,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis_title="Date", yaxis_title="MW",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🔎 Zoom: 2017")
        zoom_2017 = df.loc["2017"]
        fig2 = go.Figure(go.Scattergl(
            x=zoom_2017.index, y=zoom_2017["energy_mw"],
            mode="lines", line=dict(color="#ef6c00", width=0.8),
        ))
        fig2.update_layout(template="plotly_dark", height=280,
                           margin=dict(l=40, r=20, t=10, b=40))
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.markdown("### 🔎 Zoom: Jan 2017 (2 weeks)")
        zoom_2w = df.loc["2017-01-01":"2017-01-14"]
        fig3 = go.Figure(go.Scatter(
            x=zoom_2w.index, y=zoom_2w["energy_mw"],
            mode="lines+markers",
            line=dict(color="#43a047", width=1.5),
            marker=dict(size=3),
        ))
        fig3.update_layout(template="plotly_dark", height=280,
                           margin=dict(l=40, r=20, t=10, b=40))
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    st.markdown("### 🗓️ Seasonality Patterns")

    ca, cb, cc = st.columns(3)
    with ca:
        hour_avg = df.groupby(df.index.hour)["energy_mw"].mean().reset_index()
        hour_avg.columns = ["Hour", "Mean MW"]
        fig_h = px.line(hour_avg, x="Hour", y="Mean MW",
                        title="By Hour of Day",
                        markers=True, template="plotly_dark")
        fig_h.update_traces(line_color="#5c6bc0")
        fig_h.update_layout(height=300, margin=dict(l=30, r=20, t=40, b=30))
        st.plotly_chart(fig_h, use_container_width=True)

    with cb:
        days_lbl = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        dow_avg = df.groupby(df.index.dayofweek)["energy_mw"].mean().reset_index()
        dow_avg.columns = ["DOW", "Mean MW"]
        dow_avg["Day"] = dow_avg["DOW"].map(dict(enumerate(days_lbl)))
        fig_d = px.line(dow_avg, x="Day", y="Mean MW",
                        title="By Day of Week",
                        markers=True, template="plotly_dark")
        fig_d.update_traces(line_color="#ef6c00")
        fig_d.update_layout(height=300, margin=dict(l=30, r=20, t=40, b=30))
        st.plotly_chart(fig_d, use_container_width=True)

    with cc:
        months_lbl = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        mon_avg = df.groupby(df.index.month)["energy_mw"].mean().reset_index()
        mon_avg.columns = ["Month", "Mean MW"]
        mon_avg["MonthName"] = mon_avg["Month"].map(dict(enumerate(months_lbl, 1)))
        fig_m = px.line(mon_avg, x="MonthName", y="Mean MW",
                        title="By Month of Year",
                        markers=True, template="plotly_dark")
        fig_m.update_traces(line_color="#43a047")
        fig_m.update_layout(height=300, margin=dict(l=30, r=20, t=40, b=30))
        st.plotly_chart(fig_m, use_container_width=True)

    st.divider()
    st.markdown("### 🧩 Seasonal Decomposition (Daily Resampled)")
    from statsmodels.tsa.seasonal import seasonal_decompose

    @st.cache_data(show_spinner="Running seasonal decomposition…")
    def decompose(daily_series):
        return seasonal_decompose(daily_series, model="additive", period=365)

    decomp = decompose(daily)
    fig_dec = make_subplots(rows=4, cols=1,
                            subplot_titles=["Observed","Trend","Seasonality","Residual"],
                            shared_xaxes=True)
    for i, (comp, color) in enumerate(zip(
        [decomp.observed, decomp.trend, decomp.seasonal, decomp.resid],
        ["#5c6bc0","#ef6c00","#43a047","#e53935"]), 1):
        fig_dec.add_trace(go.Scattergl(x=comp.index, y=comp.values,
                                       mode="lines", line=dict(color=color, width=0.8),
                                       showlegend=False), row=i, col=1)
    fig_dec.update_layout(template="plotly_dark", height=700,
                          margin=dict(l=40, r=20, t=40, b=20))
    st.plotly_chart(fig_dec, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2: STATIONARITY TESTS
# ──────────────────────────────────────────────────────────────────────────────
with tab_stat:
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st.markdown("### 🔬 ADF & KPSS Tests")

    @st.cache_data(show_spinner="Running stationarity tests…")
    def run_stationarity(series):
        results = {}
        for name, s in [("Original (Daily)", series), ("First Difference", series.diff().dropna())]:
            adf   = adfuller(s.dropna(), autolag="AIC")
            kpss_ = kpss(s.dropna(), regression="c", nlags="auto")
            results[name] = {
                "ADF Statistic": round(adf[0], 4),
                "ADF p-value":   round(adf[1], 4),
                "ADF Result":    "✅ Stationary" if adf[1] < 0.05 else "❌ Non-Stationary",
                "KPSS Statistic": round(kpss_[0], 4),
                "KPSS p-value":  round(kpss_[1], 4),
                "KPSS Result":   "✅ Stationary" if kpss_[1] > 0.05 else "❌ Non-Stationary",
            }
        return results

    stat_results = run_stationarity(daily)

    for series_name, res in stat_results.items():
        st.markdown(f"#### {series_name}")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**ADF Test** (H₀: Unit root = non-stationary)")
            st.metric("Statistic", res["ADF Statistic"])
            st.metric("p-value",   res["ADF p-value"])
            st.info(res["ADF Result"])
        with col_b:
            st.markdown("**KPSS Test** (H₀: Stationary)")
            st.metric("Statistic", res["KPSS Statistic"])
            st.metric("p-value",   res["KPSS p-value"])
            st.info(res["KPSS Result"])
        st.divider()

    st.markdown("### 📉 ACF & PACF")

    @st.cache_data(show_spinner="Computing ACF/PACF…")
    def compute_acf_pacf(series, lags=60):
        from statsmodels.tsa.stattools import acf, pacf
        acf_orig  = acf(series.dropna(), nlags=lags, fft=True)
        pacf_orig = pacf(series.dropna(), nlags=lags)
        acf_diff  = acf(series.diff().dropna(), nlags=lags, fft=True)
        pacf_diff = pacf(series.diff().dropna(), nlags=lags)
        return acf_orig, pacf_orig, acf_diff, pacf_diff

    acf_o, pacf_o, acf_d, pacf_d = compute_acf_pacf(daily)
    lags = list(range(len(acf_o)))

    fig_acf = make_subplots(rows=2, cols=2,
                            subplot_titles=["ACF — Original","PACF — Original",
                                            "ACF — First Diff","PACF — First Diff"])

    ci = 1.96 / np.sqrt(len(daily.dropna()))
    for row, (a, p, label) in enumerate([(acf_o, pacf_o, "Original"),
                                          (acf_d, pacf_d, "Diff")], 1):
        for col, vals in enumerate([a, p], 1):
            fig_acf.add_trace(go.Bar(x=lags, y=vals, marker_color="#5c6bc0",
                                     showlegend=False), row=row, col=col)
            fig_acf.add_hline(y=ci,  line_dash="dash", line_color="red",  row=row, col=col)
            fig_acf.add_hline(y=-ci, line_dash="dash", line_color="red",  row=row, col=col)

    fig_acf.update_layout(template="plotly_dark", height=500,
                           margin=dict(l=40, r=20, t=60, b=20))
    st.plotly_chart(fig_acf, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3: MODEL TRAINING
# ──────────────────────────────────────────────────────────────────────────────
with tab_models:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    def eval_metrics(y_true, y_pred, name):
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae  = float(mean_absolute_error(y_true, y_pred))
        mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
        return {"Model": name, "RMSE": rmse, "MAE": mae, "MAPE": mape}

    def forecast_chart(train_s, test_s, pred_arr, name, color):
        fig = make_subplots(rows=2, cols=1,
                            subplot_titles=[f"{name} — Full Test Period",
                                            f"{name} — First 60 Days (Zoom)"])
        # Full period
        fig.add_trace(go.Scatter(x=train_s[-90:].index, y=train_s[-90:].values,
                                 name="Train (last 90d)", line=dict(color="#5c6bc0")), row=1, col=1)
        fig.add_trace(go.Scatter(x=test_s.index, y=test_s.values,
                                 name="Actual", line=dict(color="#ffffff")), row=1, col=1)
        fig.add_trace(go.Scatter(x=test_s.index, y=pred_arr,
                                 name="Forecast", line=dict(color=color, dash="dash")), row=1, col=1)
        # Zoom
        n60 = min(60, len(test_s))
        fig.add_trace(go.Scatter(x=test_s.index[:n60], y=test_s.values[:n60],
                                 name="Actual (zoom)", line=dict(color="#ffffff"),
                                 showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=test_s.index[:n60], y=pred_arr[:n60],
                                 name="Forecast (zoom)", line=dict(color=color, dash="dash"),
                                 showlegend=False), row=2, col=1)
        fig.add_traces([go.Scatter(
            x=np.concatenate([test_s.index[:n60], test_s.index[:n60][::-1]]),
            y=np.concatenate([test_s.values[:n60], pred_arr[:n60][::-1]]),
            fill="toself", fillcolor=color, opacity=0.1,
            line=dict(color="rgba(0,0,0,0)"), showlegend=False,
        )], rows=[2], cols=[1])
        fig.update_layout(template="plotly_dark", height=600,
                          margin=dict(l=40, r=20, t=50, b=20), hovermode="x unified")
        return fig

    if not run_btn:
        st.info("👈 Configure models in the sidebar and click **Run All Models** to start training.")
    else:
        results_list = []

        # ── SARIMAX ──────────────────────────────────────────────────────────
        if run_sarimax:
            st.markdown("### 📈 SARIMAX")

            def make_exog(series):
                d = pd.DataFrame(index=series.index)
                d["month"]      = series.index.month
                d["dayofweek"]  = series.index.dayofweek
                d["is_weekend"] = (series.index.dayofweek >= 5).astype(int)
                d["quarter"]    = series.index.quarter
                return d

            exog_train = make_exog(train)
            exog_test  = make_exog(test)

            @st.cache_resource(show_spinner="Fitting SARIMAX — this may take a few minutes…")
            def fit_sarimax(_train, _exog_train, p, d, q, P, D, Q, s):
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                model = SARIMAX(_train, exog=_exog_train,
                                order=(p, d, q),
                                seasonal_order=(P, D, Q, s),
                                trend="n",
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                return model.fit(disp=False, maxiter=200)

            order_str = f"({sar_p},{sar_d},{sar_q})({sar_P},{sar_D},{sar_Q},{sar_s})"
            with st.spinner(f"Fitting SARIMAX{order_str}…"):
                sarimax_res = fit_sarimax(train, exog_train,
                                         sar_p, sar_d, sar_q,
                                         sar_P, sar_D, sar_Q, sar_s)

            sarimax_fc   = sarimax_res.forecast(steps=len(test), exog=exog_test)
            sarimax_pred = np.maximum(sarimax_fc.values, 0)
            sarimax_metrics = eval_metrics(test.values, sarimax_pred, "SARIMAX")
            results_list.append(sarimax_metrics)

            m1, m2, m3 = st.columns(3)
            m1.metric("RMSE", f"{sarimax_metrics['RMSE']:,.1f} MW")
            m2.metric("MAE",  f"{sarimax_metrics['MAE']:,.1f} MW")
            m3.metric("MAPE", f"{sarimax_metrics['MAPE']:.2f} %")

            st.plotly_chart(forecast_chart(train, test, sarimax_pred, "SARIMAX", "#ef6c00"),
                            use_container_width=True)
            st.divider()

        # ── LSTM ─────────────────────────────────────────────────────────────
        if run_lstm:
            st.markdown("### 🧠 LSTM")

            @st.cache_resource(show_spinner="Training LSTM…")
            def train_lstm(_train, _test, lb, units, ep):
                import tensorflow as tf
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import LSTM as LSTMLayer, Dense, Dropout
                from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

                tf.random.set_seed(42)
                scaler = MinMaxScaler()
                tr_sc  = scaler.fit_transform(_train.values.reshape(-1, 1))
                te_sc  = scaler.transform(_test.values.reshape(-1, 1))

                def seqs(data, lookback):
                    X, y = [], []
                    for i in range(lookback, len(data)):
                        X.append(data[i-lookback:i, 0])
                        y.append(data[i, 0])
                    return np.array(X), np.array(y)

                X_tr, y_tr = seqs(tr_sc, lb)
                combined   = np.concatenate([tr_sc[-lb:], te_sc])
                X_te, _    = seqs(combined, lb)
                X_tr = X_tr.reshape(*X_tr.shape, 1)
                X_te = X_te.reshape(*X_te.shape, 1)

                model = Sequential([
                    LSTMLayer(units, return_sequences=True, input_shape=(lb, 1)),
                    Dropout(0.2),
                    LSTMLayer(units // 2),
                    Dropout(0.2),
                    Dense(32, activation="relu"),
                    Dense(1),
                ])
                model.compile(optimizer="adam", loss="mse", metrics=["mae"])

                cbs = [
                    EarlyStopping(monitor="val_loss", patience=10,
                                  restore_best_weights=True, verbose=0),
                    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=0),
                ]
                history = model.fit(X_tr, y_tr, epochs=ep, batch_size=32,
                                    validation_split=0.1, callbacks=cbs, verbose=0)
                pred_sc = model.predict(X_te, verbose=0)
                pred    = scaler.inverse_transform(pred_sc).flatten()
                return pred, history.history

            with st.spinner("Training LSTM…"):
                lstm_pred, lstm_history = train_lstm(train, test, lookback, lstm_units, epochs)

            lstm_metrics = eval_metrics(test.values, lstm_pred, "LSTM")
            results_list.append(lstm_metrics)

            m1, m2, m3 = st.columns(3)
            m1.metric("RMSE", f"{lstm_metrics['RMSE']:,.1f} MW")
            m2.metric("MAE",  f"{lstm_metrics['MAE']:,.1f} MW")
            m3.metric("MAPE", f"{lstm_metrics['MAPE']:.2f} %")

            # Training curves
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(y=lstm_history["loss"],     name="Train Loss", line=dict(color="#5c6bc0")))
            fig_hist.add_trace(go.Scatter(y=lstm_history["val_loss"], name="Val Loss",   line=dict(color="#ef6c00", dash="dash")))
            fig_hist.update_layout(template="plotly_dark", height=250,
                                   title="Training Loss (MSE)", xaxis_title="Epoch",
                                   margin=dict(l=40, r=20, t=40, b=30))
            st.plotly_chart(fig_hist, use_container_width=True)

            st.plotly_chart(forecast_chart(train, test, lstm_pred, "LSTM", "#43a047"),
                            use_container_width=True)
            st.divider()

        # ── XGBoost ──────────────────────────────────────────────────────────
        if run_xgb:
            st.markdown("### 🌲 XGBoost")

            @st.cache_resource(show_spinner="Training XGBoost…")
            def train_xgboost(_data, split, n_est, lr, md, mcw, ss, cbt, cbl, gam, ra, rl):
                from xgboost import XGBRegressor

                def make_features(series):
                    """Original compact feature set — proven to give RMSE ~816 MW."""
                    d = pd.DataFrame({"energy_mw": series})
                    # Calendar
                    d["dayofweek"]  = d.index.dayofweek
                    d["month"]      = d.index.month
                    d["quarter"]    = d.index.quarter
                    d["year"]       = d.index.year
                    d["dayofyear"]  = d.index.dayofyear
                    d["is_weekend"] = (d.index.dayofweek >= 5).astype(int)
                    d["weekofyear"] = d.index.isocalendar().week.astype(int)
                    # Lags: short / medium / long-term
                    for lag in [1, 7, 14, 30, 365]:
                        d[f"lag_{lag}"] = d["energy_mw"].shift(lag)
                    # Rolling 7-day and 30-day stats (shift by 1: no data leakage)
                    for w in [7, 30]:
                        src = d["energy_mw"].shift(1)
                        d[f"roll_mean_{w}"] = src.rolling(w).mean()
                        d[f"roll_std_{w}"]  = src.rolling(w).std()
                        d[f"roll_min_{w}"]  = src.rolling(w).min()
                        d[f"roll_max_{w}"]  = src.rolling(w).max()
                    return d.dropna()

                full = make_features(_data)
                feat_cols = [c for c in full.columns if c != "energy_mw"]
                tr = full[full.index <= split]
                te = full[full.index >  split]

                model = XGBRegressor(
                    n_estimators          = n_est,
                    learning_rate         = lr,
                    max_depth             = md,
                    min_child_weight      = mcw,
                    subsample             = ss,
                    colsample_bytree      = cbt,
                    colsample_bylevel     = cbl,
                    gamma                 = gam,
                    reg_alpha             = ra,
                    reg_lambda            = rl,
                    tree_method           = "hist",
                    early_stopping_rounds = 50,
                    eval_metric           = "rmse",
                    n_jobs                = -1,
                    random_state          = 42,
                )
                model.fit(tr[feat_cols], tr["energy_mw"],
                          eval_set=[(te[feat_cols], te["energy_mw"])], verbose=False)
                pred = model.predict(te[feat_cols])
                imps = pd.Series(model.feature_importances_, index=feat_cols).sort_values()
                return pred, te["energy_mw"], imps

            with st.spinner("Training XGBoost…"):
                xgb_pred, y_te_xgb, feat_imps = train_xgboost(
                    daily, split_date,
                    n_estimators, learning_rate, max_depth, min_child_weight,
                    subsample, colsample_bytree, colsample_bylevel,
                    gamma, reg_alpha, reg_lambda,
                )

            xgb_metrics = eval_metrics(y_te_xgb.values, xgb_pred, "XGBoost")
            results_list.append(xgb_metrics)

            m1, m2, m3 = st.columns(3)
            m1.metric("RMSE", f"{xgb_metrics['RMSE']:,.1f} MW")
            m2.metric("MAE",  f"{xgb_metrics['MAE']:,.1f} MW")
            m3.metric("MAPE", f"{xgb_metrics['MAPE']:.2f} %")

            # Feature importance
            fig_imp = go.Figure(go.Bar(
                x=feat_imps.values, y=feat_imps.index,
                orientation="h", marker_color="#5c6bc0",
            ))
            fig_imp.update_layout(template="plotly_dark", height=420,
                                  title="XGBoost — Feature Importance",
                                  margin=dict(l=150, r=20, t=40, b=30))
            st.plotly_chart(fig_imp, use_container_width=True)

            st.plotly_chart(forecast_chart(train, y_te_xgb, xgb_pred, "XGBoost", "#ab47bc"),
                            use_container_width=True)

        # Store globally for comparison tab
        if results_list:
            st.session_state["results_list"] = results_list
            if run_sarimax: st.session_state["sarimax_pred"] = sarimax_pred
            if run_lstm:    st.session_state["lstm_pred"]    = lstm_pred
            if run_xgb:
                st.session_state["xgb_pred"]   = xgb_pred
                st.session_state["y_te_xgb"]   = y_te_xgb

# ──────────────────────────────────────────────────────────────────────────────
# TAB 4: MODEL COMPARISON
# ──────────────────────────────────────────────────────────────────────────────
with tab_compare:
    st.markdown("### 🏆 Model Comparison")

    if "results_list" not in st.session_state or not st.session_state["results_list"]:
        st.info("👈 Run models first using the **Run All Models** button in the sidebar.")
    else:
        results_df = pd.DataFrame(st.session_state["results_list"]).set_index("Model")

        # Metric bar charts
        colors_map = {"SARIMAX": "#ef6c00", "LSTM": "#43a047", "XGBoost": "#ab47bc"}
        bar_colors = [colors_map.get(m, "#5c6bc0") for m in results_df.index]

        fig_cmp = make_subplots(rows=1, cols=3,
                                subplot_titles=["RMSE (MW)", "MAE (MW)", "MAPE (%)"])
        for i, metric in enumerate(["RMSE", "MAE", "MAPE"], 1):
            fig_cmp.add_trace(go.Bar(
                x=results_df.index.tolist(),
                y=results_df[metric].tolist(),
                marker_color=bar_colors,
                text=[f"{v:.1f}" for v in results_df[metric]],
                textposition="outside",
                showlegend=False,
            ), row=1, col=i)
        fig_cmp.update_layout(template="plotly_dark", height=400,
                              margin=dict(l=30, r=30, t=60, b=30))
        st.plotly_chart(fig_cmp, use_container_width=True)

        # Summary table
        st.markdown("#### 📋 Metrics Summary")
        styled = results_df.round(2)
        st.dataframe(styled, use_container_width=True)

        # Best model callouts
        best_rmse = results_df["RMSE"].idxmin()
        best_mae  = results_df["MAE"].idxmin()
        best_mape = results_df["MAPE"].idxmin()

        ca, cb, cc = st.columns(3)
        ca.success(f"🥇 Best RMSE → **{best_rmse}**")
        cb.success(f"🥇 Best MAE  → **{best_mae}**")
        cc.success(f"🥇 Best MAPE → **{best_mape}**")

        # Overlay chart (common test index)
        st.divider()
        st.markdown("#### 📊 Forecast Overlay — All Models")

        fig_ov = go.Figure()
        # Actual
        fig_ov.add_trace(go.Scatter(x=test.index, y=test.values,
                                    name="Actual", line=dict(color="white", width=1.5)))
        if "sarimax_pred" in st.session_state:
            sp = st.session_state["sarimax_pred"]
            fig_ov.add_trace(go.Scatter(x=test.index[:len(sp)], y=sp,
                                        name="SARIMAX", line=dict(color="#ef6c00", dash="dash")))
        if "lstm_pred" in st.session_state:
            lp = st.session_state["lstm_pred"]
            fig_ov.add_trace(go.Scatter(x=test.index[:len(lp)], y=lp,
                                        name="LSTM", line=dict(color="#43a047", dash="dot")))
        if "xgb_pred" in st.session_state:
            xp = st.session_state["xgb_pred"]
            yt = st.session_state["y_te_xgb"]
            fig_ov.add_trace(go.Scatter(x=yt.index[:len(xp)], y=xp,
                                        name="XGBoost", line=dict(color="#ab47bc", dash="dashdot")))

        fig_ov.update_layout(template="plotly_dark", height=450,
                             xaxis_title="Date", yaxis_title="Energy (MW)",
                             hovermode="x unified",
                             margin=dict(l=40, r=20, t=20, b=40))
        st.plotly_chart(fig_ov, use_container_width=True)

        # Model notes
        st.divider()
        st.markdown("#### 📌 Model Insights")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
**📈 SARIMAX**
- ✅ Interpretable, captures weekly seasonality
- ✅ Uses exogenous features natively
- ❌ Assumes linearity, slow on large data
- 🎯 Best for: baseline & regulated environments
""")
        with c2:
            st.markdown("""
**🧠 LSTM**
- ✅ Captures long-range non-linear dependencies
- ✅ Handles complex temporal patterns
- ❌ Needs more data/tuning, black-box
- 🎯 Best for: multi-step ahead forecasting
""")
        with c3:
            st.markdown("""
**🌲 XGBoost**
- ✅ Fast, robust to outliers, interpretable (SHAP)
- ✅ Rich feature engineering support
- ❌ Not naturally sequential
- 🎯 Best for: single-step with rich features
""")
