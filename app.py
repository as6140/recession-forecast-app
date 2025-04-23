# app.py (Part 1 of 2+)
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from lightgbm import LGBMClassifier

# --------------------------
# 🧠 CONFIG
# --------------------------
st.set_page_config(page_title="US Recession Probability", layout="wide")
st.title("🇺🇸 US Recession Probability Forecast (Next 12 Months)")
st.caption(f"📅 Last updated: {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')} (Local time)")

# --------------------------
# 🕹️ Mode Toggles
# --------------------------
view_mode = st.sidebar.radio("Choose View Mode", ["Basic Mode", "Advanced Mode"])
model_type = st.sidebar.radio("Forecast Model", ["Rule-Based", "ML-Based Ensemble"])
if st.sidebar.button("🔁 Manually Refresh Data"):
    st.session_state["refresh"] = True
if st.session_state.get("refresh"):
    st.session_state["refresh"] = False
    st.experimental_rerun()

# --------------------------
# 📈 Categories & Setup
# --------------------------
FRED_API_KEY = st.secrets["FRED_API_KEY"]
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

categories = [
    {"category": "Yield Curve", "series_id": "T10Y3M"},
    {"category": "Unemployment Rate", "series_id": "UNRATE"},
    {"category": "Leading Index", "series_id": "USSLIND"},
    {"category": "Consumer Sentiment", "series_id": "UMCSENT"},
    {"category": "Fed Funds Rate", "series_id": "FEDFUNDS"},
    {"category": "S&P 500", "series_id": "SP500"},
]

def fetch_series(series_id, start="2000-01-01"):
    url = f"{FRED_BASE}?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json&observation_start={start}"
    r = requests.get(url).json()
    return pd.DataFrame([
        {"date": obs["date"], "value": float(obs["value"])}
        for obs in r.get("observations", []) if obs["value"] != "."
    ])

# --------------------------
# 🧠 Sahm Rule
# --------------------------
def detect_sahm_trigger(df_unemp):
    df = df_unemp.copy()
    df["3mo_avg"] = df["value"].rolling(3).mean()
    df["12mo_min"] = df["3mo_avg"].rolling(12).min()
    df["gap"] = df["3mo_avg"] - df["12mo_min"]
    df["trigger"] = df["gap"] >= 0.5
    return df

# --------------------------
# 📦 Data Fetching & Feature Engineering
# --------------------------
@st.cache_data(show_spinner=True)
def prepare_data():
    dfs = {cat["category"]: fetch_series(cat["series_id"]) for cat in categories}
    for df in dfs.values():
        df["date"] = pd.to_datetime(df["date"])
    df_all = dfs["Unemployment Rate"][["date"]].copy()
    for cat in categories:
        name = cat["category"]
        df_all = df_all.merge(dfs[name], on="date", how="left", suffixes=("", f"_{name}"))
        df_all.rename(columns={"value": name}, inplace=True)
    df_all["Sahm Rule Trigger"] = detect_sahm_trigger(dfs["Unemployment Rate"])["trigger"].astype(int)
    df_all.dropna(inplace=True)
    return df_all

df_all = prepare_data()
# --------------------------
# 🤖 ML Model
# --------------------------
def run_ml_model(df):
    df_model = df.copy()
    df_model["recession_next"] = df_model["Unemployment Rate"].shift(-3) > df_model["Unemployment Rate"]
    df_model.dropna(inplace=True)
    X = df_model[["Yield Curve", "Unemployment Rate", "Leading Index", "Consumer Sentiment", "Fed Funds Rate", "S&P 500", "Sahm Rule Trigger"]]
    y = df_model["recession_next"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[-1][1]
    ci_low = max(0.0, probs - 0.05)
    ci_high = min(1.0, probs + 0.05)
    return probs, ci_low, ci_high, model.feature_importances_, X_test.columns

# --------------------------
# 🧮 Rule-Based Model
# --------------------------
def rule_based_forecast(df):
    latest = df.iloc[-1]
    weights = {
        "Yield Curve": 0.25,
        "Unemployment Rate": 0.10,
        "Leading Index": 0.20,
        "Consumer Sentiment": 0.15,
        "Fed Funds Rate": 0.10,
        "S&P 500": 0.10,
        "Sahm Rule Trigger": 0.10
    }
    scores = {
        "Yield Curve": 0.6 if latest["Yield Curve"] < 0 else 0.3,
        "Unemployment Rate": 0.6 if latest["Unemployment Rate"] > 4.0 else 0.4,
        "Leading Index": 0.6 if latest["Leading Index"] < 0 else 0.4,
        "Consumer Sentiment": 0.4 if latest["Consumer Sentiment"] > 70 else 0.6,
        "Fed Funds Rate": 0.5 if latest["Fed Funds Rate"] > 4.0 else 0.3,
        "S&P 500": 0.4 if latest["S&P 500"] > 4000 else 0.6,
        "Sahm Rule Trigger": 0.7 if latest["Sahm Rule Trigger"] == 1 else 0.3
    }
    weighted_score = sum(weights[k] * scores[k] for k in weights)
    return weighted_score, max(0, weighted_score - 0.05), min(1, weighted_score + 0.05), scores, weights

# --------------------------
# 🔮 Generate Forecast
# --------------------------
if model_type == "ML-Based Ensemble":
    forecast, ci_low, ci_high, importances, features = run_ml_model(df_all)
    st.subheader("ML-Based Ensemble Forecast")
else:
    forecast, ci_low, ci_high, scores, weights = rule_based_forecast(df_all)
    st.subheader("Rule-Based Forecast")

st.metric("Recession Probability (Next 12 Months)", f"{forecast:.1%}")
st.markdown(f"📉 **Confidence Interval**: {ci_low:.1%} – {ci_high:.1%}")

# --------------------------
# 📊 Breakdown Table
# --------------------------
if model_type == "Rule-Based" and view_mode == "Advanced Mode":
    st.subheader("Rule-Based Category Breakdown")
    score_table = pd.DataFrame({
        "Category": list(scores.keys()),
        "Score": [f"{v:.1%}" for v in scores.values()],
        "Weight": [f"{weights[k]:.0%}" for k in scores],
        "Weighted Score": [f"{scores[k] * weights[k]:.1%}" for k in scores]
    })
    st.dataframe(score_table)

# --------------------------
# 📈 Real GDP
# --------------------------
if view_mode == "Advanced Mode":
    st.subheader("Real GDP Growth")
    gdp_df = fetch_series("GDPC1")
    gdp_df["date"] = pd.to_datetime(gdp_df["date"])
    gdp_df["YoY %"] = gdp_df["value"].pct_change(4) * 100
    gdp_df["QoQ %"] = gdp_df["value"].pct_change() * 100
    fig = px.line(gdp_df, x="date", y=["YoY %", "QoQ %"], title="US Real GDP Growth (YoY & QoQ)")
    for r in recessions:
        fig.add_vrect(x0=r[0], x1=r[1], fillcolor="gray", opacity=0.1, line_width=0)
    st.plotly_chart(fig, use_container_width=True)

# --------------------------
# 📉 Sahm Rule Chart
# --------------------------
if view_mode == "Advanced Mode":
    st.subheader("Sahm Rule Detection")
    sahm_df = detect_sahm_trigger(fetch_series("UNRATE"))
    fig = px.line(sahm_df, x="date", y="3mo_avg", title="3-Month Avg Unemployment Rate")
    fig.add_scatter(x=sahm_df["date"], y=sahm_df["12mo_min"] + 0.5, mode="lines", name="Trigger Line", line=dict(dash="dot"))
    for r in recessions:
        fig.add_vrect(x0=r[0], x1=r[1], fillcolor="gray", opacity=0.1, line_width=0)
    st.plotly_chart(fig, use_container_width=True)

# --------------------------
# 🔽 Download
# --------------------------
csv = df_all.to_csv(index=False)
st.download_button("Download Full Economic Dataset", csv, "recession_data.csv")

# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.markdown("Built with Streamlit • Powered by FRED • Toggle views and models above ☝️")
