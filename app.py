# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from datetime import datetime

# --- Config ---
st.set_page_config(page_title="US Recession Probability", layout="wide")
st.title("🇺🇸 US Recession Probability Forecast (Next 12 Months)")
st.caption(f"📅 Last updated: {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}")

# --- Manual Refresh ---
if st.sidebar.button("🔁 Manually Refresh Data"):
    st.session_state["refresh"] = True
if st.session_state.get("refresh"):
    st.session_state["refresh"] = False
    st.experimental_rerun()

# --- Constants ---
FRED_API_KEY = st.secrets["FRED_API_KEY"]
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
RECESSIONS = [
    ("2001-03-01", "2001-11-30"),
    ("2007-12-01", "2009-06-30"),
    ("2020-02-01", "2020-04-30")
]

# --- Categories & Weights ---
categories = [
    {"name": "Yield Curve", "series_id": "T10Y3M", "unit": "%", "default_weight": 25,
     "description": "The spread between 10-year and 3-month Treasury yields. A negative spread is a classic leading indicator of recession."},
    {"name": "Unemployment Rate", "series_id": "UNRATE", "unit": "%", "default_weight": 15,
     "description": "The national unemployment rate. Rising unemployment can indicate economic distress."},
    {"name": "Leading Index", "series_id": "USSLIND", "unit": "index", "default_weight": 20,
     "description": "The Conference Board’s leading economic index. Persistent decline often precedes recession."},
    {"name": "Consumer Sentiment", "series_id": "UMCSENT", "unit": "index", "default_weight": 15,
     "description": "University of Michigan consumer sentiment index. Low sentiment reflects household pessimism."},
    {"name": "Fed Funds Rate", "series_id": "FEDFUNDS", "unit": "%", "default_weight": 15,
     "description": "The Federal Reserve’s benchmark interest rate. High rates can restrain growth."},
    {"name": "S&P 500", "series_id": "SP500", "unit": "points", "default_weight": 10,
     "description": "US stock market index. Sharp declines or volatility can signal investor anxiety."}
]

# --- Helper Functions ---
def fetch_series(series_id, start="2018-01-01"):
    url = f"{FRED_BASE}?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json&observation_start={start}"
    r = requests.get(url).json()
    return pd.DataFrame([{"date": obs["date"], "value": float(obs["value"])} for obs in r.get("observations", []) if obs["value"] != "."])

def detect_sahm_trigger(df):
    df = df.copy()
    df["3mo_avg"] = df["value"].rolling(3).mean()
    df["12mo_min"] = df["3mo_avg"].rolling(12).min()
    df["gap"] = df["3mo_avg"] - df["12mo_min"]
    df["trigger"] = df["gap"] >= 0.5
    return df

# --- Weight Sliders (must total 100%) ---
st.sidebar.header("⚖️ Adjust Category Weights (Total = 100%)")
weights = {}
total_weight = 0
for cat in categories:
    w = st.sidebar.slider(f"{cat['name']} Weight (%)", 0, 100, cat["default_weight"])
    weights[cat["name"]] = w
    total_weight += w

if total_weight != 100:
    st.sidebar.error(f"Weights must sum to 100%. Current total: {total_weight}%")
    st.stop()

# --- Fetch Data ---
df_all = pd.DataFrame()
df_raw = {}
for cat in categories:
    df = fetch_series(cat["series_id"])
    df["date"] = pd.to_datetime(df["date"])
    df.rename(columns={"value": cat["name"]}, inplace=True)
    df_raw[cat["name"]] = df
    if df_all.empty:
        df_all = df[["date", cat["name"]]]
    else:
        df_all = pd.merge(df_all, df, on="date", how="inner")

# --- Scoring Logic ---
latest = df_all.iloc[-1]
scores = {
    "Yield Curve": 0.6 if latest["Yield Curve"] < 0 else 0.3,
    "Unemployment Rate": 0.6 if latest["Unemployment Rate"] > 4 else 0.4,
    "Leading Index": 0.6 if latest["Leading Index"] < 0 else 0.4,
    "Consumer Sentiment": 0.4 if latest["Consumer Sentiment"] > 70 else 0.6,
    "Fed Funds Rate": 0.5 if latest["Fed Funds Rate"] > 4 else 0.3,
    "S&P 500": 0.4 if latest["S&P 500"] > 4000 else 0.6
}

# --- Aggregate Forecast ---
weighted_score = sum((weights[k] / 100) * scores[k] for k in weights)
st.subheader("📊 Recession Probability Forecast")
st.metric("Recession Probability (Rule-Based)", f"{weighted_score:.1%}")

# --- Interpretation ---
st.markdown("### 🔍 Metric Interpretations")
for cat in categories:
    val = latest[cat["name"]]
    if cat["name"] == "Yield Curve":
        interp = "Inverted" if val < 0 else "Normal"
    elif cat["name"] == "Unemployment Rate":
        interp = "Elevated" if val > 4 else "Stable"
    elif cat["name"] == "Leading Index":
        interp = "Declining" if val < 0 else "Expanding"
    elif cat["name"] == "Consumer Sentiment":
        interp = "Weak" if val < 70 else "Strong"
    elif cat["name"] == "Fed Funds Rate":
        interp = "Restrictive" if val > 4 else "Accommodative"
    elif cat["name"] == "S&P 500":
        interp = "High" if val > 4000 else "Low"
    st.markdown(f"**{cat['name']}** ({val:.2f} {cat['unit']}): {interp} → {cat['description']}")

# --- Aggregated Score Bar Chart ---
breakdown_df = pd.DataFrame({
    "Category": list(scores.keys()),
    "Score": [scores[k] for k in scores],
    "Weight": [weights[k] / 100 for k in weights],
})
breakdown_df["Weighted Score"] = breakdown_df["Score"] * breakdown_df["Weight"]
fig_bar = px.bar(breakdown_df, x="Category", y="Weighted Score", title="Weighted Score by Category",
                 labels={"Weighted Score": "Weighted Risk Score"}, color="Weighted Score",
                 color_continuous_scale="RdYlGn_r")
st.plotly_chart(fig_bar, use_container_width=True)

# --- Line Charts with Banding ---
st.markdown("### 📈 Component Trends")
for cat in categories:
    df = df_raw[cat["name"]]
    y = cat["name"]
    val = latest[y]
    fig = px.line(df, x="date", y=y, title=f"{y} ({cat['unit']})", labels={y: y})
    min_y, max_y = df[y].min(), df[y].max()
    spread = (max_y - min_y) * 0.05
    mid = val
    fig.add_hrect(y0=min_y, y1=mid - spread, fillcolor="green", opacity=0.1, line_width=0)
    fig.add_hrect(y0=mid - spread, y1=mid + spread, fillcolor="yellow", opacity=0.1, line_width=0)
    fig.add_hrect(y0=mid + spread, y1=max_y, fillcolor="red", opacity=0.1, line_width=0)
    for r in RECESSIONS:
        fig.add_vrect(x0=r[0], x1=r[1], fillcolor="gray", opacity=0.15, line_width=0)
    fig.update_layout(hoverlabel=dict(bgcolor="white", font_color="black"))
    fig.update_traces(hovertemplate=f"%{{x}}<br><b>{y}</b>: %{{y:.2f}}")
    st.plotly_chart(fig, use_container_width=True)

# --- Footer ---
st.markdown("---")
st.download_button("📥 Download Data", df_all.to_csv(index=False), "recession_forecast_data.csv")
st.markdown("Built with Streamlit • Powered by FRED • Customize weights above ☝️")
