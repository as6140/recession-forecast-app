# app.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from datetime import datetime

# --- HEADER & AUTHOR ---
st.set_page_config(page_title="US Recession Probability", layout="wide")
st.title("🇺🇸 US Recession Probability Forecast (Next 12 Months)")
st.caption(f"📅 Last updated: {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}")
st.markdown(
    "Created by [Alex Shropshire](https://www.linkedin.com/in/alexandershropshire/) • "
    "[GitHub](https://github.com/as6140)"
)

# --- Refresh Button ---
if st.sidebar.button("🔁 Manually Refresh Data"):
    st.session_state["refresh"] = True
if st.session_state.get("refresh"):
    st.session_state["refresh"] = False
    st.experimental_rerun()

# --- FRED Config ---
FRED_API_KEY = st.secrets["FRED_API_KEY"]
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
RECESSIONS = [("2001-03-01", "2001-11-30"), ("2007-12-01", "2009-06-30"), ("2020-02-01", "2020-04-30")]

# --- Category Definitions ---
categories = [
    {"name": "Yield Curve", "series_id": "T10Y3M", "unit": "%", "default_weight": 25,
     "description": "An inverted curve (10Y < 3M) often signals a coming recession."},
    {"name": "Unemployment Rate", "series_id": "UNRATE", "unit": "%", "default_weight": 15,
     "description": "Rising unemployment tends to reflect economic weakening."},
    {"name": "Leading Index", "series_id": "USSLIND", "unit": "index", "default_weight": 20,
     "description": "Persistent declines in the LEI often precede recessions."},
    {"name": "Consumer Sentiment", "series_id": "UMCSENT", "unit": "index", "default_weight": 15,
     "description": "Low household confidence can signal an economic downturn."},
    {"name": "Fed Funds Rate", "series_id": "FEDFUNDS", "unit": "%", "default_weight": 15,
     "description": "High interest rates can slow growth and trigger contraction."},
    {"name": "S&P 500", "series_id": "SP500", "unit": "points", "default_weight": 10,
     "description": "Sharp declines in equities can signal deteriorating expectations."}
]

# --- Weight Sliders ---
st.sidebar.header("⚖️ Adjust Category Weights (Total = 100%)")
weights = {}
total_weight = 0
for cat in categories:
    w = st.sidebar.slider(f"{cat['name']} Weight (%)", 0, 100, cat["default_weight"])
    weights[cat["name"]] = w
    total_weight += w
if total_weight != 100:
    st.sidebar.error(f"Weights must total 100% (currently {total_weight}%)")
    st.stop()

# --- Data Fetching ---
def fetch_series(series_id, start="2000-01-01"):
    url = f"{FRED_BASE}?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json&observation_start={start}"
    r = requests.get(url).json()
    return pd.DataFrame([
        {"date": obs["date"], "value": float(obs["value"])}
        for obs in r.get("observations", []) if obs["value"] != "."
    ])

# --- Build Combined DataFrame ---
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
df_all.sort_values("date", inplace=True)

# --- Forecast Model ---
def rule_based_prob(row):
    scores = {
        "Yield Curve": 0.6 if row["Yield Curve"] < 0 else 0.3,
        "Unemployment Rate": 0.6 if row["Unemployment Rate"] > 4 else 0.4,
        "Leading Index": 0.6 if row["Leading Index"] < 0 else 0.4,
        "Consumer Sentiment": 0.4 if row["Consumer Sentiment"] > 70 else 0.6,
        "Fed Funds Rate": 0.5 if row["Fed Funds Rate"] > 4 else 0.3,
        "S&P 500": 0.4 if row["S&P 500"] > 4000 else 0.6
    }
    return sum((weights[k] / 100) * scores[k] for k in weights)

df_all["forecast"] = df_all.apply(rule_based_prob, axis=1)

# --- Display Current Forecast ---
latest_prob = df_all["forecast"].iloc[-1]
st.subheader("📊 Recession Probability Forecast")
st.metric("Current Probability", f"{latest_prob:.1%}")

# --- Probability History Line Chart (2yr minimum) ---
st.markdown("### 📈 Recession Probability Over Time (Past 2+ Years)")
min_date = df_all["date"].max() - pd.DateOffset(months=24)
df_recent = df_all[df_all["date"] >= min_date]
fig = px.line(df_recent, x="date", y="forecast", title="Recession Probability (Trailing 2+ Years)",
              labels={"forecast": "Recession Probability"})
for r in RECESSIONS:
    fig.add_vrect(x0=r[0], x1=r[1], fillcolor="gray", opacity=0.15, line_width=0)
st.plotly_chart(fig, use_container_width=True)

# --- Component Interpretation Section ---
st.subheader("🔍 Metric Interpretations")
latest_row = df_all.iloc[-1]
for cat in categories:
    series = df_raw[cat["name"]]
    current = latest_row[cat["name"]]
    trend = "increasing" if current > series[cat["name"]].iloc[-4] else "decreasing"
    baseline = f"(Typical threshold: {70 if 'Sentiment' in cat['name'] else 0})"
    st.markdown(f"**{cat['name']}** ({current:.2f} {cat['unit']}, {trend}) — {cat['description']} {baseline}")

# --- Aggregated Score Breakdown Chart ---
st.subheader("📊 Weighted Contribution by Category")
score_components = []
for cat in categories:
    c = cat["name"]
    if "forecast" in df_all.columns:
        raw_score = rule_based_prob(df_all.iloc[-1])
    raw_score = {
        "Yield Curve": 0.6 if latest_row["Yield Curve"] < 0 else 0.3,
        "Unemployment Rate": 0.6 if latest_row["Unemployment Rate"] > 4 else 0.4,
        "Leading Index": 0.6 if latest_row["Leading Index"] < 0 else 0.4,
        "Consumer Sentiment": 0.4 if latest_row["Consumer Sentiment"] > 70 else 0.6,
        "Fed Funds Rate": 0.5 if latest_row["Fed Funds Rate"] > 4 else 0.3,
        "S&P 500": 0.4 if latest_row["S&P 500"] > 4000 else 0.6
    }[c]
    weight_pct = weights[c] / 100
    score_components.append({
        "Category": c,
        "Score": raw_score,
        "Weight": weight_pct,
        "Weighted Score": raw_score * weight_pct
    })

df_breakdown = pd.DataFrame(score_components)
fig_bar = px.bar(df_breakdown, x="Category", y="Weighted Score",
                 color="Weighted Score", color_continuous_scale="RdYlGn_r",
                 title="Weighted Score by Category")
st.plotly_chart(fig_bar, use_container_width=True)

# --- Input Data Preview & Download (Last 24+ Months) ---
st.subheader("📥 Downloadable Input Data (Last 2+ Years)")
df_input_preview = df_all[df_all["date"] >= min_date].copy()
st.dataframe(df_input_preview.tail(24).reset_index(drop=True))
csv = df_input_preview.to_csv(index=False)
st.download_button("Download Input Dataset (CSV)", csv, "recession_input_data.csv")
