# app.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from datetime import datetime

# --- Config & Header ---
st.set_page_config(page_title="US Recession Probability", layout="wide")
st.title("🇺🇸 US Recession Probability Forecast (Next 12 Months)")
st.caption(f"📅 Last updated: {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}")
st.markdown("Created by [Alex Shropshire](https://www.linkedin.com/in/alexandershropshire/) • [GitHub](https://github.com/as6140)")

# --- Manual Refresh ---
if st.sidebar.button("🔁 Manually Refresh Data"):
    st.session_state["refresh"] = True
if st.session_state.get("refresh"):
    st.session_state["refresh"] = False
    st.experimental_rerun()

# --- Constants ---
FRED_API_KEY = st.secrets["FRED_API_KEY"]
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
RECESSIONS = [("2001-03-01", "2001-11-30"), ("2007-12-01", "2009-06-30"), ("2020-02-01", "2020-04-30")]
TODAY = datetime.today()

# --- Metric Definitions ---
categories = [
    {"name": "Yield Curve", "series_id": "T10Y3M", "unit": "%", "default_weight": 25, "description": "An inverted curve (10Y < 3M) often signals a recession."},
    {"name": "Unemployment Rate", "series_id": "UNRATE", "unit": "%", "default_weight": 15, "description": "Rising unemployment reflects labor market stress."},
    {"name": "Leading Index", "series_id": "USSLIND", "unit": "index", "default_weight": 20, "description": "Persistent declines in LEI suggest economic contraction."},
    {"name": "Consumer Sentiment", "series_id": "UMCSENT", "unit": "index", "default_weight": 15, "description": "Low confidence indicates reduced spending."},
    {"name": "Fed Funds Rate", "series_id": "FEDFUNDS", "unit": "%", "default_weight": 15, "description": "High rates can dampen borrowing and growth."},
    {"name": "S&P 500", "series_id": "SP500", "unit": "points", "default_weight": 10, "description": "Falling stock prices often precede downturns."}
]

# --- User-defined Weights ---
st.sidebar.header("⚖️ Adjust Category Weights (Total = 100%)")
weights = {}
total_weight = 0
for cat in categories:
    w = st.sidebar.slider(f"{cat['name']} Weight (%)", 0, 100, cat["default_weight"])
    weights[cat["name"]] = w
    total_weight += w
if total_weight != 100:
    st.sidebar.error(f"Weights must total 100%. Current total: {total_weight}%")
    st.stop()

# --- FRED Fetch ---
def fetch_series(series_id, start="2000-01-01"):
    url = f"{FRED_BASE}?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json&observation_start={start}"
    r = requests.get(url).json()
    return pd.DataFrame([
        {"date": obs["date"], "value": float(obs["value"])}
        for obs in r.get("observations", []) if obs["value"] != "."
    ])

# --- Build Master DataFrame ---
df_all = pd.DataFrame()
df_raw = {}
for cat in categories:
    df = fetch_series(cat["series_id"])
    df["date"] = pd.to_datetime(df["date"])
    df.rename(columns={"value": cat["name"]}, inplace=True)
    df_raw[cat["name"]] = df
    df_all = df if df_all.empty else pd.merge(df_all, df, on="date", how="inner")
df_all.sort_values("date", inplace=True)

# --- Apply Rule-Based Forecast Model ---
def rule_based_score(row):
    return sum([
        (weights["Yield Curve"] / 100) * (0.6 if row["Yield Curve"] < 0 else 0.3),
        (weights["Unemployment Rate"] / 100) * (0.6 if row["Unemployment Rate"] > 4 else 0.4),
        (weights["Leading Index"] / 100) * (0.6 if row["Leading Index"] < 0 else 0.4),
        (weights["Consumer Sentiment"] / 100) * (0.4 if row["Consumer Sentiment"] > 70 else 0.6),
        (weights["Fed Funds Rate"] / 100) * (0.5 if row["Fed Funds Rate"] > 4 else 0.3),
        (weights["S&P 500"] / 100) * (0.4 if row["S&P 500"] > 4000 else 0.6)
    ])

df_all["forecast"] = df_all.apply(rule_based_score, axis=1)
df_all["weight_sum"] = sum(weights.values()) / 100

# --- Current Forecast ---
latest_prob = df_all["forecast"].iloc[-1]
st.subheader("📊 Recession Probability Forecast")
st.metric("Current Probability", f"{latest_prob:.1%}")

# --- Recession Probability Chart (last 2 years+) ---
st.markdown("### 📈 Recession Probability Over Time")
df_recent = df_all[df_all["date"] >= (df_all["date"].max() - pd.DateOffset(months=24))]
fig_prob = px.line(df_recent, x="date", y="forecast", title="Recession Probability - Past 2+ Years",
                   labels={"forecast": "Probability"})
for r in RECESSIONS:
    fig_prob.add_vrect(x0=r[0], x1=r[1], fillcolor="gray", opacity=0.15, line_width=0)
st.plotly_chart(fig_prob, use_container_width=True)

# --- Aggregated Score Breakdown ---
st.subheader("📊 Weighted Contribution by Category")
latest_row = df_all.iloc[-1]
breakdown_data = []
for cat in categories:
    name = cat["name"]
    score = (
        0.6 if name == "Yield Curve" and latest_row[name] < 0 else
        0.6 if name == "Unemployment Rate" and latest_row[name] > 4 else
        0.6 if name == "Leading Index" and latest_row[name] < 0 else
        0.6 if name == "Consumer Sentiment" and latest_row[name] <= 70 else
        0.5 if name == "Fed Funds Rate" and latest_row[name] > 4 else
        0.6 if name == "S&P 500" and latest_row[name] <= 4000 else
        0.4 if name in ["Unemployment Rate", "Leading Index"] else 0.3
    )
    wt = weights[name] / 100
    breakdown_data.append({"Category": name, "Score": score, "Weight": wt, "Weighted Score": score * wt})

df_breakdown = pd.DataFrame(breakdown_data)
fig_bar = px.bar(df_breakdown, x="Category", y="Weighted Score",
                 color="Weighted Score", color_continuous_scale="RdYlGn_r",
                 title="Current Weighted Risk Score by Category")
st.plotly_chart(fig_bar, use_container_width=True)

# --- Component Line Charts (2000+ if available) ---
st.subheader("📈 Indicator Trends Since 2000")
for cat in categories:
    df = df_raw[cat["name"]]
    fig = px.line(df, x="date", y=cat["name"], title=f"{cat['name']} ({cat['unit']})")
    for r in RECESSIONS:
        fig.add_vrect(x0=r[0], x1=r[1], fillcolor="gray", opacity=0.1, line_width=0)
    st.plotly_chart(fig, use_container_width=True)

# --- CSV Download + Preview (last 2 years) ---
st.subheader("📥 Downloadable Forecast Input Data")
export = df_all[df_all["date"] >= (df_all["date"].max() - pd.DateOffset(months=24))].copy()
export["Current Forecast"] = export["forecast"]
export["Total Weight"] = export["weight_sum"]
st.dataframe(export.tail(12).reset_index(drop=True))
csv = export.to_csv(index=False)
st.download_button("Download Full Forecast Input (CSV)", csv, "recession_input_and_score.csv")
