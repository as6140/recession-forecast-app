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
     "description": "The yield curve measures the difference between long-term and short-term interest rates. "
                    "An inverted curve (negative value) often signals a coming recession."},
    {"name": "Unemployment Rate", "series_id": "UNRATE", "unit": "%", "default_weight": 15,
     "description": "The share of the labor force that is jobless. Rising unemployment tends to reflect economic weakening."},
    {"name": "Leading Index", "series_id": "USSLIND", "unit": "index", "default_weight": 20,
     "description": "A composite of leading indicators compiled by the Conference Board. Persistent declines signal potential recessions."},
    {"name": "Consumer Sentiment", "series_id": "UMCSENT", "unit": "index", "default_weight": 15,
     "description": "Reflects household confidence in the economy. Sharp drops often precede downturns."},
    {"name": "Fed Funds Rate", "series_id": "FEDFUNDS", "unit": "%", "default_weight": 15,
     "description": "The Federal Reserve's main policy rate. High rates can tighten financial conditions and slow growth."},
    {"name": "S&P 500", "series_id": "SP500", "unit": "points", "default_weight": 10,
     "description": "The US stock market index. Falling prices or volatility can reflect economic pessimism."}
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

# --- Fetch Data ---
def fetch_series(series_id, start="2018-01-01"):
    url = f"{FRED_BASE}?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json&observation_start={start}"
    r = requests.get(url).json()
    return pd.DataFrame([
        {"date": obs["date"], "value": float(obs["value"])}
        for obs in r.get("observations", []) if obs["value"] != "."
    ])

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

# --- Recession Forecast Over Time ---
@st.cache_data
def compute_forecast_history(df_all, weights):
    rows = []
    for i in range(len(df_all)):
        row = df_all.iloc[i]
        scores = {
            "Yield Curve": 0.6 if row["Yield Curve"] < 0 else 0.3,
            "Unemployment Rate": 0.6 if row["Unemployment Rate"] > 4 else 0.4,
            "Leading Index": 0.6 if row["Leading Index"] < 0 else 0.4,
            "Consumer Sentiment": 0.4 if row["Consumer Sentiment"] > 70 else 0.6,
            "Fed Funds Rate": 0.5 if row["Fed Funds Rate"] > 4 else 0.3,
            "S&P 500": 0.4 if row["S&P 500"] > 4000 else 0.6
        }
        weighted = sum((weights[k] / 100) * scores[k] for k in weights)
        rows.append({"date": row["date"], "forecast": weighted})
    return pd.DataFrame(rows)

forecast_df = compute_forecast_history(df_all, weights)
latest_prob = forecast_df["forecast"].iloc[-1]

# --- Forecast Output ---
st.subheader("📊 Recession Probability Forecast")
st.metric("Current Probability", f"{latest_prob:.1%}")
st.markdown(f"**Trend over time:**")

# --- Forecast History Line Chart ---
fig_prob = px.line(forecast_df, x="date", y="forecast", title="Recession Probability Over Time",
                   labels={"forecast": "Probability"})
for r in RECESSIONS:
    fig_prob.add_vrect(x0=r[0], x1=r[1], fillcolor="gray", opacity=0.15, line_width=0)
st.plotly_chart(fig_prob, use_container_width=True)

# --- Aggregated Score Chart ---
scores_latest = {
    "Yield Curve": 0.6 if df_all.iloc[-1]["Yield Curve"] < 0 else 0.3,
    "Unemployment Rate": 0.6 if df_all.iloc[-1]["Unemployment Rate"] > 4 else 0.4,
    "Leading Index": 0.6 if df_all.iloc[-1]["Leading Index"] < 0 else 0.4,
    "Consumer Sentiment": 0.4 if df_all.iloc[-1]["Consumer Sentiment"] > 70 else 0.6,
    "Fed Funds Rate": 0.5 if df_all.iloc[-1]["Fed Funds Rate"] > 4 else 0.3,
    "S&P 500": 0.4 if df_all.iloc[-1]["S&P 500"] > 4000 else 0.6
}
st.subheader("📊 Weighted Contribution by Category")
breakdown_df = pd.DataFrame({
    "Category": list(scores_latest.keys()),
    "Score": [scores_latest[k] for k in scores_latest],
    "Weight": [weights[k] / 100 for k in weights]
})
breakdown_df["Weighted Score"] = breakdown_df["Score"] * breakdown_df["Weight"]
fig_bar = px.bar(breakdown_df, x="Category", y="Weighted Score",
                 title="Weighted Score by Category",
                 color="Weighted Score", color_continuous_scale="RdYlGn_r")
st.plotly_chart(fig_bar, use_container_width=True)

# --- Metric Interpretations ---
st.subheader("🔍 Metric Interpretations")
for cat in categories:
    series = df_raw[cat["name"]]
    latest_val = series[cat["name"]].iloc[-1]
    delta = latest_val - series[cat["name"]].iloc[-4]
    direction = "increasing" if delta > 0 else "decreasing"
    if cat["name"] == "Yield Curve":
        note = "Inverted yield curves (below 0%) have preceded every U.S. recession since the 1970s."
    elif cat["name"] == "Unemployment Rate":
        note = "Recessions often follow unemployment rates rising more than 0.5% from recent lows."
    elif cat["name"] == "Leading Index":
        note = "A negative LEI has consistently preceded recessions since 1960."
    elif cat["name"] == "Consumer Sentiment":
        note = "Recessions often occur when sentiment drops below 70."
    elif cat["name"] == "Fed Funds Rate":
        note = "High rates can restrain economic activity and slow hiring/investment."
    elif cat["name"] == "S&P 500":
        note = "Sharp market declines and volatility often precede downturns."
    st.markdown(f"**{cat['name']}** ({latest_val:.2f} {cat['unit']}, {direction}): {cat['description']} {note}")

# --- Line Charts (No colored backgrounds) ---
st.subheader("📈 Component Trends")
for cat in categories:
    df = df_raw[cat["name"]]
    fig = px.line(df, x="date", y=cat["name"], title=f"{cat['name']} ({cat['unit']})",
                  labels={cat["name"]: cat["name"]})
    for r in RECESSIONS:
        fig.add_vrect(x0=r[0], x1=r[1], fillcolor="gray", opacity=0.1, line_width=0)
    st.plotly_chart(fig, use_container_width=True)

# --- Bottom CSV Download ---
st.subheader("📥 Downloadable Score Breakdown")
st.dataframe(breakdown_df.style.format({
    "Score": "{:.2f}",
    "Weight": "{:.0%}",
    "Weighted Score": "{:.2%}"
}))
csv = breakdown_df.to_csv(index=False)
st.download_button("Download Score Breakdown CSV", csv, "recession_score_breakdown.csv")
