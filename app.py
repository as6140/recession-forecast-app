# app.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from datetime import datetime

# --- CONFIG & HEADER ---
st.set_page_config(page_title="US Recession Probability", layout="wide")
st.title("🇺🇸 US Recession Probability Forecast (Next 12 Months)")
st.caption(f"📅 Last updated: {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}")
st.markdown("Created by [Alex Shropshire](https://www.linkedin.com/in/alexandershropshire/) • [GitHub](https://github.com/as6140)")

# --- REFRESH ---
if st.sidebar.button("🔁 Manually Refresh Data"):
    st.session_state["refresh"] = True
if st.session_state.get("refresh"):
    st.session_state["refresh"] = False
    st.experimental_rerun()

# --- FRED Setup ---
FRED_API_KEY = st.secrets["FRED_API_KEY"]
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
RECESSIONS = [("2001-03-01", "2001-11-30"), ("2007-12-01", "2009-06-30"), ("2020-02-01", "2020-04-30")]

# --- Indicator Definitions ---
indicators_all = [
    {"name": "Yield Curve", "series_id": "T10Y3M", "unit": "%", "default_weight": 15,
     "description": "10Y minus 3M Treasury yields. Inversion = recession signal."},
    {"name": "Unemployment Rate", "series_id": "UNRATE", "unit": "%", "default_weight": 10,
     "description": "Rising unemployment often precedes recessions."},
    {"name": "Consumer Sentiment", "series_id": "UMCSENT", "unit": "index", "default_weight": 10,
     "description": "Low sentiment = weak consumption confidence."},
    {"name": "Fed Funds Rate", "series_id": "FEDFUNDS", "unit": "%", "default_weight": 5,
     "description": "Higher rates tighten financial conditions."},
    {"name": "S&P 500", "series_id": "SP500", "unit": "points", "default_weight": 5,
     "description": "Falling equity prices reflect market pessimism."},
    {"name": "CFNAI", "series_id": "CFNAI", "unit": "index", "default_weight": 10,
     "description": "Chicago Fed National Activity Index. < -0.70 = recession risk."},
    {"name": "Jobless Claims", "series_id": "ICSA", "unit": "claims", "default_weight": 10,
     "description": "Initial claims for unemployment insurance. Spike = early stress."},
    {"name": "Real PCE", "series_id": "PCEC96", "unit": "billion $", "default_weight": 5,
     "description": "Inflation-adjusted consumer spending."},
    {"name": "Industrial Production", "series_id": "INDPRO", "unit": "index", "default_weight": 5,
     "description": "Physical output of factories and utilities."},
    {"name": "ISM PMI", "series_id": "NAPM", "unit": "index", "default_weight": 5,
     "description": "Manufacturing activity. < 50 = contraction."},
    {"name": "Retail Sales", "series_id": "RSXFS", "unit": "billion $", "default_weight": 5,
     "description": "Inflation-adjusted sales, excluding autos."},
    {"name": "Housing Starts", "series_id": "HOUST", "unit": "thousands", "default_weight": 3,
     "description": "New private residential construction."},
    {"name": "Bond Spread", "series_id": "BAA10Y", "unit": "%", "default_weight": 5,
     "description": "Corporate risk premium. >2.5% = financial stress."},
    {"name": "Financial Conditions", "series_id": "NFCI", "unit": "index", "default_weight": 2,
     "description": "Liquidity, credit, and risk sentiment index."}
]

# --- Fetch Series ---
@st.cache_data
def fetch_series(series_id, start="2000-01-01"):
    url = f"{FRED_BASE}?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json&observation_start={start}"
    try:
        r = requests.get(url).json()
        data = [{"date": obs["date"], "value": float(obs["value"])}
                for obs in r.get("observations", []) if obs["value"] != "."]
        if not data:
            return None
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception:
        return None

# --- Load Available Indicators ---
df_all = pd.DataFrame()
df_raw = {}
indicators = []
weights = {}
for ind in indicators_all:
    df = fetch_series(ind["series_id"])
    if df is not None and "date" in df.columns:
        df.rename(columns={"value": ind["name"]}, inplace=True)
        df_raw[ind["name"]] = df
        indicators.append(ind)
        df_all = df if df_all.empty else pd.merge(df_all, df, on="date", how="inner")
    else:
        st.sidebar.warning(f"⚠️ Skipped: {ind['name']} — no data")

# --- Rebalance weights to sum to 100% ---
total_default = sum(ind["default_weight"] for ind in indicators)
for ind in indicators:
    rebased = round(ind["default_weight"] * 100 / total_default)
    weights[ind["name"]] = st.sidebar.number_input(f"{ind['name']} (%)", min_value=0, max_value=100, value=rebased, step=1)
if sum(weights.values()) != 100:
    st.sidebar.error(f"❌ Weights must total 100%. Current: {sum(weights.values())}%")
    st.stop()

# --- Score Function ---
def score_indicator(name, value):
    if name == "Yield Curve": return 0.6 if value < 0 else 0.3
    if name == "Unemployment Rate": return 0.6 if value > 4 else 0.4
    if name == "Consumer Sentiment": return 0.6 if value < 70 else 0.4
    if name == "Fed Funds Rate": return 0.5 if value > 4 else 0.3
    if name == "S&P 500": return 0.6 if value < 4000 else 0.4
    if name == "CFNAI": return 0.6 if value < -0.7 else (0.3 if value > 0 else 0.4)
    if name == "Jobless Claims": return 0.6 if value > 300000 else 0.3
    if name == "Real PCE": return 0.6 if value < 14000 else 0.3
    if name == "Industrial Production": return 0.6 if value < 100 else 0.4
    if name == "ISM PMI": return 0.6 if value < 47 else 0.4
    if name == "Retail Sales": return 0.6 if value < 500 else 0.4
    if name == "Housing Starts": return 0.6 if value < 1000 else 0.3
    if name == "Bond Spread": return 0.6 if value > 2.5 else 0.3
    if name == "Financial Conditions": return 0.6 if value > 0 else 0.4
    return 0.4

def forecast_score(row):
    score = 0
    for ind in indicators:
        name = ind["name"]
        if name in row and weights[name] > 0:
            score += (weights[name] / 100) * score_indicator(name, row[name])
    return score

df_all["forecast"] = df_all.apply(forecast_score, axis=1)

# --- Headline Probability ---
latest_prob = df_all["forecast"].iloc[-1]
st.subheader("📊 Recession Probability Forecast")
st.metric("Current Probability", f"{latest_prob:.1%}")

# --- Forecast Chart (2 years) ---
st.markdown("### 📈 Recession Probability Over Time")
df_recent = df_all[df_all["date"] >= (df_all["date"].max() - pd.DateOffset(months=24))]
fig_prob = px.line(df_recent, x="date", y="forecast", title="Forecast (Last 2+ Years)",
                   labels={"forecast": "Probability"})
for r in RECESSIONS:
    fig_prob.add_vrect(x0=r[0], x1=r[1], fillcolor="gray", opacity=0.1, line_width=0)
st.plotly_chart(fig_prob, use_container_width=True)

# --- Component Charts + Interpretations ---
st.subheader("📊 Indicator Trends & Interpretations")
for ind in indicators:
    df = df_raw[ind["name"]]
    val = df[ind["name"]].iloc[-1]
    trend = "↑" if val > df[ind["name"]].iloc[-4] else "↓"
    fig = px.line(df, x="date", y=ind["name"], title=f"{ind['name']} ({ind['unit']})")
    for r in RECESSIONS:
        fig.add_vrect(x0=r[0], x1=r[1], fillcolor="gray", opacity=0.1, line_width=0)
    fig.update_traces(hovertemplate=f"%{{x}}<br><b>{ind['name']}</b>: %{{y:.2f}}")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"**{ind['name']}** ({val:.2f} {ind['unit']} {trend}): {ind['description']}")

# --- Breakdown Chart ---
st.subheader("📊 Contribution by Category")
latest = df_all.iloc[-1]
breakdown = []
for ind in indicators:
    name = ind["name"]
    raw_score = score_indicator(name, latest[name])
    wt = weights[name] / 100
    breakdown.append({
        "Category": name,
        "Score": raw_score,
        "Weight": wt,
        "Weighted Score": raw_score * wt
    })
df_breakdown = pd.DataFrame(breakdown)
fig_bar = px.bar(df_breakdown, x="Category", y="Weighted Score", color="Weighted Score",
                 color_continuous_scale="RdYlGn_r", title="Current Weighted Score by Category")
st.plotly_chart(fig_bar, use_container_width=True)

# --- CSV Preview + Download ---
st.subheader("📥 Forecast Data (2 Years)")
df_export = df_all[df_all["date"] >= (df_all["date"].max() - pd.DateOffset(months=24))].copy()
df_export["Forecast"] = df_export["forecast"]
st.dataframe(df_export.tail(12).reset_index(drop=True))
st.download_button("Download CSV", df_export.to_csv(index=False), "recession_forecast_data.csv")
