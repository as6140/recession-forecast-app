# app.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from datetime import datetime

# --- CONFIG ---
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
indicators = [
    # Original 6
    {"name": "Yield Curve", "series_id": "T10Y3M", "unit": "%", "default_weight": 15,
     "description": "Difference between 10Y and 3M Treasury yields. Negative = recession warning."},
    {"name": "Unemployment Rate", "series_id": "UNRATE", "unit": "%", "default_weight": 10,
     "description": "Rising unemployment often precedes recessions."},
    {"name": "Consumer Sentiment", "series_id": "UMCSENT", "unit": "index", "default_weight": 10,
     "description": "Low household confidence signals weak consumption."},
    {"name": "Fed Funds Rate", "series_id": "FEDFUNDS", "unit": "%", "default_weight": 5,
     "description": "Higher rates may restrict growth."},
    {"name": "S&P 500", "series_id": "SP500", "unit": "points", "default_weight": 5,
     "description": "Falling equity prices often precede slowdowns."},
    {"name": "CFNAI", "series_id": "CFNAI", "unit": "index", "default_weight": 10,
     "description": "Chicago Fed National Activity Index. < –0.70 = recession signal."},

    # New 8
    {"name": "Jobless Claims", "series_id": "ICSA", "unit": "claims", "default_weight": 10,
     "description": "Initial claims for unemployment insurance."},
    {"name": "Real PCE", "series_id": "PCEC96", "unit": "billion $", "default_weight": 5,
     "description": "Inflation-adjusted personal consumption."},
    {"name": "Industrial Production", "series_id": "INDPRO", "unit": "index", "default_weight": 5,
     "description": "Total US output of goods and materials."},
    {"name": "ISM PMI", "series_id": "NAPM", "unit": "index", "default_weight": 5,
     "description": "Manufacturing PMI. < 50 = contraction."},
    {"name": "Retail Sales", "series_id": "RSXFS", "unit": "billion $", "default_weight": 5,
     "description": "Real retail sales excluding autos."},
    {"name": "Housing Starts", "series_id": "HOUST", "unit": "thousands", "default_weight": 3,
     "description": "New privately-owned housing units started."},
    {"name": "Bond Spread", "series_id": "BAA10Y", "unit": "%", "default_weight": 5,
     "description": "Baa corporate bond yield minus 10Y Treasury."},
    {"name": "Financial Conditions", "series_id": "NFCI", "unit": "index", "default_weight": 2,
     "description": "Chicago Fed Financial Conditions Index. > 0 = tighter conditions."}
]

# --- Weight Inputs ---
st.sidebar.header("⚖️ Indicator Weights (must total 100%)")
weights = {}
total_weight = 0
for ind in indicators:
    w = st.sidebar.number_input(f"{ind['name']} (%)", min_value=0, max_value=100, value=ind["default_weight"], step=1)
    weights[ind["name"]] = w
    total_weight += w

if total_weight != 100:
    st.sidebar.error(f"Total weight must be 100%. Current: {total_weight}%")
    st.stop()

# --- FRED Fetch ---
@st.cache_data
def fetch_series(series_id, start="2000-01-01"):
    url = f"{FRED_BASE}?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json&observation_start={start}"
    r = requests.get(url).json()
    return pd.DataFrame([
        {"date": obs["date"], "value": float(obs["value"])}
        for obs in r.get("observations", []) if obs["value"] != "."
    ])

# --- Build Master Data ---
df_all = pd.DataFrame()
df_raw = {}
for ind in indicators:
    df = fetch_series(ind["series_id"])
    df["date"] = pd.to_datetime(df["date"])
    df.rename(columns={"value": ind["name"]}, inplace=True)
    df_raw[ind["name"]] = df
    df_all = df if df_all.empty else pd.merge(df_all, df, on="date", how="inner")
df_all.sort_values("date", inplace=True)

# --- Scoring Model ---
def score_indicator(name, value):
    if name == "Yield Curve":
        return 0.6 if value < 0 else 0.3
    if name == "Unemployment Rate":
        return 0.6 if value > 4 else 0.4
    if name == "Consumer Sentiment":
        return 0.6 if value < 70 else 0.4
    if name == "Fed Funds Rate":
        return 0.5 if value > 4 else 0.3
    if name == "S&P 500":
        return 0.6 if value < 4000 else 0.4
    if name == "CFNAI":
        return 0.6 if value < -0.7 else (0.3 if value > 0 else 0.4)
    if name == "Jobless Claims":
        return 0.6 if value > 300000 else 0.3
    if name == "Real PCE":
        return 0.6 if value < 14000 else 0.3
    if name == "Industrial Production":
        return 0.6 if value < 100 else 0.4
    if name == "ISM PMI":
        return 0.6 if value < 47 else 0.4
    if name == "Retail Sales":
        return 0.6 if value < 500 else 0.4
    if name == "Housing Starts":
        return 0.6 if value < 1000 else 0.3
    if name == "Bond Spread":
        return 0.6 if value > 2.5 else 0.3
    if name == "Financial Conditions":
        return 0.6 if value > 0 else 0.4
    return 0.4

def apply_forecast(row):
    score = 0
    for ind in indicators:
        name = ind["name"]
        s = score_indicator(name, row[name])
        score += (weights[name] / 100) * s
    return score

df_all["forecast"] = df_all.apply(apply_forecast, axis=1)

# --- Display Current Probability ---
latest_prob = df_all["forecast"].iloc[-1]
st.subheader("📊 Recession Probability Forecast")
st.metric("Current Probability", f"{latest_prob:.1%}")

# --- Forecast Time Series Chart (2Y+) ---
st.markdown("### 📈 Recession Probability Over Time")
df_recent = df_all[df_all["date"] >= (df_all["date"].max() - pd.DateOffset(months=24))]
fig_prob = px.line(df_recent, x="date", y="forecast", title="Forecast (Last 2 Years)",
                   labels={"forecast": "Probability"})
for r in RECESSIONS:
    fig_prob.add_vrect(x0=r[0], x1=r[1], fillcolor="gray", opacity=0.1, line_width=0)
st.plotly_chart(fig_prob, use_container_width=True)

# --- Component Charts ---
st.subheader("📊 Indicator Trends")
for ind in indicators:
    df = df_raw[ind["name"]]
    fig = px.line(df, x="date", y=ind["name"], title=f"{ind['name']} ({ind['unit']})")
    for r in RECESSIONS:
        fig.add_vrect(x0=r[0], x1=r[1], fillcolor="gray", opacity=0.1, line_width=0)
    st.plotly_chart(fig, use_container_width=True)

# --- CSV Preview + Download (2Y+, incl. weights) ---
st.subheader("📥 Download Forecast Dataset")
export = df_all[df_all["date"] >= (df_all["date"].max() - pd.DateOffset(months=24))].copy()
export["Weight Sum"] = sum(weights.values()) / 100
export["Forecast"] = export["forecast"]
st.dataframe(export.tail(12).reset_index(drop=True))
csv = export.to_csv(index=False)
st.download_button("Download CSV (Full Inputs & Forecast)", csv, "recession_inputs_forecast.csv")
