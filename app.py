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

# --- REFRESH BUTTON ---
if st.sidebar.button("🔁 Manually Refresh Data"):
    st.session_state["refresh"] = True
if st.session_state.get("refresh"):
    st.session_state["refresh"] = False
    st.experimental_rerun()

# --- FRED SETTINGS ---
FRED_API_KEY = st.secrets["FRED_API_KEY"]
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
RECESSIONS = [("2001-03-01", "2001-11-30"), ("2007-12-01", "2009-06-30"), ("2020-02-01", "2020-04-30")]

# --- INDICATOR DEFINITIONS ---
indicators = [
    {"name": "Yield Curve", "series_id": "T10Y3M", "unit": "%", "default_weight": 15,
     "description": "Difference between 10-year and 3-month Treasury yields. Inversions predict recessions."},
    {"name": "Unemployment Rate", "series_id": "UNRATE", "unit": "%", "default_weight": 10,
     "description": "A rising unemployment rate indicates labor market stress."},
    {"name": "Consumer Sentiment", "series_id": "UMCSENT", "unit": "index", "default_weight": 10,
     "description": "Low consumer confidence often precedes slowdowns."},
    {"name": "Fed Funds Rate", "series_id": "FEDFUNDS", "unit": "%", "default_weight": 5,
     "description": "High interest rates tighten financial conditions."},
    {"name": "S&P 500", "series_id": "SP500", "unit": "points", "default_weight": 5,
     "description": "Falling equity markets reflect investor pessimism."},
    {"name": "CFNAI", "series_id": "CFNAI", "unit": "index", "default_weight": 10,
     "description": "Chicago Fed National Activity Index. < –0.70 suggests recession risk."},
    {"name": "Jobless Claims", "series_id": "ICSA", "unit": "claims", "default_weight": 10,
     "description": "New filings for unemployment insurance. Rising claims = labor weakness."},
    {"name": "Real PCE", "series_id": "PCEC96", "unit": "billion $", "default_weight": 5,
     "description": "Inflation-adjusted personal spending. Declines warn of contraction."},
    {"name": "Industrial Production", "series_id": "INDPRO", "unit": "index", "default_weight": 5,
     "description": "Physical output of factories, mines, and utilities."},
    {"name": "ISM PMI", "series_id": "NAPM", "unit": "index", "default_weight": 5,
     "description": "Manufacturing survey. < 50 = contraction; < 47 = warning."},
    {"name": "Retail Sales", "series_id": "RSXFS", "unit": "billion $", "default_weight": 5,
     "description": "Inflation-adjusted retail sales excluding autos."},
    {"name": "Housing Starts", "series_id": "HOUST", "unit": "thousands", "default_weight": 3,
     "description": "Construction of new homes. A cyclical early signal."},
    {"name": "Bond Spread", "series_id": "BAA10Y", "unit": "%", "default_weight": 5,
     "description": "Corporate bond risk premium. > 2.5% = financial stress."},
    {"name": "Financial Conditions", "series_id": "NFCI", "unit": "index", "default_weight": 2,
     "description": "Chicago Fed index of liquidity, risk, credit conditions."}
]

# --- WEIGHT INPUTS ---
st.sidebar.header("⚖️ Indicator Weights (Total = 100%)")
weights = {}
total_weight = 0
for ind in indicators:
    w = st.sidebar.number_input(f"{ind['name']} (%)", min_value=0, max_value=100, value=ind["default_weight"], step=1)
    weights[ind["name"]] = w
    total_weight += w

if total_weight != 100:
    st.sidebar.error(f"Weights must total 100% (currently {total_weight}%)")
    st.stop()

# --- FETCH FRED SERIES ---
@st.cache_data
def fetch_series(series_id, start="2000-01-01"):
    url = f"{FRED_BASE}?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json&observation_start={start}"
    r = requests.get(url).json()
    return pd.DataFrame([
        {"date": obs["date"], "value": float(obs["value"])}
        for obs in r.get("observations", []) if obs["value"] != "."
    ])

# --- BUILD DATASET ---
df_all = pd.DataFrame()
df_raw = {}
for ind in indicators:
    df = fetch_series(ind["series_id"])
    df["date"] = pd.to_datetime(df["date"])
    df.rename(columns={"value": ind["name"]}, inplace=True)
    df_raw[ind["name"]] = df
    df_all = df if df_all.empty else pd.merge(df_all, df, on="date", how="inner")
df_all.sort_values("date", inplace=True)

# --- SCORING LOGIC ---
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
    return sum((weights[ind["name"]] / 100) * score_indicator(ind["name"], row[ind["name"]]) for ind in indicators)

df_all["forecast"] = df_all.apply(forecast_score, axis=1)

# --- HEADLINE FORECAST ---
latest_prob = df_all["forecast"].iloc[-1]
st.subheader("📊 Recession Probability Forecast")
st.metric("Current Probability", f"{latest_prob:.1%}")

# --- PROBABILITY CHART (2y+) ---
st.markdown("### 📈 Recession Probability Over Time")
df_recent = df_all[df_all["date"] >= (df_all["date"].max() - pd.DateOffset(months=24))]
fig_prob = px.line(df_recent, x="date", y="forecast", title="Forecast (Last 2+ Years)", labels={"forecast": "Probability"})
for r in RECESSIONS:
    fig_prob.add_vrect(x0=r[0], x1=r[1], fillcolor="gray", opacity=0.1, line_width=0)
st.plotly_chart(fig_prob, use_container_width=True)

# --- COMPONENT CHARTS + INTERPRETATIONS ---
st.subheader("📊 Indicator Trends & Interpretations")
for ind in indicators:
    df = df_raw[ind["name"]]
    val = df[ind["name"]].iloc[-1]
    prev = df[ind["name"]].iloc[-4]
    trend = "increasing" if val > prev else "decreasing"
    fig = px.line(df, x="date", y=ind["name"], title=f"{ind['name']} ({ind['unit']})", labels={ind["name"]: ind["name"]})
    for r in RECESSIONS:
        fig.add_vrect(x0=r[0], x1=r[1], fillcolor="gray", opacity=0.1, line_width=0)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"**{ind['name']}** ({val:.2f} {ind['unit']}, {trend}): {ind['description']}")

# --- AGGREGATED SCORE CHART ---
st.subheader("📊 Contribution by Category")
components = []
for ind in indicators:
    raw_score = score_indicator(ind["name"], df_all.iloc[-1][ind["name"]])
    wt = weights[ind["name"]] / 100
    components.append({
        "Category": ind["name"],
        "Score": raw_score,
        "Weight": wt,
        "Weighted Score": raw_score * wt
    })
df_breakdown = pd.DataFrame(components)
fig_bar = px.bar(df_breakdown, x="Category", y="Weighted Score", color="Weighted Score",
                 color_continuous_scale="RdYlGn_r", title="Weighted Risk Score by Category")
st.plotly_chart(fig_bar, use_container_width=True)

# --- CSV PREVIEW + DOWNLOAD ---
st.subheader("📥 Download Forecast Dataset")
df_export = df_all[df_all["date"] >= (df_all["date"].max() - pd.DateOffset(months=24))].copy()
df_export["Forecast"] = df_export["forecast"]
st.dataframe(df_export.tail(12).reset_index(drop=True))
csv = df_export.to_csv(index=False)
st.download_button("Download CSV", csv, "recession_forecast_data.csv")
