import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from datetime import datetime

# Title
st.title("US Recession Probability Forecast (Next 12 Months)")

# Define categories and weights
categories = [
    {"category": "Yield Curve & Credit", "weight": 0.25},
    {"category": "Labor Market", "weight": 0.20},
    {"category": "Leading Indicators", "weight": 0.20},
    {"category": "Consumer & Retail", "weight": 0.15},
    {"category": "Fed Policy & Rates", "weight": 0.10},
    {"category": "Market Sentiment", "weight": 0.10},
]

# FRED API setup
FRED_API_KEY = st.secrets["FRED_API_KEY"] if "FRED_API_KEY" in st.secrets else "YOUR_API_KEY_HERE"
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# FRED series mapping (latest data)
fred_series = {
    "10Y-3M Spread": "T10Y3M",
    "Unemployment Rate": "UNRATE",
    "LEI Index": "USSLIND",
    "Consumer Confidence": "UMCSENT",
    "Fed Funds Rate": "FEDFUNDS",
    "S&P 500 Index": "SP500"
}

def fetch_latest_fred_value(series_id):
    url = f"{FRED_BASE}?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json"
    try:
        response = requests.get(url)
        data = response.json()
        observations = data.get("observations", [])
        for obs in reversed(observations):
            if obs["value"] != ".":
                return float(obs["value"])
    except:
        return None

# Fetch and normalize scores based on economic logic
scores = {}

spread = fetch_latest_fred_value("T10Y3M")
scores["Yield Curve & Credit"] = 0.6 if spread and spread < 0 else 0.3

unemployment = fetch_latest_fred_value("UNRATE")
scores["Labor Market"] = 0.6 if unemployment and unemployment > 4 else 0.4

lei = fetch_latest_fred_value("USSLIND")
scores["Leading Indicators"] = 0.6 if lei and lei < 0 else 0.4

confidence = fetch_latest_fred_value("UMCSENT")
scores["Consumer & Retail"] = 0.4 if confidence and confidence > 70 else 0.6

fedfunds = fetch_latest_fred_value("FEDFUNDS")
scores["Fed Policy & Rates"] = 0.5 if fedfunds and fedfunds > 4 else 0.3

sp500 = fetch_latest_fred_value("SP500")
scores["Market Sentiment"] = 0.4 if sp500 and sp500 > 4000 else 0.6

# Build DataFrame
df = pd.DataFrame(categories)
df["score"] = df["category"].map(scores)
df["weighted_score"] = df["weight"] * df["score"]
total_prob = df["weighted_score"].sum()

# Display current time
st.markdown(f"_Updated: {datetime.now().strftime('%B %d, %Y')}_")

# Show table
st.subheader("Forecast Breakdown")
st.dataframe(df.style.format({"weight": "{:.0%}", "score": "{:.0%}", "weighted_score": "{:.1%}"}))

# Show total probability
st.subheader("Total Forecast Probability")
st.metric("Recession Probability (Next 12 Months)", f"{total_prob:.1%}")

# Interpretation
st.markdown("""
### Interpretation
- A probability above 50% suggests recession risks are elevated.
- This forecast combines leading indicators, labor data, policy stance, and market signals.
- Inputs are updated automatically using FRED.
""")

# Visualizations
st.subheader("Economic Indicators Overview")
chart_data = pd.DataFrame({"Indicator": list(scores.keys()), "Score": list(scores.values())})
st.bar_chart(chart_data.set_index("Indicator"))

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Powered by FRED API | Auto-updating with live economic data")
