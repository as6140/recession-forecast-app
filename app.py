import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from datetime import datetime, timedelta

# Title
st.title("US Recession Probability Forecast (Next 12 Months)")

# Define categories and weights with explanations and insights
categories = [
    {"category": "Yield Curve & Credit", "weight": 0.25, "series_id": "T10Y3M", "explanation": "Inverted yield curves often signal future recessions.", "insight": "Currently, the yield curve remains inverted, consistent with prior pre-recessionary environments."},
    {"category": "Labor Market", "weight": 0.20, "series_id": "UNRATE", "explanation": "Rising unemployment can be an early recession signal.", "insight": "Unemployment has ticked up slightly, though not yet above recession-warning thresholds."},
    {"category": "Leading Indicators", "weight": 0.20, "series_id": "USSLIND", "explanation": "Composite index used to forecast future economic activity.", "insight": "The LEI index has declined for several months, a pattern seen before previous downturns."},
    {"category": "Consumer & Retail", "weight": 0.15, "series_id": "UMCSENT", "explanation": "Consumer sentiment is a driver of consumption patterns.", "insight": "Sentiment remains below long-term averages but has stabilized recently."},
    {"category": "Fed Policy & Rates", "weight": 0.10, "series_id": "FEDFUNDS", "explanation": "Tight monetary policy can slow economic growth.", "insight": "The Fed is holding rates steady, but financial conditions remain tight."},
    {"category": "Market Sentiment", "weight": 0.10, "series_id": "SP500", "explanation": "Equity trends reflect forward-looking investor confidence.", "insight": "Stock markets are near highs, though driven largely by tech sector concentration."},
]

# Add full NBER recessions for backtest shading
recessions = [
    ("2001-03-01", "2001-11-30"),
    ("2007-12-01", "2009-06-30"),
    ("2020-02-01", "2020-04-30")
]

# Historical narrative logic
def generate_narrative(prob):
    if prob > 0.65:
        return "🔴 Recession risk is high — multiple leading indicators suggest a likely downturn."
    elif prob > 0.5:
        return "🟠 Recession risk is elevated — several warning signals are present."
    elif prob > 0.35:
        return "🟡 Risk is moderate — economic signals are mixed but stable."
    else:
        return "🟢 Recession risk appears low — most indicators remain supportive."

# Sidebar Threshold Sensitivity
st.sidebar.header("Threshold Sensitivity")
thresholds = {
    "Yield Curve & Credit": st.sidebar.slider("10Y-3M Spread Threshold", -2.0, 2.0, 0.0, 0.1),
    "Labor Market": st.sidebar.slider("Unemployment Rate Threshold", 3.0, 6.0, 4.0, 0.1),
    "Leading Indicators": st.sidebar.slider("LEI Threshold", -2.0, 2.0, 0.0, 0.1),
    "Consumer & Retail": st.sidebar.slider("Consumer Sentiment Threshold", 50, 100, 70, 1),
    "Fed Policy & Rates": st.sidebar.slider("Fed Funds Rate Threshold", 0.0, 8.0, 4.0, 0.25),
    "Market Sentiment": st.sidebar.slider("S&P 500 Threshold", 2000, 6000, 4000, 100),
}

# FRED API setup
FRED_API_KEY = st.secrets["FRED_API_KEY"] if "FRED_API_KEY" in st.secrets else "YOUR_API_KEY_HERE"
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# Function to fetch latest value from FRED
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

# Function to fetch historical data for line charts
def fetch_fred_timeseries(series_id, start_date):
    url = f"{FRED_BASE}?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json&observation_start={start_date}"
    try:
        response = requests.get(url)
        data = response.json()
        records = [
            {"date": obs["date"], "value": float(obs["value"])}
            for obs in data.get("observations", []) if obs["value"] != "."
        ]
        return pd.DataFrame(records)
    except:
        return pd.DataFrame(columns=["date", "value"])

# Calculate recession scores dynamically
scores = {}
for cat in categories:
    val = fetch_latest_fred_value(cat["series_id"])
    t = thresholds[cat["category"]]
    if cat["category"] == "Yield Curve & Credit":
        scores[cat["category"]] = 0.6 if val is not None and val < t else 0.3
    elif cat["category"] == "Labor Market":
        scores[cat["category"]] = 0.6 if val is not None and val > t else 0.4
    elif cat["category"] == "Leading Indicators":
        scores[cat["category"]] = 0.6 if val is not None and val < t else 0.4
    elif cat["category"] == "Consumer & Retail":
        scores[cat["category"]] = 0.4 if val is not None and val > t else 0.6
    elif cat["category"] == "Fed Policy & Rates":
        scores[cat["category"]] = 0.5 if val is not None and val > t else 0.3
    elif cat["category"] == "Market Sentiment":
        scores[cat["category"]] = 0.4 if val is not None and val > t else 0.6

# Build DataFrame
df = pd.DataFrame(categories)
df["score"] = df["category"].map(scores)
df["weighted_score"] = df["weight"] * df["score"]
total_prob = df["weighted_score"].sum()

# Display current time
st.markdown(f"_Updated: {datetime.now().strftime('%B %d, %Y')}_")

# Show table
st.subheader("Forecast Breakdown")
df_display = df[["category", "weight", "score", "weighted_score"]].style.format({"weight": "{:.0%}", "score": "{:.0%}", "weighted_score": "{:.1%}"})
st.dataframe(df_display)

# Show total probability with narrative
st.subheader("Total Forecast Probability")
st.metric("Recession Probability (Next 12 Months)", f"{total_prob:.1%}")
st.markdown(generate_narrative(total_prob))

# Visualizations - Bar Chart with color
st.subheader("Economic Indicators Overview")
chart_data = pd.DataFrame({"Indicator": list(scores.keys()), "Score": list(scores.values())})
fig_bar = px.bar(chart_data, x="Indicator", y="Score", color="Score", color_continuous_scale="RdYlGn_r")
st.plotly_chart(fig_bar, use_container_width=True)

# Line Charts for Historical Trends with Recession Shading
st.subheader("2-Year Historical Trends of Each Indicator")
start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

for cat in categories:
    ts_df = fetch_fred_timeseries(cat["series_id"], start_date)
    if not ts_df.empty:
        ts_df["date"] = pd.to_datetime(ts_df["date"])
        fig = px.line(ts_df, x="date", y="value", title=cat["category"], markers=True)
        for r_start, r_end in recessions:
            fig.add_vrect(x0=r_start, x1=r_end, fillcolor="gray", opacity=0.2, line_width=0)
        fig.update_layout(hoverlabel=dict(bgcolor="white"))
        fig.update_traces(mode="lines+markers", hovertemplate="%{y:.2f}")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"{cat['explanation']}\n\n**Insight:** {cat['insight']}")

# Option to download the forecast table
csv = df.to_csv(index=False)
st.download_button("Download Forecast Data (CSV)", csv, "recession_forecast.csv")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Powered by FRED API | Interactive and auto-updating economic forecasting with enhanced visuals, historical context, and narrative intelligence")
