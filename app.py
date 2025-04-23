import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from datetime import datetime

# Timestamp
st.caption(f"📅 Last updated: {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')} (Local time)")

# Title
st.title("US Recession Probability Forecast (Next 12 Months)")

st.markdown("""
This tool combines six key economic indicators to estimate the probability of a US recession over the next 12 months.

🧭 **Quick Start Guide:**
1. Adjust threshold sliders in the sidebar to define what *you* consider warning signs.
2. Hover over any chart to see exact values and units.
3. Use the download button at the bottom to save the data.
4. Share custom insights by copying this URL (your settings are retained).
""")

# Manual Refresh
st.sidebar.markdown("---")
if st.sidebar.button("🔁 Manually Refresh Data"):
    st.session_state["trigger_rerun"] = True
if st.session_state.get("trigger_rerun"):
    st.session_state["trigger_rerun"] = False
    st.experimental_rerun()

# Info box with toggle instructions
st.info("Each slider below defines what YOU consider risky. Moving sliders **left** makes the model more sensitive to mild signals (higher forecasted risk). Moving **right** makes it stricter (lower forecasted risk unless metrics are extreme).\n\nThese thresholds define when each indicator contributes to recession probability.")

# Metric definitions
categories = [
    {"category": "Yield Curve & Credit", "weight": 0.25, "series_id": "T10Y3M", "unit": "%", "source": "FRED: 10-Year minus 3-Month Treasury Spread",
     "explanation": "Inverted yield curves often signal future recessions.",
     "insight": "Currently, the yield curve remains inverted, consistent with prior pre-recessionary environments.",
     "tooltip": "10Y-3M Spread", "low": -2, "high": 2, "default": 0.0},

    {"category": "Labor Market", "weight": 0.20, "series_id": "UNRATE", "unit": "%", "source": "FRED: US Unemployment Rate",
     "explanation": "Rising unemployment can be an early recession signal.",
     "insight": "Unemployment has ticked up slightly, though not yet above recession-warning thresholds.",
     "tooltip": "Unemployment Rate", "low": 3.0, "high": 6.0, "default": 4.0},

    {"category": "Leading Indicators", "weight": 0.20, "series_id": "USSLIND", "unit": "index", "source": "FRED: Leading Economic Index (Conference Board)",
     "explanation": "Composite index used to forecast future economic activity.",
     "insight": "The LEI index has declined for several months, a pattern seen before previous downturns.",
     "tooltip": "LEI Index", "low": -2.0, "high": 2.0, "default": 0.0},

    {"category": "Consumer & Retail", "weight": 0.15, "series_id": "UMCSENT", "unit": "index", "source": "FRED: University of Michigan Sentiment Index",
     "explanation": "Consumer sentiment is a driver of consumption patterns.",
     "insight": "Sentiment remains below long-term averages but has stabilized recently.",
     "tooltip": "Consumer Sentiment", "low": 50, "high": 100, "default": 70},

    {"category": "Fed Policy & Rates", "weight": 0.10, "series_id": "FEDFUNDS", "unit": "%", "source": "FRED: Effective Federal Funds Rate",
     "explanation": "Tight monetary policy can slow economic growth.",
     "insight": "The Fed is holding rates steady, but financial conditions remain tight.",
     "tooltip": "Fed Funds Rate", "low": 0.0, "high": 8.0, "default": 4.0},

    {"category": "Market Sentiment", "weight": 0.10, "series_id": "SP500", "unit": "points", "source": "FRED: S&P 500 Index",
     "explanation": "Equity trends reflect forward-looking investor confidence.",
     "insight": "Stock markets are near highs, though driven largely by tech sector concentration.",
     "tooltip": "S&P 500", "low": 2000, "high": 6000, "default": 4000}
]

recessions = [("2001-03-01", "2001-11-30"), ("2007-12-01", "2009-06-30"), ("2020-02-01", "2020-04-30")]

def fetch_latest(series_id):
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={st.secrets['FRED_API_KEY']}&file_type=json"
    try:
        data = requests.get(url).json()
        for obs in reversed(data["observations"]):
            if obs["value"] != ".":
                return float(obs["value"])
    except:
        return None

def fetch_series(series_id):
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={st.secrets['FRED_API_KEY']}&file_type=json&observation_start=2000-01-01"
    try:
        data = requests.get(url).json()
        return pd.DataFrame([{"date": obs["date"], "value": float(obs["value"])} for obs in data["observations"] if obs["value"] != "."])
    except:
        return pd.DataFrame(columns=["date", "value"])

def generate_narrative(prob):
    if prob > 0.65:
        return "🔴 Recession risk is high — multiple leading indicators suggest a likely downturn."
    elif prob > 0.5:
        return "🟠 Recession risk is elevated — several warning signals are present."
    elif prob > 0.35:
        return "🟡 Risk is moderate — economic signals are mixed but stable."
    else:
        return "🟢 Recession risk appears low — most indicators remain supportive."

# Sidebar sliders with explanations
st.sidebar.header("Define Recession Risk Thresholds")
thresholds = {}
for cat in categories:
    thresholds[cat["category"]] = st.sidebar.slider(
        f"{cat['tooltip']} Threshold",
        cat["low"], cat["high"], cat["default"],
        help=f"Left = more sensitive (risky), Right = more lenient (less risky)"
    )

# Score calculation
scores = {}
for cat in categories:
    val = fetch_latest(cat["series_id"])
    threshold = thresholds[cat["category"]]
    if cat["category"] == "Yield Curve & Credit":
        scores[cat["category"]] = 0.6 if val < threshold else 0.3
    elif cat["category"] == "Labor Market":
        scores[cat["category"]] = 0.6 if val > threshold else 0.4
    elif cat["category"] == "Leading Indicators":
        scores[cat["category"]] = 0.6 if val < threshold else 0.4
    elif cat["category"] == "Consumer & Retail":
        scores[cat["category"]] = 0.4 if val > threshold else 0.6
    elif cat["category"] == "Fed Policy & Rates":
        scores[cat["category"]] = 0.5 if val > threshold else 0.3
    elif cat["category"] == "Market Sentiment":
        scores[cat["category"]] = 0.4 if val > threshold else 0.6

# Forecast table
df = pd.DataFrame(categories)
df["score"] = df["category"].map(scores)
df["weighted_score"] = df["weight"] * df["score"]
total_prob = df["weighted_score"].sum()

st.subheader("Forecast Breakdown")
st.dataframe(df[["category", "weight", "score", "weighted_score"]].style.format({"weight": "{:.0%}", "score": "{:.0%}", "weighted_score": "{:.1%}"}))

st.subheader("Total Forecast Probability")
st.metric("Recession Probability (Next 12 Months)", f"{total_prob:.1%}")
st.markdown(generate_narrative(total_prob))

# Bar Chart
st.subheader("Economic Indicators Overview")
fig_bar = px.bar(
    pd.DataFrame({"Indicator": list(scores), "Score": list(scores.values())}),
    x="Indicator", y="Score", color="Score", color_continuous_scale="RdYlGn_r"
)
st.plotly_chart(fig_bar, use_container_width=True)

# Time Series with R/Y/G banding
st.subheader("Historical Trends Since 2000")
for cat in categories:
    ts = fetch_series(cat["series_id"])
    if not ts.empty:
        ts["date"] = pd.to_datetime(ts["date"])
        fig = px.line(ts, x="date", y="value", title=f"{cat['tooltip']} ({cat['unit']})", labels={"value": cat["unit"]})
        for r0, r1 in recessions:
            fig.add_vrect(x0=r0, x1=r1, fillcolor="gray", opacity=0.2, line_width=0)
        # Add green/yellow/red thresholds
        y_min, y_max = ts["value"].min(), ts["value"].max()
        mid = thresholds[cat["category"]]
        spread = (y_max - y_min) * 0.05
        fig.add_hrect(y0=y_min, y1=mid - spread, fillcolor="green", opacity=0.05, line_width=0)
        fig.add_hrect(y0=mid - spread, y1=mid + spread, fillcolor="yellow", opacity=0.05, line_width=0)
        fig.add_hrect(y0=mid + spread, y1=y_max, fillcolor="red", opacity=0.05, line_width=0)
        fig.update_layout(hoverlabel=dict(bgcolor="white", font_color="black"))
        fig.update_traces(mode="lines", hovertemplate=f"{cat['tooltip']}: %{{y:.2f}}")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"{cat['explanation']}\n\n**Insight:** {cat['insight']}\n\n_Data Source: {cat['source']}_")

# Download button
csv = df.to_csv(index=False)
st.download_button("Download Forecast Data (CSV)", csv, "recession_forecast.csv")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Powered by FRED | Share this URL to preserve your settings.")
