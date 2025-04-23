# app.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from datetime import datetime, timedelta
import io
import feedparser
from bs4 import BeautifulSoup
import re

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
    {"name": "Yield Curve", "series_id": "T10Y3M", "unit": "%", "default_weight": 20,
     "trend_window": "1M",
     "trend_desc": lambda old, new: f"{'Inverted' if new < 0 else 'Normal'} yield curve at {new:.2f}%. {'Higher' if new > old else 'Lower'} than {old:.2f}% one month ago. " + 
     f"{'The inverted yield curve strongly signals recession risk, as this pattern has preceded every recession since 1955.' if new < 0 else 'The positive yield curve suggests normal economic conditions, though the spread remains tight.'}"},
    
    {"name": "Unemployment Rate", "series_id": "UNRATE", "unit": "%", "default_weight": 15,
     "trend_window": "3M",
     "trend_desc": lambda old, new: f"Currently {new:.1f}%, {'up' if new > old else 'down'} from {old:.1f}% three months ago. " +
     f"{'Rising unemployment often precedes recessions, and this increase signals growing labor market stress.' if new > old else 'The stable/falling unemployment rate suggests continued labor market strength, reducing near-term recession risk.'}"},
    
    {"name": "CFNAI", "series_id": "CFNAI", "unit": "index", "default_weight": 15,
     "trend_window": "3M",
     "trend_desc": lambda old, new: f"At {new:.2f}, {'improved' if new > old else 'deteriorated'} from {old:.2f} three months ago. " +
     f"{'A reading below -0.7 strongly signals recession risk, while deteriorating values suggest broadening economic weakness.' if new < -0.7 else 'Current levels suggest normal economic growth, though momentum bears watching.'}"},
    
    {"name": "Jobless Claims", "series_id": "ICSA", "unit": "claims", "default_weight": 12,
     "trend_window": "1M",
     "trend_desc": lambda old, new: f"Currently {new:,.0f} claims, {'up' if new > old else 'down'} from {old:,.0f} a month ago. " +
     f"{'Rising claims often precede broader job market weakness and signal growing recession risk.' if new > old else 'Stable/falling claims suggest continued labor market resilience.'}"},
    
    {"name": "ISM PMI", "series_id": "IPMAN", "unit": "index", "default_weight": 10,
     "trend_window": "3M",
     "trend_desc": lambda old, new: f"At {new:.1f}, {'improved' if new > old else 'declined'} from {old:.1f} three months ago. " +
     f"{'Manufacturing contraction (below 50) often precedes broader economic weakness.' if new < 50 else 'Expansion territory suggests continued industrial sector growth.'}"},
    
    {"name": "Consumer Sentiment", "series_id": "UMCSENT", "unit": "index", "default_weight": 8,
     "trend_window": "3M",
     "trend_desc": lambda old, new: f"At {new:.1f}, {'improved' if new > old else 'declined'} from {old:.1f} three months ago. " +
     f"{'Low consumer confidence often precedes reduced spending and economic contraction.' if new < 70 else 'Confident consumers typically support continued economic expansion.'}"},
    
    {"name": "Bond Spread", "series_id": "BAA10Y", "unit": "%", "default_weight": 8,
     "trend_window": "1M",
     "trend_desc": lambda old, new: f"Spread at {new:.2f}%, {'widened' if new > old else 'tightened'} from {old:.2f}% a month ago. " +
     f"{'Widening spreads signal increasing credit stress and higher recession risk.' if new > old else 'Stable/tightening spreads suggest healthy credit markets.'}"},
    
    {"name": "Fed Funds Rate", "series_id": "FEDFUNDS", "unit": "%", "default_weight": 7,
     "trend_window": "3M",
     "trend_desc": lambda old, new: f"Currently at {new:.2f}%, {'up' if new > old else 'down'} from {old:.2f}% three months ago. " +
     f"{'High rates typically restrict economic activity and historically precede recessions.' if new > 5 else 'Current policy stance balances growth and inflation concerns.'}"},
    
    {"name": "Real PCE", "series_id": "PCEC96", "unit": "billion $", "default_weight": 5,
     "trend_window": "3M",
     "trend_desc": lambda old, new: f"At ${new:,.0f}B, {'grew' if new > old else 'fell'} from ${old:,.0f}B three months ago. " +
     f"{'Declining real consumption often precedes broader economic contraction.' if new < old else 'Growing consumption supports continued economic expansion.'}"},
    
    {"name": "Bank Consensus", "series_id": "BANK_CONSENSUS", "unit": "%", "default_weight": 0,
     "trend_window": "1M",
     "trend_desc": lambda old, new: f"Major banks see {new:.1f}% recession probability, {'up' if new > old else 'down'} from {old:.1f}% last month. " +
     f"{'Rising consensus suggests growing concern among institutional forecasters.' if new > old else 'Stable/falling consensus suggests easing recession concerns among major banks.'}"},
    
    {"name": "Housing Starts", "series_id": "HOUST", "unit": "thousands", "default_weight": 0,
     "trend_window": "3M",
     "trend_desc": lambda old, new: f"Currently {new:,.0f}K units, {'up' if new > old else 'down'} from {old:,.0f}K three months ago. " +
     f"{'Housing weakness often precedes broader economic slowdowns.' if new < old else 'Housing strength typically supports continued economic growth.'}"},
    
    {"name": "S&P 500", "series_id": "SP500", "unit": "points", "default_weight": 0,
     "trend_window": "3M",
     "trend_desc": lambda old, new: f"At {new:,.0f}, {'up' if new > old else 'down'} from {old:,.0f} three months ago. " +
     f"{'Market declines often precede economic weakness as investors price in future risks.' if new < old else 'Rising markets suggest investor confidence in continued growth.'}"}
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

@st.cache_data(ttl=6*3600)
def fetch_bank_consensus():
    """Fetch and process bank recession probabilities from free news sources."""
    try:
        # Initialize empty dictionary to store latest probabilities by bank
        bank_probabilities = {
            'JPM': {'value': None, 'source': None, 'date': None},
            'GS': {'value': None, 'source': None, 'date': None},
            'MS': {'value': None, 'source': None, 'date': None},
            'BOFA': {'value': None, 'source': None, 'date': None},
            'CITI': {'value': None, 'source': None, 'date': None}
        }
        
        # List of free RSS feeds to check
        rss_feeds = [
            {"url": "https://news.google.com/rss/search?q=bank+recession+probability+forecast", "name": "Google News"},
            {"url": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=JPM,GS,MS,BAC,C&region=US&lang=en-US", "name": "Yahoo Finance"}
        ]
        
        # Search last 30 days of articles
        cutoff_date = datetime.now() - timedelta(days=30)
        
        # Track sources for transparency
        sources_found = []
        
        for feed in rss_feeds:
            try:
                feed_data = feedparser.parse(feed["url"])
                sources_found.append(f"[{feed['name']}]({feed['url']})")
                
                for entry in feed_data.entries:
                    pub_date = datetime(*entry.published_parsed[:6])
                    if pub_date < cutoff_date:
                        continue
                        
                    # Combine title and description for searching
                    text = f"{entry.title.lower()} {entry.description.lower()}"
                    
                    # Look for bank-specific probabilities
                    for bank in bank_probabilities.keys():
                        if bank.lower() in text and bank_probabilities[bank]['value'] is None:
                            probability_matches = re.findall(
                                rf"{bank.lower()}.*?(\d{{1,3}})%?\s*(?:chance|probability|likelihood|risk)\s*(?:of\s*)?recession",
                                text
                            )
                            
                            if probability_matches:
                                prob = int(probability_matches[0])
                                if 0 <= prob <= 100:
                                    bank_probabilities[bank]['value'] = prob
                                    bank_probabilities[bank]['source'] = entry.link
                                    bank_probabilities[bank]['date'] = pub_date
            except Exception as e:
                continue
        
        # Filter out None values and calculate average
        valid_probs = [b['value'] for b in bank_probabilities.values() if b['value'] is not None]
        if not valid_probs:
            st.sidebar.warning("⚠️ No recent bank forecasts found")
            return pd.DataFrame({
                'date': [datetime.now()],
                'value': [0.0]
            })
            
        avg_prob = sum(valid_probs) / len(valid_probs)
        
        # Create DataFrame with single row for current date
        df = pd.DataFrame({
            'date': [datetime.now()],
            'value': [avg_prob]
        })
        
        # Add source info and links to sidebar
        st.sidebar.markdown("#### Bank Forecast Sources")
        
        # Show RSS feed sources
        st.sidebar.markdown("**Data aggregated from:**")
        for source in sources_found:
            st.sidebar.markdown(f"- {source}")
            
        # Show individual bank forecasts with sources
        st.sidebar.markdown("\n**Latest Bank Forecasts:**")
        for bank, data in bank_probabilities.items():
            if data['value'] is not None:
                date_str = data['date'].strftime('%Y-%m-%d')
                st.sidebar.markdown(
                    f"- {bank}: {data['value']}% "
                    f"([source]({data['source']}) • {date_str})"
                )
        
        return df

    except Exception as e:
        st.sidebar.warning("⚠️ Could not fetch bank consensus data")
        return pd.DataFrame({
            'date': [datetime.now()],
            'value': [0.0]
        })

# --- Load Available Indicators ---
def get_latest_value(df, target_date):
    """Get most recent value not later than target_date"""
    mask = df["date"] <= target_date
    if mask.any():
        return df[mask].iloc[-1]["value"]
    return None

# Calculate date range
end_date = pd.Timestamp.now().normalize().replace(day=1) - pd.DateOffset(days=1)  # Last day of previous month
start_date = end_date - pd.DateOffset(months=24)

df_all = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq='ME'), columns=['date'])  # Changed from 'M' to 'ME'
df_raw = {}  # Initialize df_raw dictionary
indicators = []
weights = {}

for ind in indicators_all:
    name = ind["name"]
    if ind["series_id"] == "BANK_CONSENSUS":
        df = fetch_bank_consensus()
    else:
        df = fetch_series(ind["series_id"])
        
    if df is not None and "date" in df.columns:
        # Create series with values for each month
        monthly_values = []
        for date in df_all["date"]:
            val = get_latest_value(df, date)
            if val is not None:
                monthly_values.append(val)
            else:
                monthly_values.append(None)
        
        df_all[name] = monthly_values
        indicators.append(ind)
        
        if not any(pd.isna(monthly_values)):  # Only include if we have some data
            df_raw[name] = df
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
    score = 0.01  # Minimum probability
    if name == "Yield Curve": score = 0.6 if value < 0 else 0.3
    if name == "Unemployment Rate": score = 0.6 if value > 4 else 0.4
    if name == "Consumer Sentiment": score = 0.6 if value < 70 else 0.4
    if name == "Fed Funds Rate": score = 0.5 if value > 4 else 0.3
    if name == "S&P 500": score = 0.6 if value < 4000 else 0.4
    if name == "CFNAI": score = 0.6 if value < -0.7 else (0.3 if value > 0 else 0.4)
    if name == "Jobless Claims": score = 0.6 if value > 300000 else 0.3
    if name == "Real PCE": score = 0.6 if value < 14000 else 0.3
    if name == "Industrial Production": score = 0.6 if value < 100 else 0.4
    if name == "ISM PMI": score = 0.6 if value < 47 else 0.4
    if name == "Retail Sales": score = 0.6 if value < 500 else 0.4
    if name == "Housing Starts": score = 0.6 if value < 1000 else 0.3
    if name == "Bond Spread": score = 0.6 if value > 2.5 else 0.3
    if name == "Financial Conditions": score = 0.6 if value > 0 else 0.4
    if name == "Bank Consensus": 
        return value / 100  # Convert percentage to probability score
    return min(max(score, 0.01), 0.99)  # Ensure between 1% and 99%

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
    name = ind["name"]
    df = df_raw[name].copy()  # Make a copy to avoid modifying original
    
    # Keep original 'value' column for calculations but add named column for display
    df[name] = df['value']
    current_val = df['value'].iloc[-1]
    
    # Get historical value for trend analysis
    window_map = {"1M": 1, "3M": 3, "6M": 6}
    trend_months = window_map[ind["trend_window"]]
    trend_date = df["date"].max() - pd.DateOffset(months=trend_months)
    trend_val = df[df["date"] >= trend_date]['value'].iloc[0]
    
    # Create trend description using appropriate window
    trend_text = ind["trend_desc"](trend_val, current_val)
    
    # Plot full history since 2000
    fig = px.line(df, x="date", y="value", title=f"{name} ({ind['unit']}) - Historical Trend Since 2000")
    
    # Add recession shading
    for r in RECESSIONS:
        fig.add_vrect(x0=r[0], x1=r[1], fillcolor="gray", opacity=0.1, line_width=0)
    
    # Highlight trend window
    fig.add_vrect(
        x0=trend_date,
        x1=df["date"].max(),
        fillcolor="rgba(255, 255, 0, 0.1)",  # Light yellow
        line_width=0,
        annotation_text="Trend Window",
        annotation_position="top left"
    )
    
    # Improve hover info
    fig.update_traces(
        hovertemplate=(
            "<b>Date</b>: %{x|%Y-%m-%d}<br>" +
            f"<b>{name}</b>: %{{y:.2f}} {ind['unit']}<br>"
        )
    )
    
    # Update y-axis title
    fig.update_layout(
        yaxis_title=f"{name} ({ind['unit']})"
    )
    
    # Show the chart and interpretation
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"""
    **Current Reading:** {current_val:.2f} {ind['unit']}  
    **Trend Analysis ({ind['trend_window']}):** {trend_text}
    """)
    st.markdown("---")  # Add separator between indicators

# --- Breakdown Chart ---
st.subheader("📊 Contribution by Category")
latest = df_all.iloc[-1]
breakdown = []
for ind in indicators:
    name = ind["name"]
    if name in latest:  # Add check to ensure column exists
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
st.subheader("📥 Forecast Data")
df_export = df_all[df_all["date"] >= (df_all["date"].max() - pd.DateOffset(months=24))].copy()

# Add individual scores and weights
for ind in indicators:
    name = ind["name"]
    if name in df_export.columns:
        df_export[f"{name}_Score"] = df_export[name].apply(lambda x: score_indicator(name, x))
        df_export[f"{name}_Weight"] = weights[name]
        df_export[f"{name}_Weighted"] = df_export[f"{name}_Score"] * (weights[name] / 100)

# Rename forecast column
df_export["Recession_Probability"] = df_export["forecast"]
df_export = df_export.drop("forecast", axis=1)

st.dataframe(df_export.tail(12).reset_index(drop=True))
st.download_button("Download CSV", df_export.to_csv(index=False), "recession_forecast_data.csv")

# Add Glossary
st.markdown("""
### 📚 Economic Indicator Glossary

#### Yield Curve (10Y-3M Treasury Spread)
The difference between 10-year and 3-month Treasury yields. When short-term rates exceed long-term rates (negative spread), it signals that markets expect economic weakness. Yield curve inversions have preceded every recession since 1955, typically by 12-18 months.

#### Unemployment Rate
The percentage of the labor force actively seeking work. Unemployment typically starts rising 3-6 months before a recession begins. A sudden increase of 0.3-0.4 percentage points is an early warning sign.

#### CFNAI (Chicago Fed National Activity Index)
A composite index of 85 monthly indicators covering production, income, employment, consumption, and sales. Values below -0.7 (after averaging three months) have reliably signaled recessions. The index is designed to have a mean of zero and standard deviation of one.

#### Initial Jobless Claims
Weekly new applications for unemployment insurance. A leading indicator that rises before broader unemployment. Sustained increases above 300,000 claims or sharp spikes of 15-20% signal labor market stress.

#### ISM Manufacturing PMI
Survey-based index where readings above 50 indicate expansion, below 50 indicate contraction. Manufacturing tends to lead the broader economy. Readings below 45 for several months typically precede recessions.

#### Consumer Sentiment
University of Michigan survey measuring consumer attitudes about the economy. Sharp declines often precede reduced spending and economic contraction. Readings below 70 historically align with recession risks.

#### Corporate Bond Spread
The difference between BAA-rated corporate bond yields and 10-year Treasuries. Widening spreads indicate higher perceived credit risk. Spreads above 2.5-3% often signal financial stress preceding recessions.

#### Federal Funds Rate
The Fed's primary policy rate affecting borrowing costs throughout the economy. High rates can restrict credit and economic activity. Most recessions follow Fed tightening cycles, though with variable lags.

#### Real Personal Consumption Expenditures
Inflation-adjusted consumer spending, representing about 70% of GDP. Declining real consumption, especially in discretionary categories, often precedes broader economic contraction.

#### Bank Consensus
Average recession probability forecast from major US investment banks (JPMorgan, Goldman Sachs, Morgan Stanley, Bank of America, and Citi). Banks update these forecasts periodically based on their economic research. While methodologies vary by bank, these forecasts typically consider similar indicators to our model but may include proprietary data and analyst judgment.

Key aspects of bank forecasts:
- Usually focus on 12-month forward probability
- Updated monthly or when significant economic developments occur
- May reflect both quantitative models and qualitative analysis
- Consider broader factors like policy changes and global events

### 📊 Using These Indicators

The most reliable recession signals typically combine:
1. Inverted yield curve (negative 10Y-3M spread)
2. Rising unemployment or jobless claims
3. Declining CFNAI below -0.7
4. Manufacturing PMI below 45
5. Widening credit spreads

No single indicator is perfect - the combination of signals matters most. The weighted model attempts to balance these various inputs while accounting for their historical predictive power.
""")
