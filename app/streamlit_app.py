"""AQI dashboard — full data + charts, cloud-only, cached for speed."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="London AQI Forecast", layout="wide", initial_sidebar_state="collapsed")

from src import config
from src.feature_pipeline.feature_store import fetch_features
from src.inference_pipeline.predict import daily_summary, predict_next_72h


# ============================================================================
# AQI category
# ============================================================================
def aqi_band(aqi: float) -> tuple[str, str]:
    a = float(aqi)
    if a <= 50:
        return "Good", "#2ecc71"
    if a <= 100:
        return "Moderate", "#f1c40f"
    if a <= 150:
        return "Unhealthy (Sensitive)", "#e67e22"
    if a <= 200:
        return "Unhealthy", "#e74c3c"
    if a <= 300:
        return "Very Unhealthy", "#9b59b6"
    return "Hazardous", "#7d3c98"


# ============================================================================
# CACHE: login once, fetch+forecast cached per city (10 min TTL)
# ============================================================================
@st.cache_resource
def _warm_login():
    import hopsworks
    return hopsworks.login(
        api_key_value=config.HOPSWORKS_API_KEY,
        project=config.HOPSWORKS_PROJECT or None,
    )


@st.cache_data(ttl=600, show_spinner=False)
def cached_features(city: str) -> pd.DataFrame:
    df = fetch_features(city, limit=200)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data(ttl=600, show_spinner=False)
def cached_forecast(city: str) -> pd.DataFrame:
    fc = predict_next_72h(city)
    fc["timestamp"] = pd.to_datetime(fc["timestamp"])
    return fc


try:
    _warm_login()
except Exception:
    pass


# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
<style>
  #MainMenu, header, footer {visibility: hidden;}
  .stApp {background:#ffffff;}
  .block-container {padding-top:2.5rem; max-width:1100px;}
  [data-testid="stMetricValue"] {font-weight:600;}
  h1,h2,h3 {font-weight:600; letter-spacing:-0.01em;}
</style>
""", unsafe_allow_html=True)

st.title("London Air Quality")
st.caption("AQI + 72-hour forecast")
city = config.DEFAULT_CITY

st.divider()

# ============================================================================
# DATA
# ============================================================================
try:
    with st.spinner(f"Loading {city.title()} data..."):
        features = cached_features(city)
except Exception as e:
    st.error(f"Feature load failed: {e}")
    st.stop()

if features.empty:
    st.warning(f"No cloud data for {city.title()}. Run feature pipeline.")
    st.stop()

hist = features.dropna(subset=["aqi"]).copy()
last_aqi = float(hist.iloc[-1]["aqi"])
last_ts = pd.to_datetime(hist.iloc[-1]["timestamp"])
label, color = aqi_band(last_aqi)

# ============================================================================
# CURRENT METRICS
# ============================================================================
m1, m2, m3, m4 = st.columns(4)
m1.metric("Current AQI", f"{last_aqi:.0f}")
m2.metric("Category", label)
m3.metric("Data Points", f"{len(hist)}")
m4.metric("Last Update", last_ts.strftime("%b %d, %H:%M"))

# ============================================================================
# FORECAST
# ============================================================================
try:
    with st.spinner("Generating 72h forecast..."):
        fc = cached_forecast(city)
except Exception as e:
    st.error(f"Forecast failed: {e}")
    st.stop()

# ---- Two separate charts: history + forecast ----
g1, g2 = st.columns(2)

def _minimal(fig):
    fig.update_layout(
        height=360, margin=dict(l=10, r=10, t=10, b=10),
        hovermode="x unified", template="plotly_white",
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(color="#111111"), showlegend=False,
        yaxis_title="AQI", xaxis_title=None,
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0", zeroline=False)
    return fig


with g1:
    st.subheader("History")
    hfig = go.Figure()
    hfig.add_trace(go.Scatter(
        x=hist["timestamp"], y=hist["aqi"],
        mode="lines", line=dict(color="#111111", width=1.4),
    ))
    st.plotly_chart(_minimal(hfig), use_container_width=True)

with g2:
    st.subheader("72h Forecast")
    ffig = go.Figure()
    ffig.add_trace(go.Scatter(
        x=fc["timestamp"], y=fc["upper_bound"],
        line=dict(width=0), hoverinfo="skip",
    ))
    ffig.add_trace(go.Scatter(
        x=fc["timestamp"], y=fc["lower_bound"],
        fill="tonexty", fillcolor="rgba(0,0,0,0.06)",
        line=dict(width=0), hoverinfo="skip",
    ))
    ffig.add_trace(go.Scatter(
        x=fc["timestamp"], y=fc["predicted_aqi"],
        mode="lines", line=dict(color="#111111", width=1.4),
    ))
    st.plotly_chart(_minimal(ffig), use_container_width=True)

# ---- Daily summary ----
st.subheader("3-Day Summary")
daily = daily_summary(fc)
dc = st.columns(len(daily))
for i, (_, row) in enumerate(daily.iterrows()):
    lbl, clr = aqi_band(row["mean"])
    dc[i].metric(
        f"{row['date']}",
        f"{row['mean']:.0f}",
        f"{row['min']:.0f}–{row['max']:.0f} range",
    )

# ---- Full forecast table ----
st.subheader("Full Forecast (72 hours)")
table = fc[["timestamp", "predicted_aqi", "lower_bound", "upper_bound", "model_used"]].rename(
    columns={
        "timestamp": "Time", "predicted_aqi": "AQI",
        "lower_bound": "CI Low", "upper_bound": "CI High", "model_used": "Model",
    }
)
st.dataframe(table, use_container_width=True, height=420)

# ---- Recent history table ----
with st.expander("Recent Observed History"):
    h = hist[["timestamp", "aqi"]].tail(48).rename(
        columns={"timestamp": "Time", "aqi": "AQI"}
    )
    st.dataframe(h, use_container_width=True, height=300)

st.caption(f"{fc['model_used'].iloc[0]} · Hopsworks")
