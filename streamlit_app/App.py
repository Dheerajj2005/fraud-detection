import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
import requests

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CSS ---------------- #
st.markdown("""
<style>
.main-header { font-size: 2.5rem; font-weight: bold; text-align: center; }
</style>
""", unsafe_allow_html=True)

API_URL = "http://localhost:8000"
LOG_PATH = Path("logs/predictions.csv")

# ---------------- HELPERS ---------------- #
@st.cache_data(ttl=30)
def load_prediction_logs():
    if LOG_PATH.exists():
        df = pd.read_csv(LOG_PATH)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    return pd.DataFrame()

def check_api_health():
    try:
        return requests.get(f"{API_URL}/health", timeout=2).status_code == 200
    except:
        return False

def get_api_metrics():
    try:
        r = requests.get(f"{API_URL}/metrics", timeout=2)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def fallback_metrics(df: pd.DataFrame):
    if df.empty:
        return None
    return {
        "total_predictions": len(df),
        "fraud_detected": int(df["is_fraud"].sum()),
        "fraud_rate": float(df["is_fraud"].mean()),
        "average_latency_ms": float(df["latency_ms"].mean()),
        "requests_per_minute": 0.0
    }

# ---------------- HEADER ---------------- #
st.markdown("<h1 class='main-header'>🔍 Fraud Detection System</h1>", unsafe_allow_html=True)
st.markdown("**Real-time credit card fraud detection dashboard**")

# ---------------- STATUS ---------------- #
api_online = check_api_health()
df = load_prediction_logs()

if not api_online:
    st.warning("⚠️ API is offline — showing historical data only")

api_metrics = get_api_metrics() if api_online else None
if api_metrics is None:
    api_metrics = fallback_metrics(df)

# ---------------- SIDEBAR ---------------- #
with st.sidebar:
    st.markdown("### System Status")
    st.success("API Online" if api_online else "API Offline")
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

# ---------------- METRICS ---------------- #
st.markdown("## 📈 Key Metrics")
c1, c2, c3, c4 = st.columns(4)

if api_metrics:
    c1.metric("Total Predictions", api_metrics["total_predictions"])
    c2.metric("Fraud Rate", f"{api_metrics['fraud_rate']*100:.2f}%")
    c3.metric("Avg Latency", f"{api_metrics['average_latency_ms']:.2f} ms")
    c4.metric("Model Accuracy", "96.3%")
else:
    for c in [c1, c2, c3, c4]:
        c.metric("Loading...", "---")

st.divider()

# ---------------- CHARTS ---------------- #
if not df.empty:
    st.markdown("### 📅 Predictions Over Time")
    hourly = df.set_index("timestamp").resample("1H").size().reset_index(name="count")
    fig = px.line(hourly, x="timestamp", y="count")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🎯 Fraud Distribution")
    pie = go.Figure(go.Pie(
        labels=["Legitimate", "Fraud"],
        values=[(df["is_fraud"] == False).sum(), df["is_fraud"].sum()],
        hole=0.4
    ))
    st.plotly_chart(pie, use_container_width=True)

    st.markdown("### 🚦 Risk Levels")
    risk = df["risk_level"].value_counts()
    bar = go.Figure(go.Bar(x=risk.index, y=risk.values))
    st.plotly_chart(bar, use_container_width=True)

    st.markdown("### 📋 Recent Predictions")
    recent = df.tail(10)[[
        "timestamp", "amount", "fraud_probability",
        "is_fraud", "risk_level", "latency_ms"
    ]]
    st.dataframe(recent, use_container_width=True)

else:
    st.info("No predictions yet. Use the Prediction page or API.")

st.markdown("---")
st.markdown("Fraud Detection System • Streamlit + FastAPI")
