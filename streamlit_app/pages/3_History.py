"""
Prediction History page for viewing and analyzing past predictions.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path

st.set_page_config(page_title="Prediction History", page_icon="📜", layout="wide")

st.title("📜 Prediction History")
st.markdown("View and analyze historical fraud predictions")


# Load prediction logs
@st.cache_data(ttl=60)
def load_logs():
    """Load prediction logs."""
    try:
        log_path = Path("logs/predictions.csv")
        if log_path.exists():
            df = pd.read_csv(log_path)
            if len(df) > 0:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading logs: {e}")
        return pd.DataFrame()


df = load_logs()

required_cols = {
    "timestamp",
    "amount",
    "fraud_probability",
    "is_fraud",
    "risk_level",
    "latency_ms",
}

if not required_cols.issubset(df.columns):
    st.error("Prediction logs are missing required columns.")
    st.stop()


if df.empty:
    st.info("📭 No prediction history yet. Make some predictions to see them here!")
    st.stop()

# Filters
st.markdown("## 🔍 Filters")

col1, col2, col3, col4 = st.columns(4)

with col1:
    # Date range filter
    min_date = df["timestamp"].min().date()
    max_date = df["timestamp"].max().date()

    date_range = st.date_input(
        "Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date
    )

with col2:
    # Risk level filter
    risk_levels = st.multiselect(
        "Risk Level",
        ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        default=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
    )

with col3:
    # Fraud filter
    fraud_filter = st.selectbox(
        "Transaction Type", ["All", "Fraud Only", "Legitimate Only"]
    )

with col4:
    # Amount filter
    max_amount = float(df["amount"].max())
    amount_threshold = st.number_input(
        "Max Amount ($)", min_value=0.0, max_value=max_amount, value=max_amount
    )

# Apply filters
filtered_df = df.copy()

# Date filter
if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = filtered_df[
        (filtered_df["timestamp"].dt.date >= start_date)
        & (filtered_df["timestamp"].dt.date <= end_date)
    ]

# Risk level filter
filtered_df = filtered_df[filtered_df["risk_level"].isin(risk_levels)]

# Fraud filter
if fraud_filter == "Fraud Only":
    filtered_df = filtered_df[filtered_df["is_fraud"]]
elif fraud_filter == "Legitimate Only":
    filtered_df = filtered_df[~filtered_df["is_fraud"]]

# Amount filter
filtered_df = filtered_df[filtered_df["amount"] <= amount_threshold]

st.markdown(f"**Showing {len(filtered_df):,} of {len(df):,} predictions**")

st.markdown("---")

# Statistics
st.markdown("## 📊 Statistics")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Predictions", f"{len(filtered_df):,}")

with col2:
    fraud_count = filtered_df["is_fraud"].sum()
    st.metric("Frauds Detected", f"{fraud_count:,}")

with col3:
    fraud_rate = fraud_count / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
    st.metric("Fraud Rate", f"{fraud_rate:.2f}%")

with col4:
    avg_amount = filtered_df["amount"].mean()
    st.metric("Avg Amount", f"${avg_amount:.2f}")

with col5:
    avg_latency = filtered_df["latency_ms"].mean()
    st.metric("Avg Latency", f"{avg_latency:.2f}ms")

st.markdown("---")

# Visualizations
st.markdown("## 📈 Trends and Patterns")

# Time series
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Predictions Over Time")

    # Aggregate by hour or day depending on date range
    time_range = (filtered_df["timestamp"].max() - filtered_df["timestamp"].min()).days

    if time_range <= 7:
        freq = "H"
        title = "Predictions per Hour"
    else:
        freq = "D"
        title = "Predictions per Day"

    time_series = filtered_df.set_index("timestamp").resample(freq).size().reset_index()
    time_series.columns = ["Time", "Count"]

    fig = px.line(time_series, x="Time", y="Count", title=title)
    fig.update_traces(line_color="#1f77b4", line_width=2)
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### Fraud Rate Over Time")

    fraud_series = (
        filtered_df.set_index("timestamp")
        .resample(freq)
        .agg({"is_fraud": ["sum", "count"]})
    )
    fraud_series.columns = ["fraud_count", "total"]
    fraud_series["fraud_rate"] = (
        fraud_series["fraud_count"] / fraud_series["total"] * 100
    )
    fraud_series = fraud_series.reset_index()

    fig = px.line(
        fraud_series, x="timestamp", y="fraud_rate", title="Fraud Rate % Over Time"
    )
    fig.update_traces(line_color="#e74c3c", line_width=2)
    fig.update_yaxes(title_text="Fraud Rate (%)")
    st.plotly_chart(fig, use_container_width=True)

# Amount distribution
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Amount Distribution")

    fig = px.histogram(
        filtered_df,
        x="amount",
        color="is_fraud",
        nbins=50,
        title="Transaction Amount Distribution",
        labels={"amount": "Amount ($)", "is_fraud": "Is Fraud"},
        color_discrete_map={True: "#e74c3c", False: "#2ecc71"},
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### Risk Level Distribution")

    risk_counts = filtered_df["risk_level"].value_counts()

    colors = {
        "LOW": "#2ecc71",
        "MEDIUM": "#f39c12",
        "HIGH": "#e67e22",
        "CRITICAL": "#e74c3c",
    }

    fig = go.Figure(
        data=[
            go.Bar(
                x=risk_counts.index,
                y=risk_counts.values,
                marker_color=[
                    colors.get(level, "#95a5a6") for level in risk_counts.index
                ],
                text=risk_counts.values,
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        title="Transactions by Risk Level",
        xaxis_title="Risk Level",
        yaxis_title="Count",
    )

    st.plotly_chart(fig, use_container_width=True)

# Fraud probability distribution
st.markdown("### Fraud Probability Distribution")

fig = px.histogram(
    filtered_df,
    x="fraud_probability",
    color="is_fraud",
    nbins=50,
    title="Distribution of Fraud Probabilities",
    labels={"fraud_probability": "Fraud Probability", "is_fraud": "Actual Fraud"},
    color_discrete_map={True: "#e74c3c", False: "#2ecc71"},
)
fig.update_layout(bargap=0.1)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Hourly patterns
st.markdown("### 🕐 Hourly Patterns")

hourly_stats = filtered_df.copy()
hourly_stats["hour"] = hourly_stats["timestamp"].dt.hour

col1, col2 = st.columns(2)

with col1:
    hourly_volume = hourly_stats.groupby("hour").size().reset_index()
    hourly_volume.columns = ["Hour", "Count"]

    fig = px.bar(
        hourly_volume,
        x="Hour",
        y="Count",
        title="Transaction Volume by Hour of Day",
        labels={"Hour": "Hour of Day (0-23)", "Count": "Number of Transactions"},
    )
    fig.update_traces(marker_color="#1f77b4")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    hourly_fraud = (
        hourly_stats.groupby("hour")["is_fraud"].agg(["sum", "count"]).reset_index()
    )
    hourly_fraud["fraud_rate"] = hourly_fraud["sum"] / hourly_fraud["count"] * 100
    hourly_fraud.columns = ["Hour", "fraud_count", "total", "fraud_rate"]

    fig = px.line(
        hourly_fraud,
        x="Hour",
        y="fraud_rate",
        title="Fraud Rate by Hour of Day",
        labels={"Hour": "Hour of Day (0-23)", "fraud_rate": "Fraud Rate (%)"},
    )
    fig.update_traces(line_color="#e74c3c", line_width=3)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Data table
st.markdown("## 📋 Detailed Records")

# Sorting options
col1, col2 = st.columns([3, 1])

with col1:
    sort_by = st.selectbox(
        "Sort by",
        [
            "Timestamp (Recent First)",
            "Amount (High to Low)",
            "Probability (High to Low)",
            "Latency (High to Low)",
        ],
    )

with col2:
    page_size = st.selectbox("Rows per page", [10, 25, 50, 100], index=1)

# Apply sorting
if sort_by == "Timestamp (Recent First)":
    display_df = filtered_df.sort_values("timestamp", ascending=False)
elif sort_by == "Amount (High to Low)":
    display_df = filtered_df.sort_values("amount", ascending=False)
elif sort_by == "Probability (High to Low)":
    display_df = filtered_df.sort_values("fraud_probability", ascending=False)
else:
    display_df = filtered_df.sort_values("latency_ms", ascending=False)

# Format for display
display_df_formatted = display_df.copy()
display_df_formatted["timestamp"] = display_df_formatted["timestamp"].dt.strftime(
    "%Y-%m-%d %H:%M:%S"
)
display_df_formatted["amount"] = display_df_formatted["amount"].apply(
    lambda x: f"${x:.2f}"
)
display_df_formatted["fraud_probability"] = display_df_formatted[
    "fraud_probability"
].apply(lambda x: f"{x:.2%}")
display_df_formatted["is_fraud"] = display_df_formatted["is_fraud"].apply(
    lambda x: "🚨 Fraud" if x else "✅ Safe"
)
display_df_formatted["latency_ms"] = display_df_formatted["latency_ms"].apply(
    lambda x: f"{x:.2f}ms"
)

display_df_formatted = display_df_formatted[
    ["timestamp", "amount", "fraud_probability", "is_fraud", "risk_level", "latency_ms"]
]
display_df_formatted.columns = [
    "Timestamp",
    "Amount",
    "Fraud Prob.",
    "Status",
    "Risk",
    "Latency",
]

# Pagination
total_pages = len(display_df_formatted) // page_size + (
    1 if len(display_df_formatted) % page_size > 0 else 0
)

if total_pages > 1:
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    st.dataframe(
        display_df_formatted.iloc[start_idx:end_idx],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        f"Page {page} of {total_pages} ({len(display_df_formatted)} total records)"
    )
else:
    st.dataframe(
        display_df_formatted.head(page_size), use_container_width=True, hide_index=True
    )

# Export options
st.markdown("---")
st.markdown("## 📥 Export Data")

col1, col2 = st.columns(2)

with col1:
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data (CSV)",
        data=csv,
        file_name=f"fraud_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col2:
    # Summary statistics
    summary = {
        "total_predictions": len(filtered_df),
        "fraud_detected": int(filtered_df["is_fraud"].sum()),
        "fraud_rate": float(filtered_df["is_fraud"].mean()),
        "avg_amount": float(filtered_df["amount"].mean()),
        "avg_latency_ms": float(filtered_df["latency_ms"].mean()),
        "date_range": {
            "start": filtered_df["timestamp"].min().isoformat(),
            "end": filtered_df["timestamp"].max().isoformat(),
        },
    }

    import json

    summary_json = json.dumps(summary, indent=2)

    st.download_button(
        label="📥 Download Summary (JSON)",
        data=summary_json,
        file_name=f"fraud_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True,
    )

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Quick Stats")

    st.metric("Total Records", f"{len(df):,}")
    st.metric("Filtered Records", f"{len(filtered_df):,}")
    st.metric("Date Range", f"{(max_date - min_date).days} days")

    st.markdown("---")

    st.markdown("### 🔄 Refresh")
    if st.button("Clear Cache & Refresh"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")

    st.markdown("### 💡 Tips")
    st.markdown(
        """
    - Use filters to narrow down results
    - Sort by different columns
    - Export filtered data for analysis
    - Check hourly patterns for insights
    """
    )
