"""
Streamlit Web Dashboard for FreightTiger Smart Shipping Cost Assistant.
Interactive exploration of weekly route costs, baselines, anomalies, and grounded explanations.
Usage:
    streamlit run app.py
"""

from pathlib import Path
import pandas as pd
import streamlit as st

from src.config import default_config
from src.pipeline import ShippingCostAssistantPipeline


st.set_page_config(
    page_title="FreightTiger Shipping Cost Assistant",
    page_icon="🚚",
    layout="wide",
)

st.title("🚚 FreightTiger Smart Shipping Cost Assistant")
st.markdown(
    "Automated surveillance of freight rates, weekly cost-per-tonne-km tracking, "
    "baseline comparisons (own history vs. peer routes), and grounded context-note evidence retrieval."
)

@st.cache_data
def get_pipeline_data():
    pipeline = ShippingCostAssistantPipeline(default_config)
    # Get both full analysis and summary
    df, summary = pipeline.run(output_mode="all")
    return df, summary

try:
    with st.spinner("Loading and processing shipment records and context notes..."):
        df, summary = get_pipeline_data()
except Exception as e:
    st.error(f"Failed to load pipeline data: {e}")
    st.stop()

# Top metric summary cards
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Route-Weeks", summary["total_route_weeks"])
with col2:
    st.metric("Candidate Anomalies", summary["candidate_anomalies"])
with col3:
    st.metric("Justified Rises", summary["justified_anomalies"])
with col4:
    st.metric("Unexplained Flags", summary["unexplained_anomalies"])
with col5:
    st.metric("Normal Weeks", summary["normal_weeks"])

st.divider()

# Sidebar filters
st.sidebar.header("🔍 Filters")

unique_routes = ["All"] + sorted(df["route"].unique().tolist())
selected_route = st.sidebar.selectbox("Select Route", unique_routes)

flag_filter = st.sidebar.multiselect(
    "Filter by Verdict / Flag",
    options=["Yes", "No (justified)", "No"],
    default=["Yes", "No (justified)"],
)

min_date = df["week_of"].min()
max_date = df["week_of"].max()
date_range = st.sidebar.date_input(
    "Week Range",
    value=[pd.to_datetime(min_date), pd.to_datetime(max_date)],
    min_value=pd.to_datetime(min_date),
    max_value=pd.to_datetime(max_date),
)

# Apply filters
filtered_df = df.copy()

if selected_route != "All":
    filtered_df = filtered_df[filtered_df["route"] == selected_route]

if flag_filter:
    filtered_df = filtered_df[filtered_df["flagged"].isin(flag_filter)]

if len(date_range) == 2:
    start_str = date_range[0].strftime("%Y-%m-%d")
    end_str = date_range[1].strftime("%Y-%m-%d")
    filtered_df = filtered_df[(filtered_df["week_of"] >= start_str) & (filtered_df["week_of"] <= end_str)]

st.subheader(f"📊 Filtered Shipping Cost Records ({len(filtered_df)} route-weeks)")

# Color formatting function
def highlight_verdict(val):
    if val == "Yes":
        return "background-color: #ffebee; color: #c62828; font-weight: bold;"
    elif val == "No (justified)":
        return "background-color: #e8f5e9; color: #2e7d32; font-weight: bold;"
    return ""

styler = filtered_df.style
if hasattr(styler, "map"):
    styled_df = styler.map(highlight_verdict, subset=["flagged"])
else:
    styled_df = styler.applymap(highlight_verdict, subset=["flagged"])

st.dataframe(
    styled_df,
    use_container_width=True,
    height=450,
)

# Download button for filtered view
csv_data = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Current View as CSV",
    data=csv_data,
    file_name="freighttiger_filtered_analysis.csv",
    mime="text/csv",
)

st.divider()

# Detailed Anomaly & Evidence Inspector
st.subheader("🔎 Context Note & Evidence Inspector")
anomalies_df = filtered_df[filtered_df["flagged"].isin(["Yes", "No (justified)"])]

if len(anomalies_df) > 0:
    for _, row in anomalies_df.iterrows():
        is_justified = row["flagged"] == "No (justified)"
        badge = "🟢 JUSTIFIED COST RISE" if is_justified else "🔴 UNEXPLAINED COST RISE (FLAGGED)"
        note_badge = f" [Cited Note: {row['matched_note_id']}]" if row['matched_note_id'] else " [No Supporting Note]"

        with st.expander(f"{badge} — {row['route']} (Week of {row['week_of']}) | Cost: ₹{row['cost_per_tonne_km']}/T-km {note_badge}"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Baseline A (vs. Own History):** `{row['vs_own_history']}`")
                st.markdown(f"**Baseline B (vs. Peer Routes):** `{row['vs_similar_routes']}`")
            with c2:
                st.markdown(f"**Verdict:** `{row['flagged']}`")
                st.markdown(f"**Matched Note ID:** `{row['matched_note_id'] or 'None'}`")

            st.info(f"**Explanation:** {row['reason']}")
else:
    st.info("No anomalies match the current filter selection.")
