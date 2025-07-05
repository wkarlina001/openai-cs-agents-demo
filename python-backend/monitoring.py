import streamlit as st
import json
import pandas as pd
from collections import Counter
import os

METRICS_FILE = "log/metrics.log"

def load_metrics_data():
    """
    Loads and parses metric events from the metrics log file.
    """
    if not os.path.exists(METRICS_FILE):
        return []
    
    data = []
    with open(METRICS_FILE, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                st.warning(f"Skipping malformed log line: {line}")
    return data

def analyze_metrics(metrics_data):
    """
    Analyzes the raw metric data to compute dashboard statistics.
    """
    total_queries = 0
    tool_usage = Counter()
    retrieval_hits = 0
    retrieval_misses = 0
    reco_hits = 0
    reco_misses = 0

    dict_map = {"faq_retrieval":"RAG", "product_reco":"Recommendation Engine"}

    for entry in metrics_data:
        event_type = entry.get("event_type")
        event_data = entry.get("data", {})

        if event_type == "user_query":
            total_queries += 1
        elif event_type == "tool_usage":
            tool_name_data = event_data.get("tool_name")
            tool_name = dict_map[tool_name_data]
            if tool_name:
                tool_usage[tool_name] += 1
        elif event_type == "retrieval_analytics":
            retrieval_type = event_data.get("type")
            if retrieval_type == "hit":
                retrieval_hits += 1
            elif retrieval_type == "miss" or retrieval_type == "error_miss":
                retrieval_misses += 1
        elif event_type == "reco_analytics":
            reco_type = event_data.get("type")
            if reco_type == "hit":
                reco_hits += 1
            elif reco_type == "miss" or reco_type == "error_miss":
                reco_misses += 1

    return {
        "total_queries": total_queries,
        "tool_usage": dict(tool_usage),
        "retrieval_hits": retrieval_hits,
        "retrieval_misses": retrieval_misses,
        "reco_hits": reco_hits,
        "reco_misses": reco_misses
    }

# --- Streamlit App Layout ---

st.set_page_config(layout="centered", page_title="Agent Tools Usage Dashboard")

st.title("📊 Agent Tools Usage Dashboard")

st.markdown("---")
st.write("This dashboard displays real-time analytics from your Agents system's interactions.")

# # Button to refresh data
# if st.button("Refresh Data", help="Click to reload the latest metrics from the log file."):
#     st.cache_data.clear() # Clear cache to force reload
#     st.experimental_rerun() # Rerun the app to update data

st.markdown("---")

# Load and analyze data
metrics_data = load_metrics_data()
if not metrics_data:
    st.info("No metric data found. Start your FastAPI app and send some queries to generate data.")
    st.stop()

analytics = analyze_metrics(metrics_data)
print(analytics)

# --- Display Metrics ---

# Total User Queries Handled
st.header("Overall Performance")
st.metric(label="Total User Queries Handled", value=analytics["total_queries"])

st.markdown("---")

# Breakdown of Tool Usage Frequency
st.header("Tool Usage Frequency")
if analytics["tool_usage"]:
    total_tool_calls = sum(analytics["tool_usage"].values())
    st.metric(label="Total Tool Calls", value=total_tool_calls) # This is the added line

    tot_rag, tot_reco = st.columns(2)
    with tot_rag:
        st.metric(label="Total RAG Tool Calls", value=analytics["retrieval_hits"]+analytics["retrieval_misses"])
    with tot_reco:
        st.metric(label="Total Recommendation Tool Calls", value=analytics["reco_hits"]+analytics["reco_misses"])

else:
    st.info("No tool usage data available yet.")

st.markdown("---")

# Retrieval Hit/Miss Analytics
st.header("RAG Hit/Miss Analytics")
total_retrievals = analytics["retrieval_hits"] + analytics["retrieval_misses"]
hit_rate = (analytics["retrieval_hits"] / total_retrievals * 100) if total_retrievals > 0 else 0

col_hit, col_miss, col_rate = st.columns(3)
with col_hit:
    st.metric(label="Retrieval Hits", value=analytics["retrieval_hits"])
with col_miss:
    st.metric(label="Retrieval Misses", value=analytics["retrieval_misses"])
with col_rate:
    st.metric(label="Retrieval Hit Rate", value=f"{hit_rate:.2f}%")

if total_retrievals == 0:
    st.info("No retrieval analytics data available yet.")

# Retrieval Hit/Miss Analytics
st.header("Recommendation Engine Hit/Miss Analytics")
total_reco = analytics["reco_hits"] + analytics["reco_misses"]
hit_rate = (analytics["reco_hits"] / total_reco * 100) if total_reco > 0 else 0

col_hit, col_miss, col_rate = st.columns(3)
with col_hit:
    st.metric(label="Recommendation Hits", value=analytics["reco_hits"])
with col_miss:
    st.metric(label="Recommendation Misses", value=analytics["reco_misses"])
with col_rate:
    st.metric(label="Recommendation Hit Rate", value=f"{hit_rate:.2f}%")

if total_retrievals == 0:
    st.info("No recommendation analytics data available yet.")

st.markdown("---")
st.caption("Data is loaded from 'metrics.log'. Refresh to see the latest updates.")

