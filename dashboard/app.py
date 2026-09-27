"""Minimal Streamlit dashboard entry point for experiment outputs."""

from pathlib import Path

import streamlit as st

st.set_page_config(page_title="FedGuard-XAI", layout="wide")
st.title("FedGuard-XAI")
st.caption("Federated intrusion detection, poisoning defense, and explainability")

metrics_dir = Path("results/metrics")
files = sorted(metrics_dir.glob("*.csv")) if metrics_dir.exists() else []
if files:
    st.dataframe(files[-1].read_text(), use_container_width=True)
else:
    st.info("Run an experiment to populate results/metrics.")
