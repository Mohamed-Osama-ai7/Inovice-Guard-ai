import streamlit as st
import pandas as pd
from typing import Dict, Any
from src.ui.components import page_header, kpi_row, divider, empty_state
from app import render_single_prediction as orig_single, render_risk_center as orig_risk_center, render_batch_predictions as orig_batch

# We just wrap the existing app functions but apply the new page headers
# To keep the UI pristine without rewriting 1000 lines of working prediction logic.
# The user said: "Do not rewrite working backend/business logic just for architectural aesthetics."
# But we must update the UI shell.

def render_invoice_risk(artifacts: Dict[str, Any]) -> None:
    page_header("Invoice Risk", "Predict payment delays and risks for individual invoices.")
    # The original single prediction function has its own title logic which we might need to hide.
    # But for now, we just call it. We could monkeypatch or pass a flag, but let's just use it.
    st.markdown("<style>h1:first-child {display:none;}</style>", unsafe_allow_html=True)
    orig_single(artifacts, cold_start=False)

def render_collections(artifacts: Dict[str, Any]) -> None:
    page_header("Collections & Alert Center", "Batch operations and at-risk queues for the collections team.")
    st.markdown("<style>h1:first-child {display:none;}</style>", unsafe_allow_html=True)
    orig_batch(artifacts)
    divider()
    from app import render_alert_center
    render_alert_center(artifacts)

def render_customer_risk(artifacts: Dict[str, Any]) -> None:
    page_header("Customer Risk", "Aggregate payment risk and historical trends at the customer level.")
    st.markdown("<style>h1:first-child {display:none;}</style>", unsafe_allow_html=True)
    orig_risk_center(artifacts)
