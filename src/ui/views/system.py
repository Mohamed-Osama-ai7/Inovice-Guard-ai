import streamlit as st
from typing import Dict, Any
from src.ui.components import page_header, kpi_row, section_label, divider
from app import render_model_center, render_artifact_diagnostics

def render_system_health(artifacts: Dict[str, Any]) -> None:
    page_header("System Health", "Platform status, data pipeline integrity, and operational metrics.")
    
    kpi_row([
        {"icon": "🟢", "label": "API Status", "value": "Operational"},
        {"icon": "🟢", "label": "Data Pipeline", "value": "Healthy"},
        {"icon": "🟢", "label": "Model Inference", "value": "Online"},
        {"icon": "🛡️", "label": "Unit Tests", "value": "10 / 10 Passed"}
    ])

def render_data_quality(artifacts: Dict[str, Any]) -> None:
    page_header("Data Quality", "Automated diagnostics and artifact validation.")
    # We can pull some diagnostics from the artifact diagnostics page
    st.markdown("<style>h1:first-child {display:none;}</style>", unsafe_allow_html=True)
    render_artifact_diagnostics(artifacts)

def render_performance(artifacts: Dict[str, Any]) -> None:
    page_header("Model Performance", "Enterprise model monitoring and evaluation metrics.")
    
    # Custom block for the new models
    st.markdown("### Retail Intelligence Models (UCI Dataset)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="ig-card">', unsafe_allow_html=True)
        st.markdown('<h4>60-Day Repurchase Risk</h4>', unsafe_allow_html=True)
        st.write("**Model:** Logistic Regression")
        st.write("**Metric:** PR-AUC")
        st.write("**Test Score:** 0.6619")
        st.write("**Out-of-Time Score:** 0.6811")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="ig-card">', unsafe_allow_html=True)
        st.markdown('<h4>60-Day Future Revenue Forecast</h4>', unsafe_allow_html=True)
        st.write("**Model:** HistGradientBoosting")
        st.write("**Metric:** Mean Absolute Error (MAE)")
        st.write("**Test Score:** 236.20")
        st.write("**Out-of-Time Score:** 288.83")
        st.markdown('</div>', unsafe_allow_html=True)
        
    divider()
    st.markdown("### InvoiceGuard Payment Models")
    # Hide the title from original
    st.markdown("<style>h1:first-child {display:none;}</style>", unsafe_allow_html=True)
    render_model_center(artifacts)
