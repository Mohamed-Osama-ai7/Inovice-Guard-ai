import streamlit as st
import pandas as pd
from typing import Dict, Any
from src.ui.components import page_header, kpi_row, section_label, divider
from src.ui.data import build_dashboard_dataset, get_dashboard_dataset

def render_overview(artifacts: Dict[str, Any]) -> None:
    page_header(
        "Overview",
        "Enterprise intelligence across receivables, risk, and customer revenue.",
        '<span class="ig-env-badge">Production</span>'
    )

    from src.ui.data import get_dashboard_dataset
    df = get_dashboard_dataset(artifacts)
    
    if df.empty:
        st.warning("No data available.")
        return

    # Metrics
    total_invoices = len(df)
    total_amount = df["invoice_amount"].sum()
    outstanding = df["outstanding_amount"].sum()
    
    late_df = df[df["predicted_late"] == 1]
    at_risk_amount = late_df["outstanding_amount"].sum()
    high_risk_count = len(df[df["risk_level"].isin(["HIGH", "CRITICAL"])])

    kpi_row([
        {"icon": "🏢", "label": "Customers", "value": f"{df['customer'].nunique():,}"},
        {"icon": "💰", "label": "Open Receivables", "value": f"${outstanding:,.0f}"},
        {"icon": "⚠️", "label": "At-Risk Revenue", "value": f"${at_risk_amount:,.0f}", "trend": "Critical", "trend_dir": "down"},
        {"icon": "🚨", "label": "High-Risk Accounts", "value": f"{high_risk_count}", "trend": "Requires Attention", "trend_dir": "down"},
    ])

    divider()
    section_label("Risk & Revenue Overview")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="ig-card">', unsafe_allow_html=True)
        st.markdown('<div class="ig-card-title">Customer Risk Distribution</div>', unsafe_allow_html=True)
        risk_counts = df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Invoices"]
        
        import plotly.express as px
        color_map = {"LOW": "#10b981", "MEDIUM": "#f59e0b", "HIGH": "#ef4444", "CRITICAL": "#b91c1c"}
        fig = px.pie(risk_counts, values="Invoices", names="Risk Level", hole=0.7, color="Risk Level", color_discrete_map=color_map)
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=True, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f9fafb"))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="ig-card">', unsafe_allow_html=True)
        st.markdown('<div class="ig-card-title">Top Accounts at Risk</div>', unsafe_allow_html=True)
        top_risk = df[df["predicted_late"] == 1].groupby("customer")["outstanding_amount"].sum().sort_values(ascending=False).head(5).reset_index()
        
        if not top_risk.empty:
            fig2 = px.bar(top_risk, y="customer", x="outstanding_amount", orientation='h', color_discrete_sequence=["#ef4444"])
            fig2.update_layout(
                margin=dict(t=10, b=10, l=10, r=10), 
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)", 
                font=dict(color="#f9fafb"),
                xaxis_title="Revenue at Risk ($)",
                yaxis_title="",
                yaxis={'categoryorder':'total ascending'}
            )
            st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("No at-risk revenue detected.")
        st.markdown('</div>', unsafe_allow_html=True)
