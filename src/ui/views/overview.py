import streamlit as st
import pandas as pd
from typing import Dict, Any
from src.ui.components import page_header, kpi_row, section_label, divider, get_chart_theme
from src.ui.data import get_dashboard_dataset


def render_overview(artifacts: Dict[str, Any]) -> None:
    page_header(
        "Overview",
        "Executive intelligence across receivables, customer risk, and exposed revenue.",
        '<span class="ig-badge ig-ok">Operational</span>'
    )

    df = get_dashboard_dataset(artifacts)
    
    if df.empty:
        st.warning("No portfolio data available.")
        return

    # Metrics
    outstanding = df["outstanding_amount"].sum()
    late_df = df[df["predicted_late"] == 1]
    at_risk_amount = late_df["outstanding_amount"].sum()
    high_risk_invoices = int((df["risk_level"].isin(["HIGH", "CRITICAL"])).sum())
    high_risk_customers = int(df.loc[df["risk_level"].isin(["HIGH", "CRITICAL"]), "customer"].nunique())

    kpi_row([
        {"icon": "💰", "label": "Outstanding Receivables", "value": f"${outstanding:,.0f}"},
        {"icon": "⚠️", "label": "Revenue at Risk", "value": f"${at_risk_amount:,.0f}", "trend": "High Priority", "trend_dir": "down"},
        {"icon": "📄", "label": "High-Risk Invoices", "value": f"{high_risk_invoices}", "trend": "At Risk", "trend_dir": "down"},
        {"icon": "👥", "label": "Customers Requiring Attention", "value": f"{high_risk_customers}", "trend": "Action Required", "trend_dir": "down"},
    ])

    divider()

    # Attention Required Section
    section_label("Attention Required — Highest Risk Invoices")
    high_risk_df = df[df["risk_level"].isin(["HIGH", "CRITICAL"])].sort_values("late_probability", ascending=False).head(5)
    
    if not high_risk_df.empty:
        table_rows = []
        for _, r in high_risk_df.iterrows():
            table_rows.append({
                "Invoice": r.get("invoice_id", "—"),
                "Customer": r.get("customer", "—"),
                "Due Date": str(r.get("due_date", "—")),
                "Amount": f"${r.get('invoice_amount', 0):,.0f}",
                "Payment Risk": f"{r.get('late_probability', 0):.1%}",
                "Risk Level": r.get("risk_level", "HIGH"),
                "Revenue at Risk": f"${r.get('estimated_financial_exposure', 0):,.0f}",
                "Recommended Action": r.get("recommendation", "Prioritize follow-up"),
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    else:
        st.success("No critical-risk invoices requiring immediate attention.")

    divider()
    section_label("Receivables & Risk Analytics")

    chart_theme = get_chart_theme()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="ig-card">', unsafe_allow_html=True)
        st.markdown('<div class="ig-card-title">Portfolio Risk Distribution</div>', unsafe_allow_html=True)
        risk_counts = df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Invoices"]
        
        try:
            import plotly.express as px
        except Exception:
            px = None

        if px is not None:
            color_map = chart_theme["color_map"]
            fig = px.pie(risk_counts, values="Invoices", names="Risk Level", hole=0.68, color="Risk Level", color_discrete_map=color_map)
            fig.update_layout(
                margin=dict(t=10, b=10, l=10, r=10), 
                showlegend=True, 
                paper_bgcolor=chart_theme["paper_bg"], 
                plot_bgcolor=chart_theme["plot_bg"], 
                font=dict(color=chart_theme["font_color"])
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.bar_chart(risk_counts.set_index("Risk Level")["Invoices"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="ig-card">', unsafe_allow_html=True)
        st.markdown('<div class="ig-card-title">Top Exposure by Account</div>', unsafe_allow_html=True)
        top_risk = df[df["predicted_late"] == 1].groupby("customer")["outstanding_amount"].sum().sort_values(ascending=False).head(5).reset_index()
        
        if not top_risk.empty:
            if px is not None:
                fig2 = px.bar(top_risk, y="customer", x="outstanding_amount", orientation='h', color_discrete_sequence=[chart_theme["danger_color"]])
                fig2.update_layout(
                    margin=dict(t=10, b=10, l=10, r=10), 
                    paper_bgcolor=chart_theme["paper_bg"], 
                    plot_bgcolor=chart_theme["plot_bg"], 
                    font=dict(color=chart_theme["font_color"]),
                    xaxis_title="Revenue at Risk ($)",
                    yaxis_title="",
                    yaxis={'categoryorder':'total ascending'}
                )
                st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
            else:
                st.bar_chart(top_risk.set_index("customer")["outstanding_amount"])
        else:
            st.info("No at-risk revenue detected.")
        st.markdown('</div>', unsafe_allow_html=True)

    divider()
    section_label("Quick Access")
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        if st.button("Invoices", use_container_width=True):
            st.session_state.current_page = "Invoices"
            st.rerun()
    with q2:
        if st.button("Customer 360", use_container_width=True):
            st.session_state.current_page = "Customer 360"
            st.rerun()
    with q3:
        if st.button("Risk Drivers", use_container_width=True):
            st.session_state.current_page = "Risk Drivers"
            st.rerun()
    with q4:
        if st.button("Message Intelligence", use_container_width=True):
            st.session_state.current_page = "Message Intelligence"
            st.rerun()
