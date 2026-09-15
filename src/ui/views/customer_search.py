import streamlit as st
import pandas as pd
from typing import Dict, Any
from src.ui.components import page_header, section_label, divider
from src.ui.data import get_dashboard_dataset


def render_customer_search(artifacts: Dict[str, Any]) -> None:
    page_header(
        "Customer Search",
        "Directory search and customer account lookup across the portfolio."
    )

    df = get_dashboard_dataset(artifacts)
    if df.empty:
        st.warning("No customer data available.")
        return

    # Build customer summary directory
    cust_summary = (
        df.groupby("customer")
        .agg(
            total_invoices=("invoice_id", "count"),
            total_outstanding=("outstanding_amount", "sum"),
            avg_risk=("late_probability", "mean"),
            risk_level=("risk_level", lambda s: "CRITICAL" if "CRITICAL" in s.values else ("HIGH" if "HIGH" in s.values else ("MEDIUM" if "MEDIUM" in s.values else "LOW"))),
        )
        .reset_index()
    )

    search_query = st.text_input("Search customer name", placeholder="Type customer name to filter…")
    
    if search_query:
        cust_summary = cust_summary[cust_summary["customer"].str.contains(search_query, case=False, na=False)]

    st.caption(f"Showing {len(cust_summary):,} accounts")

    # Business formatting
    display_df = cust_summary.sort_values("total_outstanding", ascending=False).copy()
    display_df.rename(columns={
        "customer": "Customer",
        "total_invoices": "Invoices",
        "total_outstanding": "Outstanding ($)",
        "avg_risk": "Payment Risk",
        "risk_level": "Account Risk Tier",
    }, inplace=True)
    display_df["Outstanding ($)"] = display_df["Outstanding ($)"].apply(lambda x: f"${x:,.0f}")
    display_df["Payment Risk"] = display_df["Payment Risk"].apply(lambda x: f"{x:.1%}")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    divider()
    section_label("Inspect Account")
    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        target_customer = st.selectbox("Select Account for Detailed 360 Profile", options=sorted(df["customer"].dropna().unique()))
    with col_btn:
        st.write("")
        st.write("")
        if st.button("Open Customer 360", type="primary", use_container_width=True):
            st.session_state.current_page = "Customer 360"
            st.rerun()
