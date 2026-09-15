import streamlit as st
import pandas as pd
from typing import Dict, Any
from src.ui.components import page_header, kpi_row, section_label, divider, badge, empty_state
from src.ui.data import get_dashboard_dataset
from src.customer_360 import get_customer_profile

def render_customer_360(artifacts: Dict[str, Any]) -> None:
    page_header(
        "Customer 360",
        "Unified intelligence across receivables, revenue, and customer behavior."
    )
    
    analytics = get_dashboard_dataset(artifacts)
    
    # We also want to allow selecting customers that might only exist in Retail dataset.
    # For a true SaaS, we should merge the unique customers from both.
    try:
        retail_df = pd.read_csv("data/processed/retail_customer_snapshots.csv")
        retail_customers = retail_df["Customer ID"].dropna().unique().tolist()
    except Exception:
        retail_customers = []
        
    invoice_customers = analytics["customer"].dropna().unique().tolist() if not analytics.empty else []
    all_customers = sorted(list(set(invoice_customers + retail_customers)))
    
    if not all_customers:
        empty_state("👤", "No customer data", "No customer profiles are available in the system.")
        return

    # Customer Selector
    col_search, _ = st.columns([1, 2])
    with col_search:
        customer_option = st.selectbox("Search Customer ID or Name...", all_customers)
        
    if not customer_option:
        return
        
    # Fetch Profile
    profile = get_customer_profile(customer_option)
    ig_data = profile.get("invoiceguard_data", {})
    retail_data = profile.get("retail_intelligence", {})
    
    current_risk = ig_data.get("risk_level", "UNKNOWN")
    total_invoices = ig_data.get("total_invoices", 0)
    outstanding = ig_data.get("total_outstanding", 0.0)
    
    # Customer Header Panel
    risk_badge_html = badge(current_risk)
    
    st.markdown(f"""
    <div class="ig-c360-header">
      <div class="ig-c360-avatar">🏢</div>
      <div class="ig-c360-info">
        <h2 class="ig-c360-name">{customer_option}</h2>
        <div class="ig-c360-meta">
          <span>{risk_badge_html} Risk Status</span>
          <span>💳 ${outstanding:,.0f} Outstanding</span>
          <span>📄 {total_invoices} Invoices</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs
    tab_overview, tab_tx, tab_risk, tab_revenue, tab_ai = st.tabs([
        "OVERVIEW", "TRANSACTIONS", "RISK", "REVENUE", "AI INSIGHTS"
    ])
    
    with tab_overview:
        kpis = []
        if ig_data:
            kpis.extend([
                {"icon": "⚠️", "label": "Late-Payment Rate", "value": f"{ig_data.get('late_rate', 0):.0%}"},
                {"icon": "⏱️", "label": "Avg Delay", "value": f"{ig_data.get('avg_delay_days', 0):.1f} days"},
            ])
        if retail_data:
            preds = retail_data.get("predictions", {})
            kpis.extend([
                {"icon": "⭐", "label": "Retention Risk", "value": f"{preds.get('retention_risk', 'UNKNOWN')}"},
                {"icon": "💵", "label": "Est Future Rev", "value": f"${preds.get('expected_future_revenue_60d', 0):,.0f}"},
            ])
            
        if kpis:
            kpi_row(kpis)
        else:
            st.info("No predictive metrics available for this customer.")
            
        divider()
        if retail_data:
            section_label("Retail Behavioral Profile")
            rfm = retail_data.get("rfm", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Recency", f"{rfm.get('recency_days', 0)} Days")
            c2.metric("Frequency", f"{rfm.get('frequency', 0)} Orders")
            c3.metric("Monetary Value", f"${rfm.get('monetary', 0):,.0f}")
            c4.metric("Avg Order Value", f"${rfm.get('avg_order_value', 0):,.0f}")
            
    with tab_tx:
        if ig_data and not analytics.empty:
            cust_df = analytics[analytics["customer"] == customer_option].sort_values("invoice_date", ascending=False)
            cols = ["invoice_id", "invoice_date", "due_date", "invoice_amount", "risk_level", "late_probability", "expected_delay_days"]
            disp_df = cust_df[[c for c in cols if c in cust_df.columns]].copy()
            # formatting
            if "invoice_amount" in disp_df.columns:
                disp_df["invoice_amount"] = disp_df["invoice_amount"].apply(lambda x: f"${x:,.2f}")
            if "late_probability" in disp_df.columns:
                disp_df["late_probability"] = disp_df["late_probability"].apply(lambda x: f"{x:.1%}")
            if "expected_delay_days" in disp_df.columns:
                disp_df["expected_delay_days"] = disp_df["expected_delay_days"].apply(lambda x: f"{x:.1f}")
                
            st.dataframe(disp_df, use_container_width=True, hide_index=True)
        else:
            empty_state("📄", "No invoice data", "This customer does not have any invoice history in the InvoiceGuard system.")
            
    with tab_risk:
        if ig_data:
            st.markdown('<div class="ig-card">', unsafe_allow_html=True)
            st.markdown(f'<h3>Payment Risk Score: {ig_data.get("avg_risk_probability", 0):.1%}</h3>', unsafe_allow_html=True)
            st.write(f"This customer is currently rated as **{current_risk}** risk for late payment.")
            st.write(f"Historically, they pay {ig_data.get('late_rate', 0):.0%} of invoices late.")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            empty_state("🛡️", "No Risk Profile", "No B2B payment risk data available.")

    with tab_revenue:
        if retail_data:
            preds = retail_data.get("predictions", {})
            st.markdown('<div class="ig-card">', unsafe_allow_html=True)
            st.markdown(f'<h3>60-Day Revenue Forecast: ${preds.get("expected_future_revenue_60d", 0):,.2f}</h3>', unsafe_allow_html=True)
            st.write("Estimated based on behavioral patterns and historical purchasing activity.")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            empty_state("📈", "No Revenue Intelligence", "No retail behavioral data available for this customer.")
            
    with tab_ai:
        has_insights = False
        if retail_data and "recommendations" in retail_data:
            for rec in retail_data["recommendations"]:
                has_insights = True
                st.markdown(f"""
                <div class="ig-recommendation">
                  <div class="ig-recommendation-title">💡 Recommended Action</div>
                  <div class="ig-recommendation-body">{rec}</div>
                </div>
                """, unsafe_allow_html=True)
                
        if ig_data and current_risk in ["HIGH", "CRITICAL"]:
            has_insights = True
            st.markdown(f"""
            <div class="ig-recommendation" style="border-left-color: var(--danger);">
              <div class="ig-recommendation-title">⚠️ Payment Risk Alert</div>
              <div class="ig-recommendation-body">
                Prioritize collections outreach. Customer has an elevated payment risk with an average late rate of {ig_data.get('late_rate', 0):.0%}.
              </div>
            </div>
            """, unsafe_allow_html=True)
            
        if not has_insights:
            empty_state("✨", "No Critical Alerts", "No critical payment risk alerts or follow-up recommendations for this customer at this time.")

