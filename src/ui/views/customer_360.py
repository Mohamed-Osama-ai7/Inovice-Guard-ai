import streamlit as st
import pandas as pd
from typing import Dict, Any
from src.ui.components import page_header, kpi_row, section_label, divider, badge, empty_state
from src.ui.data import get_dashboard_dataset
from src.customer_360 import get_customer_profile


def render_customer_360(artifacts: Dict[str, Any]) -> None:
    page_header(
        "Customer 360",
        "Unified intelligence across receivables risk, purchase telemetry, and behavioral retention.",
    )
    
    analytics = get_dashboard_dataset(artifacts)
    
    # Extract available customer IDs across both domains without false entity linkage
    retail_customers = []
    try:
        retail_df = pd.read_csv("data/processed/retail_customer_snapshots.csv")
        col = "customer_id" if "customer_id" in retail_df.columns else "Customer ID"
        retail_customers = [str(c) for c in retail_df[col].dropna().unique().tolist()]
    except Exception:
        pass
        
    invoice_customers = [str(c) for c in analytics["customer"].dropna().unique().tolist()] if not analytics.empty else []
    
    # Categorize options clearly by domain
    invoice_options = [f"B2B Invoice Account: {cid}" for cid in sorted(invoice_customers)]
    retail_options = [f"Retail Account: {cid}" for cid in sorted(retail_customers[:500])] # Top 500 for snappy dropdown
    
    all_options = invoice_options + retail_options
    
    if not all_options:
        empty_state("👤", "No customer data", "No customer profiles are available in the system.")
        return

    # Customer Selector
    col_search, col_filter = st.columns([2, 1])
    with col_search:
        selected_option = st.selectbox("Select Customer Profile", all_options)
        
    if not selected_option:
        return

    # Extract true customer ID and domain
    if selected_option.startswith("B2B Invoice Account: "):
        raw_cust_id = selected_option.replace("B2B Invoice Account: ", "").strip()
        domain_tag = "InvoiceGuard Receivables Domain"
    else:
        raw_cust_id = selected_option.replace("Retail Account: ", "").strip()
        domain_tag = "UCI Online Retail Domain"
        
    # Fetch Profile
    profile = get_customer_profile(raw_cust_id)
    ig_data = profile.get("invoice_intelligence", {})
    retail_data = profile.get("retail_intelligence", {})
    comp_risk = profile.get("composite_risk", {})
    recommendations = profile.get("recommendations", [])
    
    current_risk = comp_risk.get("risk_tier", "LOW")
    comp_score = comp_risk.get("composite_score", 15)
    
    # Customer Header Panel
    risk_badge_html = badge(current_risk)
    
    st.markdown(
        f"""
        <div class="ig-c360-header">
          <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem; width:100%;">
            <div>
              <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.35rem;">
                <h2 class="ig-c360-name">{raw_cust_id}</h2>
                <span class="ig-domain-badge">
                  {domain_tag}
                </span>
              </div>
              <div style="font-size:0.8rem; color:var(--text-muted);">
                Segment: <strong style="color:var(--text-main);">{retail_data.get('segment') or ig_data.get('segment') or 'Commercial'}</strong> · 
                Industry: <strong style="color:var(--text-main);">{ig_data.get('industry') or 'Omnichannel Retail'}</strong>
              </div>
            </div>
            <div style="display:flex; align-items:center; gap:1rem;">
              <div style="text-align:right;">
                <div style="font-size:0.72rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.05em;">Composite Risk Index</div>
                <div style="font-size:1.35rem; font-weight:800; color:var(--text-main);">{comp_score} <span style="font-size:0.85rem; color:var(--text-muted); font-weight:500;">/ 100</span></div>
              </div>
              <div>{risk_badge_html}</div>
            </div>
          </div>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Tabs
    tab_overview, tab_tx, tab_risk, tab_revenue, tab_ai = st.tabs([
        "OVERVIEW", "TRANSACTION TELEMETRY", "MULTI-SIGNAL RISK", "REVENUE & RETENTION", "ACTIONABLE RECOMMENDATIONS"
    ])
    
    with tab_overview:
        kpis = []
        if ig_data:
            kpis.extend([
                {"icon": "📄", "label": "Total Invoices", "value": f"{ig_data.get('total_invoices', 0)}"},
                {"icon": "💳", "label": "Outstanding Amount", "value": f"${ig_data.get('total_outstanding', 0):,.0f}"},
                {"icon": "⚠️", "label": "Historical Late Rate", "value": f"{ig_data.get('historical_late_ratio', 0):.1%}"},
                {"icon": "⏱️", "label": "Avg Payment Delay", "value": f"{ig_data.get('historical_avg_delay_days', 0):.1f} days"},
            ])
        if retail_data:
            rfm = retail_data.get("rfm", {})
            preds = retail_data.get("predictions", {})
            kpis.extend([
                {"icon": "📦", "label": "Total Orders", "value": f"{rfm.get('frequency', 0)}"},
                {"icon": "💰", "label": "Monetary Value", "value": f"${rfm.get('monetary', 0):,.0f}"},
                {"icon": "⏳", "label": "Recency", "value": f"{rfm.get('recency_days', 0)} days ago"},
                {"icon": "📉", "label": "Inactivity Risk (60d)", "value": f"{preds.get('inactivity_risk_level', 'LOW')}"},
            ])
            
        if kpis:
            kpi_row(kpis)
            
        divider()
        if retail_data:
            section_label("Retail Behavioral Telemetry")
            rfm = retail_data.get("rfm", {})
            beh = retail_data.get("behavior", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Average Order Value", f"${rfm.get('avg_order_value', 0):,.2f}")
            c2.metric("Cancellation / Return Rate", f"{beh.get('cancellation_rate', 0):.1%}")
            c3.metric("Recent 90d Orders", f"{beh.get('orders_90d', 0)}")
            c4.metric("Recent 90d Revenue", f"${beh.get('revenue_90d', 0):,.0f}")
            
    with tab_tx:
        if ig_data and not analytics.empty:
            section_label("Invoice Ledger History")
            cust_df = analytics[analytics["customer"] == raw_cust_id].sort_values("invoice_date", ascending=False)
            cols = ["invoice_id", "invoice_date", "due_date", "invoice_amount", "risk_level", "late_probability", "expected_delay_days"]
            disp_df = cust_df[[c for c in cols if c in cust_df.columns]].copy()
            if "invoice_amount" in disp_df.columns:
                disp_df["invoice_amount"] = disp_df["invoice_amount"].apply(lambda x: f"${x:,.2f}")
            if "late_probability" in disp_df.columns:
                disp_df["late_probability"] = disp_df["late_probability"].apply(lambda x: f"{x:.1%}")
            if "expected_delay_days" in disp_df.columns:
                disp_df["expected_delay_days"] = disp_df["expected_delay_days"].apply(lambda x: f"{x:.1f}d")
                
            disp_df.rename(columns={
                "invoice_id": "Invoice ID",
                "invoice_date": "Invoice Date",
                "due_date": "Due Date",
                "invoice_amount": "Amount",
                "risk_level": "Risk Tier",
                "late_probability": "Late Probability",
                "expected_delay_days": "Est. Delay",
            }, inplace=True)
            st.dataframe(disp_df, use_container_width=True, hide_index=True)
        elif retail_data:
            section_label("Retail Transaction Summary")
            st.markdown(
                f"""
                <div class="ig-card">
                  <div class="ig-card-title">UCI Retail Order Summary</div>
                  <p style="color:var(--text-main); font-size:0.9rem; margin-bottom:0.5rem;">
                    Customer account <strong>{raw_cust_id}</strong> has accumulated <strong>{retail_data.get('rfm', {}).get('frequency', 0)} orders</strong>
                    generating <strong>${retail_data.get('rfm', {}).get('monetary', 0):,.2f}</strong> in gross volume across the two-year historical horizon.
                  </p>
                  <p style="color:var(--text-muted); font-size:0.82rem; margin:0;">
                    Latest active purchase occurred <strong>{retail_data.get('rfm', {}).get('recency_days', 0)} days</strong> prior to observation cutoff with a lifetime cancellation rate of <strong>{retail_data.get('behavior', {}).get('cancellation_rate', 0):.1%}</strong>.
                  </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            empty_state("📄", "No transaction ledger", "No transaction ledger records available for this customer identifier.")
            
    with tab_risk:
        section_label("Multi-Signal Risk Decomposition")
        signals = comp_risk.get("contributing_signals", {})
        weights = comp_risk.get("signal_weights", {})
        
        st.markdown(
            f"""
            <div class="ig-card" style="margin-bottom:1rem;">
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                  <div class="ig-card-title">Composite Business Risk Index</div>
                  <div style="font-size:1.25rem; font-weight:700; color:var(--text-main);">{comp_score} / 100 ({current_risk})</div>
                </div>
                <div>{badge(current_risk)}</div>
              </div>
              <div style="font-size:0.82rem; color:var(--text-muted); margin-top:0.4rem;">
                Methodology: {comp_risk.get('methodology', 'Transparent Weighted Composite Index')}
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        if signals:
            sig_cols = st.columns(len(signals))
            for i, (sig_name, sig_val) in enumerate(signals.items()):
                with sig_cols[i % len(sig_cols)]:
                    weight_val = weights.get(sig_name, 0.0)
                    formatted_name = sig_name.replace("_", " ").title()
                    st.metric(f"{formatted_name} (Weight: {weight_val:.0%})", f"{sig_val:.1%}")

    with tab_revenue:
        section_label("Revenue Forecast & Inactivity Horizon")
        if retail_data:
            preds = retail_data.get("predictions", {})
            st.markdown(
                f"""
                <div class="ig-card" style="margin-bottom:1rem;">
                  <div class="ig-card-title">60-Day Forward Predictive Outlook</div>
                  <div style="display:flex; gap:2rem; margin-top:0.75rem; flex-wrap:wrap;">
                    <div>
                      <div style="font-size:0.75rem; color:var(--text-muted); text-transform:uppercase;">60-Day Expected Revenue</div>
                      <div style="font-size:1.4rem; font-weight:700; color:var(--success);">${preds.get('expected_future_revenue_60d', 0):,.2f}</div>
                    </div>
                    <div>
                      <div style="font-size:0.75rem; color:var(--text-muted); text-transform:uppercase;">Repurchase Probability</div>
                      <div style="font-size:1.4rem; font-weight:700; color:var(--primary);">{preds.get('repurchase_probability_60d', 0):.1%}</div>
                    </div>
                    <div>
                      <div style="font-size:0.75rem; color:var(--text-muted); text-transform:uppercase;">Dormancy Risk</div>
                      <div style="font-size:1.4rem; font-weight:700; color:{'var(--danger)' if preds.get('inactivity_risk_probability_60d', 0) >= 0.70 else 'var(--text-main)'};">{preds.get('inactivity_risk_probability_60d', 0):.1%}</div>
                    </div>
                  </div>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            if "top_drivers" in retail_data:
                section_label("Local Predictive Factor Drivers")
                for driver in retail_data["top_drivers"]:
                    st.markdown(
                        f"""
                        <div style="padding:0.5rem 0.75rem; margin-bottom:0.4rem; background:var(--surface-hover); border-left:3px solid var(--primary); border-radius:0 4px 4px 0; font-size:0.85rem; color:var(--text-main);">
                          {driver}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        elif ig_data:
            st.markdown(
                f"""
                <div class="ig-card">
                  <div class="ig-card-title">B2B Receivables Exposure</div>
                  <div style="font-size:1.25rem; font-weight:700; color:var(--text-main); margin-top:0.5rem;">
                    Total Outstanding: ${ig_data.get('total_outstanding', 0):,.2f}
                  </div>
                  <p style="color:var(--text-muted); font-size:0.82rem; margin-top:0.25rem;">
                    Invoice portfolio volume: ${ig_data.get('total_invoice_amount', 0):,.2f} across {ig_data.get('total_invoices', 0)} invoices.
                  </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
    with tab_ai:
        section_label("Actionable Operational Recommendations")
        if recommendations:
            for rec in recommendations:
                p_color = "var(--danger)" if "P1" in rec.get("priority", "") else ("var(--warning)" if "P2" in rec.get("priority", "") else "var(--primary)")
                st.markdown(
                    f"""
                    <div class="ig-card" style="border-left: 4px solid {p_color}; margin-bottom:0.75rem; padding:1rem 1.25rem;">
                      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.35rem;">
                        <span style="font-weight:700; color:var(--text-main); font-size:0.95rem;">{rec.get('action')}</span>
                        <span style="font-size:0.72rem; font-weight:600; color:{p_color}; background:var(--surface-hover); border:1px solid var(--border); padding:0.15rem 0.5rem; border-radius:3px;">
                          {rec.get('priority')}
                        </span>
                      </div>
                      <div style="font-size:0.82rem; color:var(--primary); margin-bottom:0.3rem;">
                        <strong>Trigger:</strong> {rec.get('trigger')}
                      </div>
                      <div style="font-size:0.85rem; color:var(--text-muted); line-height:1.4;">
                        {rec.get('detail')}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            empty_state("✨", "Account Healthy", "No high-priority risk interventions required for this customer.")

    divider()
    # Scientific domain separation footer note
    st.markdown(
        """
        <div style="font-size:0.72rem; color:var(--text-muted); line-height:1.4; opacity:0.8; padding:0.5rem 0.2rem;">
          🛡️ <strong>Domain Separation Guarantee:</strong> InvoiceGuard receivables accounts and UCI Online Retail II records represent distinct, non-overlapping empirical customer cohorts. To protect scientific integrity, identifiers from the respective datasets are maintained in strict domain isolation with zero synthetic entity resolution.
        </div>
        """,
        unsafe_allow_html=True
    )
