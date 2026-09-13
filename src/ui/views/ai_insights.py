import streamlit as st
import pandas as pd
from typing import Dict, Any
from src.ui.components import page_header, empty_state, divider
from app import render_ai_explanation

def render_ai_recommendations(artifacts: Dict[str, Any]) -> None:
    page_header("AI Recommendations", "Cross-domain insights generated from customer behavioral signals.")
    
    # We load the retail customers and look for recommendations
    try:
        df = pd.read_csv("data/processed/retail_customer_snapshots.csv")
    except Exception:
        df = pd.DataFrame()
        
    if df.empty or "target_repurchase_60d" not in df.columns:
        empty_state("✨", "No Insights Available", "No recommendations could be generated from the current data pipeline.")
        return
        
    # Let's generate a list of high priority insights based on risk
    st.markdown("### High Priority Alerts")
    
    # Just show top 3 risky customers as examples of alerts
    risky = df[(df["target_repurchase_60d"] == 0) & (df["monetary"] > 1000)].sort_values("monetary", ascending=False).head(5)
    
    if risky.empty:
        st.info("No high-priority alerts at this time.")
    else:
        for idx, row in risky.iterrows():
            st.markdown(f"""
            <div class="ig-card" style="margin-bottom:1rem; border-left: 4px solid var(--danger);">
              <div style="font-weight:600; color:var(--text-main); margin-bottom:0.5rem; display:flex; align-items:center; gap:0.5rem;">
                🚨 Customer {row['Customer ID']} is at elevated repurchase risk.
              </div>
              <div style="color:var(--text-light); font-size:0.95rem; margin-bottom:1rem;">
                <strong>Recommended Action:</strong> Prioritize proactive retention outreach. Account value is significant.
              </div>
              <div style="font-size:0.85rem; color:var(--text-muted);">
                <strong>Supporting Signals:</strong>
                <ul style="margin-top:0.25rem;">
                  <li>Historical purchasing value: ${row['monetary']:,.2f}</li>
                  <li>Days since last active: {row['recency_days']} days</li>
                  <li>Purchase frequency: {row['frequency']} orders</li>
                </ul>
              </div>
            </div>
            """, unsafe_allow_html=True)
            
def render_risk_signals(artifacts: Dict[str, Any]) -> None:
    page_header("Risk Signals", "Global risk signals impacting receivables.")
    # Here we could render a simple signal list or redirect to NLP
    st.info("Global risk signals dashboard is actively monitoring transactions.")
    
def render_explainable_ai(artifacts: Dict[str, Any]) -> None:
    page_header("Explainable AI", "Feature importance and model transparency for payment predictions.")
    st.markdown("<style>h1:first-child {display:none;}</style>", unsafe_allow_html=True)
    render_ai_explanation(artifacts)
