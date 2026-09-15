import streamlit as st
import pandas as pd
from typing import Dict, Any
from src.ui.components import page_header, empty_state, divider
from app import render_ai_explanation


def render_ai_recommendations(artifacts: Dict[str, Any]) -> None:
    page_header(
        "Recommendations",
        "Priority actions generated from customer behavioral signals and payment risk intelligence."
    )

    try:
        df = pd.read_csv("data/processed/retail_customer_snapshots.csv")
    except Exception:
        df = pd.DataFrame()

    if df.empty or "target_repurchase_60d" not in df.columns:
        empty_state(
            "💡",
            "No Recommendations Available",
            "No recommendations could be generated from the current data pipeline.",
        )
        return

    st.markdown("### Priority Actions")

    risky = (
        df[(df["target_repurchase_60d"] == 0) & (df["monetary"] > 1000)]
        .sort_values("monetary", ascending=False)
        .head(5)
    )

    if risky.empty:
        st.info("No high-priority actions at this time.")
    else:
        for _, row in risky.iterrows():
            st.markdown(
                f"""
            <div class="ig-card" style="margin-bottom:1rem; border-left: 4px solid var(--danger);">
              <div style="font-weight:600; color:var(--text-main); margin-bottom:0.5rem;">
                Customer {row['Customer ID']} — Elevated Retention Risk
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
            """,
                unsafe_allow_html=True,
            )
