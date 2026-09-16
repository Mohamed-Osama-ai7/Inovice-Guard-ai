import streamlit as st
import pandas as pd
from typing import Dict, Any
from src.ui.components import page_header, empty_state, kpi_row

def _load_retail_data() -> pd.DataFrame:
    try:
        df = pd.read_csv("data/processed/retail_customer_snapshots.csv")
        if df.empty:
            return pd.DataFrame()
        # Standardize column naming
        if "customer_id" in df.columns and "Customer ID" not in df.columns:
            df["Customer ID"] = df["customer_id"].astype(str)
        if "future_revenue_60d" in df.columns and "target_future_revenue_60d" not in df.columns:
            df["target_future_revenue_60d"] = df["future_revenue_60d"]
        if "repurchase_60d" in df.columns and "target_repurchase_60d" not in df.columns:
            df["target_repurchase_60d"] = df["repurchase_60d"]
            
        # Deduplicate to latest snapshot per customer for accurate cross-sectional pipeline views
        if "snapshot_date" in df.columns:
            df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
            df = df.sort_values("snapshot_date").groupby("Customer ID").tail(1).copy()
            
        return df
    except Exception:
        return pd.DataFrame()

def render_revenue_forecast(artifacts: Dict[str, Any]) -> None:
    page_header("Revenue Forecast", "60-day revenue projections based on customer behavioral patterns and historical transaction activity.")
    df = _load_retail_data()
    if df.empty or "target_future_revenue_60d" not in df.columns:
        empty_state("📈", "No Revenue Data", "Retail data pipeline has not generated forecast data.")
        return
    
    total_forecast = df["target_future_revenue_60d"].sum()
    avg_forecast = df["target_future_revenue_60d"].mean()
    
    kpi_row([
        {"icon": "💰", "label": "Total Forecasted Pipeline", "value": f"${total_forecast:,.0f}"},
        {"icon": "💳", "label": "Average Expected per Customer", "value": f"${avg_forecast:,.0f}"},
    ])
    
    st.markdown("### Top Accounts by Forecasted Revenue")
    top = df.sort_values("target_future_revenue_60d", ascending=False).head(20)[["Customer ID", "target_future_revenue_60d", "monetary", "recency_days", "frequency"]]
    # Format columns for display
    top.columns = ["Customer ID", "Forecast (60d)", "Historical Value", "Recency (Days)", "Frequency"]
    top["Forecast (60d)"] = top["Forecast (60d)"].apply(lambda x: f"${x:,.2f}")
    top["Historical Value"] = top["Historical Value"].apply(lambda x: f"${x:,.2f}")
    st.dataframe(top, use_container_width=True, hide_index=True)

def render_repurchase_risk(artifacts: Dict[str, Any]) -> None:
    page_header("Customer Retention", "Accounts showing behavioral signals associated with elevated churn risk over the next 60 days.")
    df = _load_retail_data()
    if df.empty or "target_repurchase_60d" not in df.columns:
        empty_state("📉", "No Retention Data", "Retail data pipeline has not generated retention data.")
        return
        
    churn_rate = 1 - df["target_repurchase_60d"].mean()
    at_risk_count = len(df[df["target_repurchase_60d"] == 0])
    
    kpi_row([
        {"icon": "⚠️", "label": "Projected Churn Rate", "value": f"{churn_rate:.1%}"},
        {"icon": "👥", "label": "Accounts At Risk", "value": f"{at_risk_count}"},
    ])
    
    st.markdown("### High-Risk Accounts")
    risky = df[df["target_repurchase_60d"] == 0].sort_values("monetary", ascending=False).head(20)[["Customer ID", "monetary", "recency_days", "frequency"]]
    risky.columns = ["Customer ID", "Historical Value", "Recency (Days)", "Frequency"]
    risky["Historical Value"] = risky["Historical Value"].apply(lambda x: f"${x:,.2f}")
    st.dataframe(risky, use_container_width=True, hide_index=True)

    # Contextual priority retention actions
    priority_recs = df[(df["target_repurchase_60d"] == 0) & (df["monetary"] > 1000)].sort_values("monetary", ascending=False).head(3)
    if not priority_recs.empty:
        st.markdown("### Priority Retention Actions")
        for _, row in priority_recs.iterrows():
            st.markdown(
                f"""<div class="ig-card" style="margin-bottom:0.75rem; border-left: 4px solid var(--danger);">
                  <div style="font-weight:600; color:var(--text-main); margin-bottom:0.25rem;">
                    Account {row['Customer ID']} — Elevated Churn Risk (${row['monetary']:,.0f} Historical Value)
                  </div>
                  <div style="color:var(--text-light); font-size:0.9rem;">
                    <strong>Recommended Action:</strong> Initiate proactive account management outreach. Inactive for {row['recency_days']:.0f} days across {row['frequency']:.0f} previous orders.
                  </div>
                </div>""",
                unsafe_allow_html=True,
            )


def render_revenue_at_risk(artifacts: Dict[str, Any]) -> None:
    page_header("Revenue at Risk", "Intersection of predicted churn and historical monetary value.")
    df = _load_retail_data()
    if df.empty or "target_repurchase_60d" not in df.columns:
        empty_state("📉", "No Retention Data", "Retail data pipeline has not generated retention data.")
        return
        
    at_risk_df = df[df["target_repurchase_60d"] == 0]
    total_at_risk = at_risk_df["monetary"].sum()
    
    kpi_row([
        {"icon": "🚨", "label": "Total Revenue at Risk", "value": f"${total_at_risk:,.0f}"},
        {"icon": "📊", "label": "Avg Risk Exposure per Account", "value": f"${at_risk_df['monetary'].mean() if len(at_risk_df)>0 else 0:,.0f}"}
    ])
    
    try:
        import plotly.express as px
    except Exception:
        px = None

    if px is not None and not at_risk_df.empty:
        fig = px.scatter(
            at_risk_df, x="recency_days", y="monetary", color="frequency",
            title="Revenue at Risk: Recency vs Monetary Value",
            labels={"recency_days": "Days Since Last Purchase", "monetary": "Historical Value ($)", "frequency": "Purchase Frequency"},
            color_continuous_scale="Reds"
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f9fafb"))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    elif not at_risk_df.empty:
        st.scatter_chart(at_risk_df, x="recency_days", y="monetary", color="frequency")
