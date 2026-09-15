"""
InvoiceGuard AI - Retail Intelligence Views: Customer Segmentation & Product Intelligence
Presents empirical clustering and product performance telemetry from the UCI Online Retail II dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

from src.ui.components import badge, divider, kpi_row, page_header, section_label

ROOT = Path(__file__).resolve().parents[3]
REPORTS_DIR = ROOT / "reports"
PROCESSED_DIR = ROOT / "data" / "processed"


def render_customer_segmentation(artifacts: Dict[str, Any]) -> None:
    page_header(
        "Customer Segmentation",
        "Empirical RFM and behavioral cohort segmentation powered by KMeans clustering on UCI Online Retail II.",
    )

    seg_file = REPORTS_DIR / "retail_segmentation_report.json"
    if not seg_file.exists():
        st.info("Segmentation analytics are compiling. Run 'py -m src.retail_analytics' to generate precomputed reports.")
        return

    try:
        report = json.loads(seg_file.read_text(encoding="utf-8"))
    except Exception as exc:
        st.error(f"Could not load segmentation report: {exc}")
        return

    clusters = report.get("clusters", {})
    sil_score = report.get("silhouette_score", 0.0)
    db_score = report.get("davies_bouldin_score", 0.0)
    total_cust = report.get("evaluated_customers", 0)

    # Top KPI row
    kpis = [
        {"icon": "👥", "label": "Segmented Customers", "value": f"{total_cust:,}"},
        {"icon": "🎯", "label": "Cohort Clusters", "value": f"{report.get('n_clusters', 4)} Cohorts"},
        {"icon": "📐", "label": "Silhouette Score", "value": f"{sil_score:.4f}"},
        {"icon": "📊", "label": "Davies-Bouldin Index", "value": f"{db_score:.4f}"},
    ]
    kpi_row(kpis)

    divider()

    section_label("Behavioral Cohort Profiles")

    cols = st.columns(len(clusters))
    tier_colors = {
        "Champions / High Value": "var(--success)",
        "Promising / Growing": "var(--primary)",
        "At-Risk High Value": "var(--warning)",
        "Dormant / Low Activity": "var(--danger)",
    }

    for idx, (c_name, c_data) in enumerate(clusters.items()):
        with cols[idx % len(cols)]:
            color = tier_colors.get(c_name, "var(--primary)")
            st.markdown(
                f"""
                <div class="ig-card" style="border-top: 4px solid {color}; margin-bottom: 1rem; padding: 1.1rem;">
                  <div style="font-weight:700; color:var(--text-main); font-size:1rem; margin-bottom:0.25rem;">
                    {c_name}
                  </div>
                  <div style="font-size:0.8rem; color:var(--text-muted); margin-bottom:0.75rem;">
                    {c_data.get('size', 0):,} accounts ({c_data.get('share', 0):.1%} of portfolio)
                  </div>
                  <div style="font-size:0.82rem; line-height:1.6;">
                    <div style="display:flex; justify-content:space-between; border-bottom:1px solid var(--border); padding:0.25rem 0;">
                      <span style="color:var(--text-muted);">Median Recency:</span>
                      <strong style="color:var(--text-main);">{int(c_data.get('median_recency', 0))} days</strong>
                    </div>
                    <div style="display:flex; justify-content:space-between; border-bottom:1px solid var(--border); padding:0.25rem 0;">
                      <span style="color:var(--text-muted);">Median Frequency:</span>
                      <strong style="color:var(--text-main);">{int(c_data.get('median_frequency', 0))} orders</strong>
                    </div>
                    <div style="display:flex; justify-content:space-between; border-bottom:1px solid var(--border); padding:0.25rem 0;">
                      <span style="color:var(--text-muted);">Median Spend:</span>
                      <strong style="color:var(--text-main);">${c_data.get('median_monetary', 0):,.0f}</strong>
                    </div>
                    <div style="display:flex; justify-content:space-between; padding:0.25rem 0;">
                      <span style="color:var(--text-muted);">Return Rate:</span>
                      <strong style="color:{'var(--danger)' if c_data.get('mean_cancellation_rate', 0) > 0.25 else 'var(--text-main)'};">
                        {c_data.get('mean_cancellation_rate', 0):.1%}
                      </strong>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    divider()

    section_label("Strategic Retention Recommendations by Cohort")
    st.markdown(
        """
        <div class="ig-card">
          <div style="margin-bottom:0.6rem; padding-bottom:0.4rem; border-bottom:1px solid var(--border);">
            <strong style="color:var(--success);">🏆 Champions / High Value:</strong> High engagement and frequency. Prioritize VIP customer service, loyalty rewards, and early access to high-demand inventory.
          </div>
          <div style="margin-bottom:0.6rem; padding-bottom:0.4rem; border-bottom:1px solid var(--border);">
            <strong style="color:var(--primary);">📈 Promising / Growing:</strong> Recent purchasers with moderate order frequency. Automate catalog cross-sell campaigns to increase average order volume.
          </div>
          <div style="margin-bottom:0.6rem; padding-bottom:0.4rem; border-bottom:1px solid var(--border);">
            <strong style="color:var(--warning);">⚠️ At-Risk High Value:</strong> Historically valuable buyers with elevated recency and high return rates. Assign account managers for service audit and targeted win-back outreach.
          </div>
          <div>
            <strong style="color:var(--danger);">💤 Dormant / Low Activity:</strong> Inactive for extended horizons (>300 days). Deploy automated re-engagement workflows or reallocate marketing capital.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_product_intelligence(artifacts: Dict[str, Any]) -> None:
    page_header(
        "Product Intelligence",
        "SKU velocity, transaction volume, revenue contribution, and return rates across 1M+ transactions.",
    )

    prod_file = REPORTS_DIR / "retail_product_intelligence.json"
    if not prod_file.exists():
        st.info("Product intelligence is compiling. Run 'py -m src.retail_analytics' to generate precomputed reports.")
        return

    try:
        report = json.loads(prod_file.read_text(encoding="utf-8"))
    except Exception as exc:
        st.error(f"Could not load product intelligence report: {exc}")
        return

    top_revenue = report.get("top_products_by_revenue", [])
    top_volume = report.get("top_products_by_volume", [])
    countries = report.get("country_distribution", [])
    total_skus = report.get("total_analyzed_products", 0)

    # KPI Row
    kpis = [
        {"icon": "📦", "label": "Analyzed SKUs", "value": f"{total_skus:,}"},
        {"icon": "🌍", "label": "Global Markets", "value": f"{len(countries)} Countries"},
        {"icon": "💎", "label": "Top SKU Revenue", "value": f"${top_revenue[0]['gross_revenue']:,.0f}" if top_revenue else "N/A"},
        {"icon": "⚡", "label": "Top SKU Volume", "value": f"{top_volume[0]['total_quantity']:,} units" if top_volume else "N/A"},
    ]
    kpi_row(kpis)

    divider()

    tab_rev, tab_vol, tab_geo = st.tabs([
        "TOP PRODUCTS BY REVENUE",
        "TOP PRODUCTS BY VOLUME",
        "GEOGRAPHIC DISTRIBUTION",
    ])

    with tab_rev:
        section_label("Leading Revenue Drivers")
        if top_revenue:
            rev_df = pd.DataFrame(top_revenue)
            rev_df = rev_df.rename(columns={
                "stock_code": "Stock Code",
                "description": "Product Description",
                "gross_revenue": "Gross Revenue ($)",
                "total_quantity": "Units Sold",
                "cancellation_rate": "Return Rate",
            })
            rev_df["Gross Revenue ($)"] = rev_df["Gross Revenue ($)"].apply(lambda x: f"${x:,.2f}")
            rev_df["Units Sold"] = rev_df["Units Sold"].apply(lambda x: f"{x:,}")
            rev_df["Return Rate"] = rev_df["Return Rate"].apply(lambda x: f"{x:.1%}")
            st.dataframe(rev_df, use_container_width=True, hide_index=True)

    with tab_vol:
        section_label("High Velocity SKUs")
        if top_volume:
            vol_df = pd.DataFrame(top_volume)
            vol_df = vol_df.rename(columns={
                "stock_code": "Stock Code",
                "description": "Product Description",
                "total_quantity": "Units Sold",
                "gross_revenue": "Gross Revenue ($)",
                "cancellation_rate": "Return Rate",
            })
            vol_df["Units Sold"] = vol_df["Units Sold"].apply(lambda x: f"{x:,}")
            vol_df["Gross Revenue ($)"] = vol_df["Gross Revenue ($)"].apply(lambda x: f"${x:,.2f}")
            vol_df["Return Rate"] = vol_df["Return Rate"].apply(lambda x: f"{x:.1%}")
            st.dataframe(vol_df, use_container_width=True, hide_index=True)

    with tab_geo:
        section_label("International Revenue & Order Contribution")
        if countries:
            geo_df = pd.DataFrame(countries)
            geo_df = geo_df.rename(columns={
                "country": "Country",
                "revenue": "Revenue ($)",
                "orders": "Total Orders",
                "customers": "Active Customers",
            })
            geo_df["Revenue ($)"] = geo_df["Revenue ($)"].apply(lambda x: f"${x:,.2f}")
            geo_df["Total Orders"] = geo_df["Total Orders"].apply(lambda x: f"{x:,}")
            geo_df["Active Customers"] = geo_df["Active Customers"].apply(lambda x: f"{x:,}")
            st.dataframe(geo_df, use_container_width=True, hide_index=True)
