"""
InvoiceGuard AI - Retail Analytics: Customer Segmentation & Product Intelligence
Provides point-in-time RFM clustering (KMeans), cluster evaluation (silhouette, Davies-Bouldin),
and product-level risk and revenue intelligence on the UCI Online Retail II dataset.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("invoiceguard.retail.analytics")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT / "reports"

SNAPSHOTS_PATH = PROCESSED_DIR / "retail_customer_snapshots.csv"
CLEANED_PARQUET_PATH = PROCESSED_DIR / "online_retail_II_cleaned.parquet"
SEGMENTATION_REPORT_PATH = REPORTS_DIR / "retail_segmentation_report.json"
PRODUCT_REPORT_PATH = REPORTS_DIR / "retail_product_intelligence.json"


def compute_customer_segmentation(
    snapshots_df: Optional[pd.DataFrame] = None,
    n_clusters: int = 4,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fits KMeans on RFM features (recency_days, frequency, monetary, cancellation_rate).
    Computes silhouette score and Davies-Bouldin index.
    Assigns business-interpretable segment labels.
    """
    if snapshots_df is None:
        if not SNAPSHOTS_PATH.exists():
            raise FileNotFoundError(f"Snapshots dataset not found at {SNAPSHOTS_PATH}")
        df = pd.read_csv(SNAPSHOTS_PATH)
    else:
        df = snapshots_df.copy()

    # Use the latest snapshot per customer to avoid duplicate customer representation
    if "snapshot_date" in df.columns:
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
        customer_df = df.sort_values("snapshot_date").groupby("customer_id").tail(1).copy()
    else:
        customer_df = df.copy()

    feature_cols = ["recency_days", "frequency", "monetary", "cancellation_rate"]
    X = customer_df[feature_cols].copy().fillna(0)

    # Log-transform highly skewed monetary and frequency features for cluster stability
    X_scaled_input = X.copy()
    X_scaled_input["monetary"] = np.log1p(np.maximum(0, X_scaled_input["monetary"]))
    X_scaled_input["frequency"] = np.log1p(np.maximum(0, X_scaled_input["frequency"]))

    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_scaled_input)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_norm)
    customer_df["cluster_id"] = cluster_labels

    # Evaluate clusters on a sample if dataset is large (for fast execution)
    sample_size = min(len(X_norm), 3000)
    sample_indices = np.random.RandomState(random_state).choice(len(X_norm), sample_size, replace=False)
    sil_score = float(silhouette_score(X_norm[sample_indices], cluster_labels[sample_indices]))
    db_score = float(davies_bouldin_score(X_norm, cluster_labels))

    # Profile clusters to assign interpretable business labels
    cluster_profiles: Dict[int, Dict[str, Any]] = {}
    for cid in range(n_clusters):
        c_subset = customer_df[customer_df["cluster_id"] == cid]
        cluster_profiles[cid] = {
            "size": len(c_subset),
            "share": round(len(c_subset) / len(customer_df), 4),
            "median_recency": float(c_subset["recency_days"].median()),
            "median_frequency": float(c_subset["frequency"].median()),
            "median_monetary": float(c_subset["monetary"].median()),
            "mean_cancellation_rate": float(c_subset["cancellation_rate"].mean()),
        }

    # Assign business labels based on relative median monetary and recency
    # Order by monetary desc
    sorted_by_monetary = sorted(cluster_profiles.keys(), key=lambda c: cluster_profiles[c]["median_monetary"], reverse=True)
    
    label_map: Dict[int, str] = {}
    top_cid = sorted_by_monetary[0]
    second_cid = sorted_by_monetary[1]
    third_cid = sorted_by_monetary[2]
    fourth_cid = sorted_by_monetary[3]

    label_map[top_cid] = "Champions / High Value"
    
    # Check if second cluster has higher recency (at risk) or lower recency
    if cluster_profiles[second_cid]["median_recency"] > cluster_profiles[third_cid]["median_recency"]:
        label_map[second_cid] = "At-Risk High Value"
        label_map[third_cid] = "Promising / Growing"
    else:
        label_map[second_cid] = "Promising / Growing"
        label_map[third_cid] = "At-Risk High Value"
        
    label_map[fourth_cid] = "Dormant / Low Activity"

    customer_df["segment_name"] = customer_df["cluster_id"].map(label_map)

    report: Dict[str, Any] = {
        "model_type": "KMeans (RFM + Cancellation Behavior)",
        "n_clusters": n_clusters,
        "silhouette_score": round(sil_score, 4),
        "davies_bouldin_score": round(db_score, 4),
        "evaluated_customers": len(customer_df),
        "clusters": {
            label_map[cid]: {
                **cluster_profiles[cid],
                "cluster_id": cid,
            }
            for cid in range(n_clusters)
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SEGMENTATION_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Saved segmentation report to %s", SEGMENTATION_REPORT_PATH)

    return customer_df, report


def compute_product_intelligence(
    cleaned_df: Optional[pd.DataFrame] = None,
    top_n: int = 50,
) -> Dict[str, Any]:
    """
    Computes top products by volume, revenue, return/cancellation rates,
    and country-level distribution from the cleaned transaction records.
    Serializes aggregated results to JSON for instant sub-millisecond UI rendering.
    """
    if cleaned_df is None:
        if CLEANED_PARQUET_PATH.exists():
            df = pd.read_parquet(CLEANED_PARQUET_PATH)
        elif (PROCESSED_DIR / "online_retail_II_cleaned.csv").exists():
            df = pd.read_csv(PROCESSED_DIR / "online_retail_II_cleaned.csv")
        else:
            raise FileNotFoundError("Cleaned transaction dataset not found. Run retail_ingestion first.")
    else:
        df = cleaned_df.copy()

    # Product aggregation
    prod_group = df.groupby(["stock_code", "description"]).agg(
        total_quantity=("quantity", "sum"),
        gross_revenue=("revenue", "sum"),
        transactions_count=("invoice_no", "count"),
        cancellations_count=("is_cancellation", "sum"),
    ).reset_index()

    prod_group["cancellation_rate"] = np.where(
        prod_group["transactions_count"] > 0,
        prod_group["cancellations_count"] / prod_group["transactions_count"],
        0.0,
    )

    # Top by revenue
    top_revenue_df = prod_group.sort_values("gross_revenue", ascending=False).head(top_n)
    top_revenue_items = [
        {
            "stock_code": str(row["stock_code"]),
            "description": str(row["description"]),
            "gross_revenue": round(float(row["gross_revenue"]), 2),
            "total_quantity": int(row["total_quantity"]),
            "cancellation_rate": round(float(row["cancellation_rate"]), 4),
        }
        for _, row in top_revenue_df.iterrows()
    ]

    # Top by volume
    top_volume_df = prod_group.sort_values("total_quantity", ascending=False).head(top_n)
    top_volume_items = [
        {
            "stock_code": str(row["stock_code"]),
            "description": str(row["description"]),
            "total_quantity": int(row["total_quantity"]),
            "gross_revenue": round(float(row["gross_revenue"]), 2),
            "cancellation_rate": round(float(row["cancellation_rate"]), 4),
        }
        for _, row in top_volume_df.iterrows()
    ]

    # Country aggregation
    country_group = df.groupby("country").agg(
        revenue=("revenue", "sum"),
        orders=("invoice_no", "nunique"),
        customers=("customer_id", lambda s: s[s != ""].nunique()),
    ).reset_index().sort_values("revenue", ascending=False)

    country_summary = [
        {
            "country": str(row["country"]),
            "revenue": round(float(row["revenue"]), 2),
            "orders": int(row["orders"]),
            "customers": int(row["customers"]),
        }
        for _, row in country_group.head(15).iterrows()
    ]

    report: Dict[str, Any] = {
        "dataset": "UCI Online Retail II",
        "top_products_by_revenue": top_revenue_items,
        "top_products_by_volume": top_volume_items,
        "country_distribution": country_summary,
        "total_analyzed_products": int(prod_group["stock_code"].nunique()),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PRODUCT_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Saved product intelligence report to %s", PRODUCT_REPORT_PATH)

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("Running Customer Segmentation...")
    _, seg_report = compute_customer_segmentation()
    print("Segmentation complete. Silhouette Score:", seg_report["silhouette_score"])
    print("Running Product Intelligence...")
    prod_report = compute_product_intelligence()
    print("Product Intelligence complete. Top revenue items:", len(prod_report["top_products_by_revenue"]))
