"""Tests for UCI Online Retail II integration, Multi-Signal Risk Engine,
and Customer 360 domain isolation in InvoiceGuard AI.
"""

import os
import json
import pytest
import joblib
import numpy as np
import pandas as pd
from unittest.mock import patch

from src.risk_engine import RiskEngine
from src.customer_360 import Customer360Service, get_customer_profile


# =====================================================================
# 1. Retail Data Quality Report & Empirical Ingestion Statistics
# =====================================================================

def test_retail_data_quality_report_empirical_integrity():
    """Verify the empirical data quality report matches exact measured statistics."""
    report_path = os.path.join("reports", "retail_data_quality_report.json")
    assert os.path.exists(report_path), f"Report not found at {report_path}"

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    # Core empirical integrity checks
    assert report["dataset_name"] == "UCI Online Retail II"
    assert report["total_rows"] == 1067371
    assert report["duplicate_rows"] == 34335
    assert report["missing_customer_id_count"] == 243007
    assert report["missing_description_count"] == 4382
    assert report["cancellation_transactions_count"] == 19494
    assert report["negative_quantities_count"] == 22950
    assert report["zero_or_negative_price_count"] == 6207
    assert report["suspicious_stock_codes_count"] == 5351

    # Cardinality checks
    cardinality = report["cardinality"]
    assert cardinality["unique_customers"] == 5942
    assert cardinality["unique_invoices"] == 53628
    assert cardinality["unique_products"] == 5305
    assert cardinality["countries_count"] == 43

    # Date range integrity
    date_range = report["date_range"]
    assert date_range["min_date"] == "2009-12-01"
    assert date_range["max_date"] == "2011-12-09"


# =====================================================================
# 2. Multi-Signal Transparent Risk Engine
# =====================================================================

def test_risk_engine_score_bounds_and_weights():
    """Verify RiskEngine produces scores strictly within [0, 100] and maps tiers accurately."""
    engine = RiskEngine()

    # Low-risk scenario
    low_res = engine.calculate_business_risk_score(
        invoice_late_prob=0.05,
        retail_inactivity_prob=0.10,
        exposure_amount=500.0,
        invoice_amount=5000.0,
        cancellation_rate=0.01,
        recency_days=10.0,
        nlp_risk_prob=0.10
    )
    assert 0 <= low_res["composite_score"] <= 100
    assert low_res["risk_tier"] in ["LOW", "MEDIUM"]
    assert "contributing_signals" in low_res
    assert "signal_weights" in low_res

    # High-risk scenario
    high_res = engine.calculate_business_risk_score(
        invoice_late_prob=0.95,
        retail_inactivity_prob=0.90,
        exposure_amount=60000.0,
        invoice_amount=60000.0,
        cancellation_rate=0.45,
        recency_days=180.0,
        nlp_risk_prob=0.95
    )
    assert 0 <= high_res["composite_score"] <= 100
    assert high_res["risk_tier"] in ["HIGH", "CRITICAL"]
    assert high_res["composite_score"] > low_res["composite_score"]


def test_risk_engine_component_transparency():
    """Verify each component is explicitly exposed with normalized weight and signal."""
    engine = RiskEngine()
    res = engine.calculate_business_risk_score(
        invoice_late_prob=0.5,
        retail_inactivity_prob=0.5,
        exposure_amount=5000.0,
        invoice_amount=10000.0,
        cancellation_rate=0.1,
        recency_days=60.0,
        nlp_risk_prob=0.5
    )
    weights = res["signal_weights"]
    weights_sum = sum(weights.values())
    assert abs(weights_sum - 1.0) < 0.05
    assert "payment_risk" in weights
    assert "inactivity_risk" in weights
    assert "cancellation_risk" in weights
    assert "financial_exposure" in weights
    assert "nlp_communication_risk" in weights


def test_risk_engine_deterministic_recommendations():
    """Verify deterministic recommendations triggered by concrete thresholds."""
    engine = RiskEngine()

    # Trigger P1: high late probability
    p1_recs = engine.generate_recommendations(
        composite_score=85,
        invoice_late_prob=0.85,
        retail_inactivity_prob=0.2,
        cancellation_rate=0.05,
        recency_days=20.0,
        monetary=1000.0,
        exposure_amount=5000.0,
        nlp_risk_prob=0.8
    )
    assert len(p1_recs) > 0
    priorities = [r["priority"] for r in p1_recs]
    assert any("P1" in p for p in priorities)
    assert any("Accelerate Receivables Follow-Up" in r["action"] or "Senior Credit Controller Review" in r["action"] for r in p1_recs)

    # Trigger P2: High cancellation rate
    p2_recs = engine.generate_recommendations(
        composite_score=45,
        invoice_late_prob=0.1,
        retail_inactivity_prob=0.2,
        cancellation_rate=0.25,
        recency_days=15.0,
        monetary=500.0,
        exposure_amount=0.0,
        nlp_risk_prob=0.1
    )
    assert any("P2" in r["priority"] for r in p2_recs)
    assert any("Audit Order Fulfillment" in r["action"] for r in p2_recs)


# =====================================================================
# 3. Customer 360 Domain Isolation (Zero Fake Mapping)
# =====================================================================

def test_customer_360_domain_isolation():
    """Ensure strict domain isolation: B2B accounts and UCI retail accounts are separate."""
    service = Customer360Service()

    # Query an InvoiceGuard customer ID (synthetic/B2B convention e.g. C-0161)
    b2b_profile = service.get_customer_profile("C-0161")
    assert b2b_profile is not None
    assert "InvoiceGuard Receivables" in b2b_profile["data_sources"]
    assert "UCI Online Retail" not in b2b_profile["data_sources"]
    assert bool(b2b_profile["invoice_intelligence"]) is True
    assert bool(b2b_profile["retail_intelligence"]) is False

    # Query a Retail customer ID (UCI convention e.g. 13085)
    retail_profile = service.get_customer_profile("13085")
    assert retail_profile is not None
    assert "UCI Online Retail" in retail_profile["data_sources"]
    assert "InvoiceGuard Receivables" not in retail_profile["data_sources"]
    assert bool(retail_profile["retail_intelligence"]) is True
    assert bool(retail_profile["invoice_intelligence"]) is False

    # Unknown customer
    unknown_profile = service.get_customer_profile("NON_EXISTENT_99999")
    assert unknown_profile["data_sources"] == []
    assert bool(unknown_profile["invoice_intelligence"]) is False
    assert bool(unknown_profile["retail_intelligence"]) is False
    assert "error" in unknown_profile


def test_customer_360_domain_separation_guarantee_text():
    """Verify customer profile includes explicit domain separation disclosure."""
    service = Customer360Service()
    profile = service.get_customer_profile("13085")
    assert "domain_separation_guarantee" in profile
    assert "disjoint populations" in profile["domain_separation_guarantee"]


# =====================================================================
# 4. Customer Segmentation & Product Intelligence Reports
# =====================================================================

def test_segmentation_report_integrity():
    """Verify KMeans segmentation report exists and conforms to cluster standards."""
    seg_path = os.path.join("reports", "retail_segmentation_report.json")
    assert os.path.exists(seg_path)

    with open(seg_path, "r", encoding="utf-8") as f:
        seg_data = json.load(f)

    assert "KMeans" in seg_data["model_type"]
    assert seg_data["n_clusters"] == 4
    assert "silhouette_score" in seg_data
    assert "davies_bouldin_score" in seg_data
    assert len(seg_data["clusters"]) == 4

    # Ensure all 4 identified segments are present
    cluster_names = list(seg_data["clusters"].keys())
    assert "Champions / High Value" in cluster_names
    assert "Promising / Growing" in cluster_names
    assert "At-Risk High Value" in cluster_names
    assert "Dormant / Low Activity" in cluster_names


def test_product_intelligence_report_integrity():
    """Verify product intelligence report exists and contains top products and country data."""
    prod_path = os.path.join("reports", "retail_product_intelligence.json")
    assert os.path.exists(prod_path)

    with open(prod_path, "r", encoding="utf-8") as f:
        prod_data = json.load(f)

    assert prod_data["dataset"] == "UCI Online Retail II"
    assert "top_products_by_revenue" in prod_data
    assert "top_products_by_volume" in prod_data
    assert "country_distribution" in prod_data
    assert "total_analyzed_products" in prod_data
    assert len(prod_data["top_products_by_revenue"]) > 0
    assert len(prod_data["country_distribution"]) > 0


# =====================================================================
# 5. Retail Predictive Models & Leakage Prevention
# =====================================================================

def test_retail_models_prediction_and_bounds():
    """Verify retail models load correctly and output valid calibrated predictions."""
    repurchase_path = os.path.join("models", "retail_repurchase_model.pkl")
    future_rev_path = os.path.join("models", "retail_future_revenue_model.pkl")
    features_path = os.path.join("models", "retail_features.pkl")

    assert os.path.exists(repurchase_path)
    assert os.path.exists(future_rev_path)
    assert os.path.exists(features_path)

    repurchase_model = joblib.load(repurchase_path)
    revenue_model = joblib.load(future_rev_path)
    features = joblib.load(features_path)

    assert len(features) == 8

    # Test sample point-in-time RFM snapshot
    sample = pd.DataFrame([{
        "recency_days": 30.0,
        "customer_age_days": 180.0,
        "frequency": 4.0,
        "monetary": 1250.0,
        "avg_order_value": 312.5,
        "cancellation_rate": 0.05,
        "revenue_90d": 650.0,
        "orders_90d": 2.0
    }])[features]

    # Predict repurchase
    prob = repurchase_model.predict_proba(sample)[0, 1]
    assert 0.0 <= prob <= 1.0

    # Predict future revenue
    pred_rev = revenue_model.predict(sample)[0]
    assert np.isfinite(pred_rev)
    assert pred_rev >= 0.0


def test_model_metadata_lineage_and_leakage_declarations():
    """Verify both retail models have documented lineage and temporal split declarations."""
    repurchase_meta_path = os.path.join("models", "retail_repurchase_model_metadata.json")
    rev_meta_path = os.path.join("models", "retail_future_revenue_model_metadata.json")

    for meta_path in [repurchase_meta_path, rev_meta_path]:
        assert os.path.exists(meta_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["domain"] == "UCI Online Retail II"
        assert "temporal_validation" in meta
        assert "leakage_prevention" in meta
        assert "Point-in-time" in meta["leakage_prevention"]


# =====================================================================
# 6. Direct Functional Tests of API Logic (Without requiring httpx)
# =====================================================================

def test_api_functional_logic():
    """Verify API prediction functions execute without requiring external HTTP client."""
    from src.api import predict_retail_repurchase, predict_retail_revenue, evaluate_risk_score
    from src.api import RetailInput, RiskScoreInput

    # 1. Repurchase prediction
    r_in = RetailInput(
        recency_days=15.0,
        customer_age_days=200.0,
        frequency=5.0,
        monetary=1500.0,
        avg_order_value=300.0,
        cancellation_rate=0.02,
        revenue_90d=500.0,
        orders_90d=2.0
    )
    rep_res = predict_retail_repurchase(r_in)
    assert "repurchase_probability_60d" in rep_res
    assert "inactivity_risk_probability_60d" in rep_res
    assert "inactivity_tier" in rep_res
    assert 0.0 <= rep_res["repurchase_probability_60d"] <= 1.0

    # 2. Revenue prediction
    rev_res = predict_retail_revenue(r_in)
    assert "expected_future_revenue_60d" in rev_res
    assert rev_res["expected_future_revenue_60d"] >= 0.0

    # 3. Risk evaluate
    risk_in = RiskScoreInput(
        invoice_late_prob=0.4,
        retail_inactivity_prob=0.3,
        cancellation_rate=0.05,
        exposure_amount=2000.0,
        invoice_amount=10000.0,
        recency_days=30.0,
        nlp_risk_prob=0.35
    )
    eval_res = evaluate_risk_score(risk_in)
    assert "risk_evaluation" in eval_res
    assert "recommendations" in eval_res
    assert 0 <= eval_res["risk_evaluation"]["composite_score"] <= 100
