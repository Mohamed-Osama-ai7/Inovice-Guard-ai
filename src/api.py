from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.customer_360 import get_customer_profile
from src.risk_engine import RiskEngine

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
TEMPLATES = ROOT / "templates"

app = FastAPI(title="InvoiceGuard AI - Enterprise Receivables & Retail Intelligence API", version="2.0.0")

classifier = joblib.load(MODELS / "classifier.joblib") if (MODELS / "classifier.joblib").exists() else None
regressor = joblib.load(MODELS / "delay_regressor.joblib") if (MODELS / "delay_regressor.joblib").exists() else None
meta = json.loads((MODELS / "metadata.json").read_text(encoding="utf-8")) if (MODELS / "metadata.json").exists() else {}

retail_repurchase_model = (
    joblib.load(MODELS / "retail_repurchase_model.pkl") if (MODELS / "retail_repurchase_model.pkl").exists() else None
)
retail_revenue_model = (
    joblib.load(MODELS / "retail_future_revenue_model.pkl") if (MODELS / "retail_future_revenue_model.pkl").exists() else None
)
retail_features = (
    joblib.load(MODELS / "retail_features.pkl") if (MODELS / "retail_features.pkl").exists() else []
)


class InvoiceInput(BaseModel):
    invoice_date: str = Field(..., description="Invoice creation date (YYYY-MM-DD)")
    due_date: str = Field(..., description="Invoice due date (YYYY-MM-DD)")
    invoice_amount: float = Field(..., gt=0, description="Total invoice amount")
    customer_seen_before: int = Field(default=0, ge=0, description="Times this customer has been seen")
    prior_late_count: float = Field(default=0.0, ge=0.0, description="Number of prior late payments")
    prior_late_ratio: float = Field(default=0.0, ge=0.0, le=1.0, description="Ratio of late payments to total (0.0 to 1.0)")
    prior_avg_delay: float = Field(default=0.0, ge=0.0, description="Average delay in days")
    industry: Optional[str] = Field(default=None, description="Customer industry")
    company_size: Optional[str] = Field(default=None, description="Customer company size")
    payment_method: Optional[str] = Field(default=None, description="Expected payment method")
    customer_segment: Optional[str] = Field(default=None, description="Customer segment (e.g. SME, Enterprise)")
    outstanding_amount: float = Field(default=0.0, ge=0.0, description="Total outstanding balance for this customer")


class RetailInput(BaseModel):
    recency_days: int = Field(..., ge=0, description="Days since last customer transaction")
    frequency: int = Field(..., ge=0, description="Lifetime total orders count")
    monetary: float = Field(..., ge=0.0, description="Lifetime net spend")
    avg_order_value: float = Field(default=0.0, ge=0.0, description="Average order value")
    cancellation_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Ratio of cancelled orders")
    customer_age_days: int = Field(default=0, ge=0, description="Days since first transaction")
    revenue_90d: float = Field(default=0.0, ge=0.0, description="Revenue in the last 90 days")
    orders_90d: int = Field(default=0, ge=0, description="Order count in the last 90 days")


class RiskScoreInput(BaseModel):
    invoice_late_prob: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    retail_inactivity_prob: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    exposure_amount: float = Field(default=0.0, ge=0.0)
    invoice_amount: float = Field(default=0.0, ge=0.0)
    cancellation_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    recency_days: float = Field(default=0.0, ge=0.0)
    nlp_risk_prob: Optional[float] = Field(default=None, ge=0.0, le=1.0)


def row(i: InvoiceInput):
    inv = pd.Timestamp(i.invoice_date)
    due = pd.Timestamp(i.due_date)
    d = {
        "invoice_amount_clean": i.invoice_amount,
        "amount_log1p": float(np.log1p(i.invoice_amount)),
        "days_to_due": (due - inv).days,
        "invoice_year": inv.year,
        "invoice_month": inv.month,
        "invoice_quarter": inv.quarter,
        "invoice_dayofweek": inv.dayofweek,
        "customer_seen_before": i.customer_seen_before,
        "customer_is_new": int(i.customer_seen_before == 0),
        "prior_late_count": i.prior_late_count,
        "prior_late_ratio": i.prior_late_ratio,
        "prior_avg_delay": i.prior_avg_delay,
        "outstanding_amount": i.outstanding_amount,
    }
    for c in ["industry", "company_size", "payment_method", "customer_segment"]:
        d[c] = getattr(i, c)
    return pd.DataFrame([d])


@app.get("/", response_class=HTMLResponse)
def home():
    if (TEMPLATES / "index.html").exists():
        return (TEMPLATES / "index.html").read_text(encoding="utf8")
    return "<html><body><h2>InvoiceGuard AI Enterprise API</h2><p>Operational.</p></body></html>"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": classifier is not None,
        "invoice_classifier_loaded": classifier is not None,
        "invoice_regressor_loaded": regressor is not None,
        "retail_repurchase_model_loaded": retail_repurchase_model is not None,
        "retail_revenue_model_loaded": retail_revenue_model is not None,
        "best_model": meta.get("best_model"),
        "test_metrics": meta.get("test_metrics", {}),
    }


@app.get("/metrics")
def metrics():
    return {
        "best_model": meta.get("best_model"),
        "test_metrics": meta.get("test_metrics", {}),
        "regression_metrics": meta.get("regression_metrics", {}),
        "data_source": meta.get("source_file"),
    }


@app.post("/predict")
def predict(i: InvoiceInput):
    if classifier is None:
        raise HTTPException(status_code=503, detail="Invoice classifier model not loaded.")
    X = row(i)
    p = float(classifier.predict_proba(X)[:, 1][0])
    th = float(meta.get("threshold", 0.5))
    late = int(p >= th)
    delay = float(np.clip(regressor.predict(X)[0], 0, None)) if regressor else 0.0
    exposure = float(i.invoice_amount * p)
    risk = "LOW" if p < 0.35 else ("MEDIUM" if p < 0.65 else ("HIGH" if p < 0.85 else "CRITICAL"))
    reasons = []
    if i.prior_late_ratio >= 0.5:
        reasons.append("High historical late-payment ratio")
    if i.prior_avg_delay >= 10:
        reasons.append("High historical average delay")
    if i.outstanding_amount > i.invoice_amount:
        reasons.append("Outstanding balance exceeds current invoice amount")
    if i.invoice_amount >= 100000:
        reasons.append("Large invoice amount")
    if i.customer_seen_before == 0:
        reasons.append("New customer: prediction relies on profile/invoice features")
    rec = (
        "Prioritize collection follow-up before the due date."
        if risk in ("HIGH", "CRITICAL")
        else "Monitor payment and schedule routine follow-up."
    )
    return {
        "late_probability": round(p, 4),
        "predicted_late": late,
        "expected_delay_days": round(delay, 1),
        "risk_level": risk,
        "estimated_financial_exposure": round(exposure, 2),
        "explanation": reasons[:4],
        "recommendation": rec,
    }


@app.post("/retail/predict/repurchase")
def predict_retail_repurchase(r: RetailInput):
    if retail_repurchase_model is None:
        raise HTTPException(status_code=503, detail="Retail repurchase model not loaded.")
    feat_dict = {
        "recency_days": r.recency_days,
        "customer_age_days": r.customer_age_days,
        "frequency": r.frequency,
        "monetary": r.monetary,
        "avg_order_value": r.avg_order_value,
        "cancellation_rate": r.cancellation_rate,
        "revenue_90d": r.revenue_90d,
        "orders_90d": r.orders_90d,
    }
    x_df = pd.DataFrame([feat_dict])[retail_features].fillna(0)
    prob = float(retail_repurchase_model.predict_proba(x_df)[:, 1][0])
    inact = 1.0 - prob
    return {
        "repurchase_probability_60d": round(prob, 4),
        "inactivity_risk_probability_60d": round(inact, 4),
        "inactivity_tier": "HIGH" if inact >= 0.70 else ("MEDIUM" if inact >= 0.35 else "LOW"),
    }


@app.post("/retail/predict/revenue")
def predict_retail_revenue(r: RetailInput):
    if retail_revenue_model is None:
        raise HTTPException(status_code=503, detail="Retail future revenue model not loaded.")
    feat_dict = {
        "recency_days": r.recency_days,
        "customer_age_days": r.customer_age_days,
        "frequency": r.frequency,
        "monetary": r.monetary,
        "avg_order_value": r.avg_order_value,
        "cancellation_rate": r.cancellation_rate,
        "revenue_90d": r.revenue_90d,
        "orders_90d": r.orders_90d,
    }
    x_df = pd.DataFrame([feat_dict])[retail_features].fillna(0)
    pred_rev = float(retail_revenue_model.predict(x_df)[0])
    return {
        "expected_future_revenue_60d": round(max(0.0, pred_rev), 2),
    }


@app.post("/risk/evaluate")
def evaluate_risk_score(s: RiskScoreInput):
    score_res = RiskEngine.calculate_business_risk_score(
        invoice_late_prob=s.invoice_late_prob,
        retail_inactivity_prob=s.retail_inactivity_prob,
        exposure_amount=s.exposure_amount,
        invoice_amount=s.invoice_amount,
        cancellation_rate=s.cancellation_rate,
        recency_days=s.recency_days,
        nlp_risk_prob=s.nlp_risk_prob,
    )
    recs = RiskEngine.generate_recommendations(
        composite_score=score_res["composite_score"],
        invoice_late_prob=s.invoice_late_prob,
        retail_inactivity_prob=s.retail_inactivity_prob,
        cancellation_rate=s.cancellation_rate,
        recency_days=s.recency_days,
        exposure_amount=s.exposure_amount,
        nlp_risk_prob=s.nlp_risk_prob,
    )
    return {
        "risk_evaluation": score_res,
        "recommendations": recs,
    }


@app.get("/customer/profile/{customer_id}")
def customer_profile(customer_id: str):
    profile = get_customer_profile(customer_id)
    if not profile.get("data_sources"):
        raise HTTPException(status_code=404, detail="Customer not found in either domain.")
    return profile
