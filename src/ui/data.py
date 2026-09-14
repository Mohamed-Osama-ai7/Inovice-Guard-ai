from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

DEFAULT_CATEGORICALS = {
    "industry": ["Construction", "Retail", "IT", "Manufacturing", "Services"],
    "company_size": ["Small", "Medium", "Large"],
    "payment_method": ["Bank Transfer", "Cheque", "Card", "Cash"],
    "customer_segment": ["SME", "Mid-Market", "Enterprise"],
}

DEFAULT_FEATURES = [
    "invoice_amount_clean",
    "amount_log1p",
    "days_to_due",
    "invoice_year",
    "invoice_month",
    "invoice_quarter",
    "invoice_dayofweek",
    "customer_seen_before",
    "customer_is_new",
    "prior_late_count",
    "prior_late_ratio",
    "prior_avg_delay",
    "outstanding_amount",
    "industry",
    "company_size",
    "payment_method",
    "customer_segment",
]


def get_artifact_signature() -> str:
    model_files = {
        "classifier": MODELS / "classifier.joblib",
        "delay_regressor": MODELS / "delay_regressor.joblib",
        "logistic_regression": MODELS / "logistic_regression.joblib",
        "random_forest": MODELS / "random_forest.joblib",
        "mlp_neural_network": MODELS / "mlp_neural_network.joblib",
        "nlp_payment_risk": MODELS / "nlp_payment_risk.joblib",
    }
    signature: Dict[str, Dict[str, int]] = {}
    for name, path in model_files.items():
        if not path.exists():
            continue
        stat = path.stat()
        signature[name] = {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size}
    return json.dumps(signature, sort_keys=True)


def build_validation_row(metadata: Dict[str, Any]) -> Dict[str, Any]:
    feature_order = metadata.get("features") or DEFAULT_FEATURES
    row: Dict[str, Any] = {
        "invoice_amount_clean": 1000.0,
        "amount_log1p": float(np.log1p(1000.0)),
        "days_to_due": 30.0,
        "invoice_year": 2025,
        "invoice_month": 1,
        "invoice_quarter": 1,
        "invoice_dayofweek": 0,
        "customer_seen_before": 0,
        "customer_is_new": 1,
        "prior_late_count": 0.0,
        "prior_late_ratio": 0.0,
        "prior_avg_delay": 0.0,
        "outstanding_amount": 0.0,
    }
    for column in ["industry", "company_size", "payment_method", "customer_segment"]:
        row[column] = DEFAULT_CATEGORICALS.get(column, ["Retail"])[0]
    for column in feature_order:
        row.setdefault(column, 0.0)
    return {column: row.get(column, 0.0) for column in feature_order}


def validate_loaded_model(name: str, model: Any, metadata: Dict[str, Any]) -> None:
    if name == "nlp_payment_risk":
        if not hasattr(model, "predict_proba"):
            raise RuntimeError("NLP model does not expose predict_proba().")
        model.predict_proba(["Payment has been scheduled and will be completed on the agreed date."])
        return
    validation_frame = pd.DataFrame([build_validation_row(metadata)])
    if hasattr(model, "predict_proba"):
        model.predict_proba(validation_frame)
        return
    if hasattr(model, "predict"):
        model.predict(validation_frame)
        return
    raise RuntimeError("Model does not expose a usable prediction interface.")


@st.cache_resource
def load_project_artifacts(artifact_signature: str) -> Dict[str, Any]:
    metadata_path = MODELS / "metadata.json"
    metadata: Dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    model_files = {
        "classifier": MODELS / "classifier.joblib",
        "delay_regressor": MODELS / "delay_regressor.joblib",
        "logistic_regression": MODELS / "logistic_regression.joblib",
        "random_forest": MODELS / "random_forest.joblib",
        "mlp_neural_network": MODELS / "mlp_neural_network.joblib",
        "nlp_payment_risk": MODELS / "nlp_payment_risk.joblib",
    }
    loaded_models: Dict[str, Any] = {}
    load_errors: Dict[str, str] = {}
    for name, path in model_files.items():
        if not path.exists():
            continue
        try:
            model = joblib.load(path)
            validate_loaded_model(name, model, metadata)
            loaded_models[name] = model
        except Exception as exc:
            load_errors[name] = f"{type(exc).__name__}: {exc}"

    reports = {}
    for report_name in [
        "test_metrics.json",
        "regression_metrics.json",
        "all_model_artifacts.json",
        "nlp_metrics.json",
        "nlp_thresholds.json",
    ]:
        path = REPORTS / report_name
        if path.exists():
            reports[report_name] = json.loads(path.read_text(encoding="utf-8"))

    return {
        "metadata": metadata,
        "models": loaded_models,
        "reports": reports,
        "load_errors": load_errors,
    }


def build_feature_dict(payload: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    invoice_date = pd.Timestamp(payload["invoice_date"])
    due_date = pd.Timestamp(payload["due_date"])
    row: Dict[str, Any] = {
        "invoice_amount_clean": float(payload.get("invoice_amount", 0) or 0),
        "amount_log1p": float(np.log1p(max(float(payload.get("invoice_amount", 0) or 0), 0))),
        "days_to_due": float((due_date - invoice_date).days),
        "invoice_year": int(invoice_date.year),
        "invoice_month": int(invoice_date.month),
        "invoice_quarter": int(invoice_date.quarter),
        "invoice_dayofweek": int(invoice_date.dayofweek),
        "customer_seen_before": int(payload.get("customer_seen_before", 0) or 0),
        "customer_is_new": int((payload.get("customer_seen_before", 0) or 0) == 0),
        "prior_late_count": float(payload.get("prior_late_count", 0) or 0),
        "prior_late_ratio": float(payload.get("prior_late_ratio", 0) or 0),
        "prior_avg_delay": float(payload.get("prior_avg_delay", 0) or 0),
        "outstanding_amount": float(payload.get("outstanding_amount", 0) or 0),
    }
    for col in ["industry", "company_size", "payment_method", "customer_segment"]:
        v = payload.get(col)
        row[col] = v if v not in (None, "") else np.nan
    feature_order = metadata.get("features") or DEFAULT_FEATURES
    return {col: row.get(col, np.nan) for col in feature_order}


def run_prediction(
    payload: Dict[str, Any], selected_model: str, artifacts: Dict[str, Any]
) -> Dict[str, Any]:
    metadata = artifacts["metadata"]
    models = artifacts["models"]
    if "classifier" not in models and selected_model == "classifier":
        raise RuntimeError(
            "No trained classifier was found in /models. Run python -m src.train --source demo first."
        )
    model = models.get(selected_model)
    if model is None:
        raise RuntimeError(f"Selected model '{selected_model}' was not loaded.")
    feature_frame = pd.DataFrame([build_feature_dict(payload, metadata)])
    if not hasattr(model, "predict_proba"):
        raise RuntimeError(f"Model '{selected_model}' does not support predict_proba().")
    probability = float(model.predict_proba(feature_frame)[0, 1])

    delay_model = models.get("delay_regressor")
    delay_days = 0.0
    if delay_model is not None:
        try:
            delay_days = float(np.clip(delay_model.predict(feature_frame)[0], 0, None))
        except Exception:
            delay_days = 0.0

    if probability < 0.35:
        risk_level = "LOW"
    elif probability < 0.65:
        risk_level = "MEDIUM"
    elif probability < 0.85:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"

    exposure = float((payload.get("invoice_amount", 0) or 0) * probability)
    recommendation = (
        "Prioritize collection follow-up before the due date."
        if risk_level in ["HIGH", "CRITICAL"]
        else "Monitor payment and schedule routine follow-up."
    )
    return {
        "late_probability": probability,
        "risk_level": risk_level,
        "expected_delay_days": delay_days,
        "estimated_financial_exposure": exposure,
        "recommendation": recommendation,
        "feature_frame": feature_frame,
    }


@st.cache_data(show_spinner=False)
def _cached_build_dashboard_dataset(artifact_signature: str, _artifacts: Dict[str, Any]) -> pd.DataFrame:
    demo_csv = ROOT / "data" / "demo" / "demo_invoices.csv"
    if not demo_csv.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(demo_csv)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()

    if "invoice_id" not in df.columns:
        df["invoice_id"] = [f"INV-{i:06d}" for i in range(len(df))]
    if "customer_id" in df.columns:
        df["customer"] = df["customer_id"]
    elif "customer" not in df.columns:
        df["customer"] = "UNKNOWN"

    for col in ["invoice_date", "due_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ["invoice_amount", "outstanding_amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    models = _artifacts.get("models", {})
    metadata = _artifacts.get("metadata", {})
    classifier = models.get("classifier")
    if classifier is None or not hasattr(classifier, "predict_proba"):
        return pd.DataFrame()

    feature_order = metadata.get("features") or DEFAULT_FEATURES
    batch_df = df.copy()
    if "invoice_amount_clean" not in batch_df.columns:
        if "invoice_amount" in batch_df.columns:
            batch_df["invoice_amount_clean"] = pd.to_numeric(batch_df["invoice_amount"], errors="coerce").fillna(0)
        else:
            batch_df["invoice_amount_clean"] = 0.0

    batch_df["amount_log1p"] = np.log1p(np.maximum(batch_df["invoice_amount_clean"].values, 0))
    
    invoice_dates = pd.to_datetime(batch_df.get("invoice_date"), errors="coerce")
    due_dates = pd.to_datetime(batch_df.get("due_date"), errors="coerce")
    batch_df["days_to_due"] = (due_dates - invoice_dates).dt.days.fillna(0).astype(float)
    batch_df["invoice_year"] = invoice_dates.dt.year.fillna(2025).astype(int)
    batch_df["invoice_month"] = invoice_dates.dt.month.fillna(1).astype(int)
    batch_df["invoice_quarter"] = invoice_dates.dt.quarter.fillna(1).astype(int)
    batch_df["invoice_dayofweek"] = invoice_dates.dt.dayofweek.fillna(0).astype(int)

    if "customer_seen_before" in batch_df.columns:
        seen = pd.to_numeric(batch_df["customer_seen_before"], errors="coerce").fillna(0).astype(int)
    else:
        seen = pd.Series(0, index=batch_df.index)
        
    batch_df["customer_seen_before"] = seen
    batch_df["customer_is_new"] = (seen == 0).astype(int)
    
    for col in ["prior_late_count", "prior_late_ratio", "prior_avg_delay", "outstanding_amount"]:
        if col in batch_df.columns:
            batch_df[col] = pd.to_numeric(batch_df[col], errors="coerce").fillna(0)
        else:
            batch_df[col] = 0.0

    for col in ["industry", "company_size", "payment_method", "customer_segment"]:
        if col not in batch_df.columns:
            batch_df[col] = np.nan

    X_batch = batch_df[feature_order]
    late_probs = classifier.predict_proba(X_batch)[:, 1]
    
    delay_model = models.get("delay_regressor")
    if delay_model is not None and hasattr(delay_model, "predict"):
        try:
            expected_delays = np.clip(delay_model.predict(X_batch), 0, None)
        except Exception:
            expected_delays = np.zeros(len(df))
    else:
        expected_delays = np.zeros(len(df))

    conditions = [
        late_probs < 0.35,
        late_probs < 0.65,
        late_probs < 0.85
    ]
    choices = ["LOW", "MEDIUM", "HIGH"]
    risk_levels = np.select(conditions, choices, default="CRITICAL")

    inv_amounts = pd.to_numeric(df.get("invoice_amount", 0), errors="coerce").fillna(0).values
    exposures = inv_amounts * late_probs
    recommendations = np.where(
        np.isin(risk_levels, ["HIGH", "CRITICAL"]),
        "Prioritize collection follow-up before the due date.",
        "Monitor payment and schedule routine follow-up."
    )

    threshold = metadata.get("threshold", 0.5)
    predicted_late = (late_probs >= threshold).astype(int)
    days_to_due = (due_dates - invoice_dates).dt.days.fillna(0).astype(int)

    analytics = df.copy()
    analytics["late_probability"] = late_probs
    analytics["expected_delay_days"] = expected_delays
    analytics["risk_level"] = risk_levels
    analytics["estimated_financial_exposure"] = exposures
    analytics["recommendation"] = recommendations
    analytics["predicted_late"] = predicted_late
    analytics["days_to_due"] = days_to_due

    out_amounts = pd.to_numeric(analytics.get("outstanding_amount", 0), errors="coerce").fillna(0)
    analytics["amount_at_risk"] = out_amounts.where(
        analytics["risk_level"].isin(["HIGH", "CRITICAL"]), 0
    )
    analytics["due_soon"] = analytics["days_to_due"].le(7)
    return analytics


def build_dashboard_dataset(artifacts: Dict[str, Any]) -> pd.DataFrame:
    return _cached_build_dashboard_dataset(get_artifact_signature(), artifacts)


def get_dashboard_dataset(artifacts: Dict[str, Any]) -> pd.DataFrame:
    return build_dashboard_dataset(artifacts)
