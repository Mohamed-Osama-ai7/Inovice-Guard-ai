from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    import shap
except Exception:  # pragma: no cover - SHAP is optional in demo environments
    shap = None

ROOT = Path(__file__).resolve().parent
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

st.set_page_config(
    page_title="InvoiceGuard AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# BUSINESS LOGIC  (unchanged — do not modify)
# ─────────────────────────────────────────────────────────────────────────────

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


@st.cache_data
def load_demo_options() -> Dict[str, List[str]]:
    options = {key: list(value) for key, value in DEFAULT_CATEGORICALS.items()}
    demo_csv = ROOT / "data" / "demo" / "demo_invoices.csv"
    if not demo_csv.exists():
        return options
    try:
        df = pd.read_csv(demo_csv)
    except Exception:
        return options
    for key in options:
        if key in df.columns:
            values = pd.Series(df[key].dropna().astype(str).unique()).sort_values().tolist()
            if values:
                options[key] = values
    return options


@st.cache_data
def load_text_examples() -> List[str]:
    nlp_metrics = REPORTS / "nlp_metrics.json"
    if nlp_metrics.exists():
        try:
            data = json.loads(nlp_metrics.read_text(encoding="utf-8"))
            if data.get("samples", 0):
                return [
                    "We expect a short delay while the payment receives internal approval.",
                    "Payment has been scheduled and will be completed on the agreed date.",
                    "There is a temporary cash flow constraint and the payment will be late.",
                ]
        except Exception:
            pass
    return [
        "We expect a short delay while the payment receives internal approval.",
        "Payment has been scheduled and will be completed on the agreed date.",
        "There is a temporary cash flow constraint and the payment will be late.",
    ]


@st.cache_data
def load_nlp_thresholds() -> Dict[str, float]:
    path = REPORTS / "nlp_thresholds.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {
                    "high": float(data.get("high", 0.5)),
                    "medium": float(data.get("medium", 0.45)),
                }
        except Exception:
            pass
    return {"high": 0.5, "medium": 0.45}


def get_nlp_class_order(model: Any) -> List[str]:
    if hasattr(model, "classes_"):
        return [str(v) for v in model.classes_]
    if hasattr(model, "named_steps"):
        clf = model.named_steps.get("classifier")
        if clf is not None and hasattr(clf, "classes_"):
            return [str(v) for v in clf.classes_]
    return ["HIGH_RISK", "LOW_RISK", "MEDIUM_RISK"]


def score_nlp_message(
    model: Any, customer_message: str, thresholds: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    message = (customer_message or "").strip()
    if not message:
        raise ValueError("Customer message cannot be empty.")
    try:
        probabilities = np.asarray(model.predict_proba([message])[0], dtype=float)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Unable to evaluate NLP message: {exc}") from exc

    class_order = get_nlp_class_order(model)
    probability_map = dict(zip(class_order, probabilities.tolist()))

    if len(probabilities) < 3:
        raise RuntimeError("NLP model did not return the expected three-class probability vector.")

    resolved_thresholds = thresholds or {"high": 0.5, "medium": 0.45}
    high_p = float(probability_map.get("HIGH_RISK", 0.0))
    med_p = float(probability_map.get("MEDIUM_RISK", 0.0))
    low_p = float(probability_map.get("LOW_RISK", 0.0))

    if high_p >= float(resolved_thresholds.get("high", 0.5)):
        risk_label, selected_p = "HIGH RISK", high_p
    elif med_p >= float(resolved_thresholds.get("medium", 0.45)):
        risk_label, selected_p = "MEDIUM RISK", med_p
    else:
        risk_label, selected_p = "LOW RISK", low_p

    return {
        "risk_label": risk_label,
        "probability": selected_p,
        "probabilities": probability_map,
        "thresholds": resolved_thresholds,
    }


def validate_single_input(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    try:
        invoice_date = pd.Timestamp(payload.get("invoice_date"))
        due_date = pd.Timestamp(payload.get("due_date"))
    except Exception:
        errors.append("Please provide valid invoice_date and due_date values.")
        return errors
    if due_date < invoice_date:
        errors.append("due_date cannot be earlier than invoice_date.")
    if float(payload.get("invoice_amount", 0) or 0) <= 0:
        errors.append("invoice_amount must be greater than zero.")
    if float(payload.get("outstanding_amount", 0) or 0) < 0:
        errors.append("outstanding_amount cannot be negative.")
    if int(payload.get("customer_seen_before", 0) or 0) < 0:
        errors.append("customer_seen_before cannot be negative.")
    for field in ["prior_late_count", "prior_avg_delay"]:
        if float(payload.get(field, 0) or 0) < 0:
            errors.append(f"{field} cannot be negative.")
    r = float(payload.get("prior_late_ratio", 0) or 0)
    if r < 0 or r > 1:
        errors.append("prior_late_ratio must be between 0 and 1.")
    for field in ["industry", "company_size", "payment_method", "customer_segment"]:
        val = payload.get(field)
        if val is not None and isinstance(val, str) and not val.strip():
            errors.append(f"{field} cannot be empty.")
    return errors


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


def format_top_features(
    model: Any, feature_frame: pd.DataFrame, metadata: Dict[str, Any]
) -> List[Dict[str, Any]]:
    if shap is None:
        return []
    try:
        if hasattr(model, "named_steps") and "prep" in model.named_steps:
            preprocessor = model.named_steps["prep"]
            transformed = preprocessor.transform(feature_frame)
            transformed_names = preprocessor.get_feature_names_out()
            explainer = shap.Explainer(model.named_steps["model"], transformed)
            sv = explainer(transformed)
            values = np.asarray(sv.values)
            if values.ndim == 3:
                values = values[:, :, 1] if values.shape[-1] > 1 else values[:, :, 0]
            if values.ndim == 2:
                values = values[0]
            if len(values) != len(transformed_names):
                raise ValueError("Unexpected SHAP output shape.")
            scores = np.abs(values)
            top_indices = np.argsort(scores)[::-1][:5]
            return [
                {
                    "feature": transformed_names[i],
                    "impact": float(np.abs(values[i])),
                    "direction": "increases risk" if values[i] >= 0 else "reduces risk",
                }
                for i in top_indices
            ]
        explainer = shap.Explainer(model)
        sv = explainer(feature_frame)
        values = np.asarray(sv.values)
        if values.ndim == 3:
            values = values[:, :, 1] if values.shape[-1] > 1 else values[:, :, 0]
        if values.ndim == 2:
            values = values[0]
        names = list(feature_frame.columns)
        scores = np.abs(values)
        top_indices = np.argsort(scores)[::-1][:5]
        return [
            {
                "feature": names[i],
                "impact": float(np.abs(values[i])),
                "direction": "increases risk" if values[i] >= 0 else "reduces risk",
            }
            for i in top_indices
        ]
    except Exception:
        return []


def generate_fallback_top_features(
    model: Any, feature_frame: pd.DataFrame
) -> List[Dict[str, Any]]:
    if hasattr(model, "feature_importances_"):
        names = list(feature_frame.columns)
        scores = np.asarray(model.feature_importances_, dtype=float)
        top_indices = np.argsort(scores)[::-1][:5]
        return [
            {
                "feature": names[i],
                "impact": float(scores[i]),
                "direction": "increases risk",
            }
            for i in top_indices
        ]
    if hasattr(model, "named_steps"):
        est = model.named_steps.get("model")
        if est is not None and hasattr(est, "coef_"):
            coef = np.asarray(est.coef_).reshape(-1)
            names = list(feature_frame.columns)
            top_indices = np.argsort(np.abs(coef))[::-1][:5]
            return [
                {
                    "feature": names[i],
                    "impact": float(np.abs(coef[i])),
                    "direction": "increases risk" if coef[i] >= 0 else "reduces risk",
                }
                for i in top_indices
            ]
    return []


def explain_prediction(
    model: Any, feature_frame: pd.DataFrame, metadata: Dict[str, Any]
) -> List[Dict[str, Any]]:
    top = format_top_features(model, feature_frame, metadata)
    if top:
        return top
    return generate_fallback_top_features(model, feature_frame)


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

    metadata = _artifacts.get("metadata", {})
    predictions: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        payload = row.to_dict()
        try:
            pred = run_prediction(payload, "classifier", _artifacts)
        except Exception:
            continue
        record = row.to_dict()
        record["late_probability"] = float(pred["late_probability"])
        record["expected_delay_days"] = float(pred["expected_delay_days"])
        record["risk_level"] = pred["risk_level"]
        record["estimated_financial_exposure"] = float(pred["estimated_financial_exposure"])
        record["recommendation"] = pred["recommendation"]
        record["predicted_late"] = int(pred["late_probability"] >= metadata.get("threshold", 0.5))
        try:
            record["days_to_due"] = int(
                (pd.Timestamp(record["due_date"]) - pd.Timestamp(record["invoice_date"])).days
            )
        except Exception:
            record["days_to_due"] = 0
        predictions.append(record)

    if not predictions:
        return pd.DataFrame()

    analytics = pd.DataFrame(predictions)
    analytics["risk_level"] = analytics["risk_level"].fillna("LOW")
    analytics["predicted_late"] = analytics["predicted_late"].fillna(0).astype(int)
    analytics["amount_at_risk"] = analytics["outstanding_amount"].where(
        analytics["risk_level"].isin(["HIGH", "CRITICAL"]), 0
    )
    analytics["due_soon"] = analytics["days_to_due"].le(7)
    return analytics


def build_dashboard_dataset(artifacts: Dict[str, Any]) -> pd.DataFrame:
    return _cached_build_dashboard_dataset(get_artifact_signature(), artifacts)



# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

def inject_css() -> None:
    st.markdown(
        """
<style>
/* ════════════════════════════════════════════════════
   INVOICEGUARD AI  —  Design System v3
   Palette: Deep Navy + Electric Blue + Semantic colors
   ════════════════════════════════════════════════════ */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* ── Tokens ── */
:root {
  --bg:          #070C18;
  --bg2:         #0A1020;
  --surface:     #0F1729;
  --surface2:    #141E30;
  --card:        #192032;
  --card2:       #1E2840;
  --border:      #1F2D44;
  --border2:     #263552;

  --blue:        #3B82F6;
  --blue-h:      #2563EB;
  --blue-s:      rgba(59,130,246,.14);

  --green:       #10B981;
  --green-s:     rgba(16,185,129,.13);
  --amber:       #F59E0B;
  --amber-s:     rgba(245,158,11,.13);
  --red:         #EF4444;
  --red-s:       rgba(239,68,68,.13);
  --crimson:     #DC2626;
  --crimson-s:   rgba(220,38,38,.15);

  --text:        #F1F5F9;
  --text2:       #CBD5E1;
  --muted:       #64748B;
  --muted2:      #475569;

  --r:           10px;
  --rl:          14px;
  --rxl:         18px;
  --font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
}

/* ── Reset ── */
*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
.stApp {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--font) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu,
footer,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
header[data-testid="stHeader"] { display: none !important; }

/* ── Layout ── */
.main .block-container {
  padding: 1.75rem 2.25rem 3rem !important;
  max-width: 1440px !important;
}

section.main > div { background: var(--bg) !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--bg2) !important;
  border-right: 1px solid var(--border) !important;
  min-width: 230px !important;
}
[data-testid="stSidebar"] > div:first-child {
  background: var(--bg2) !important;
}
[data-testid="stSidebarContent"] {
  background: var(--bg2) !important;
}
[data-testid="stSidebar"] .block-container {
  padding: 1.25rem 0.9rem 1.5rem !important;
}

/* ── Typography ── */
h1, h2, h3, h4, h5, h6 {
  color: var(--text) !important;
  font-family: var(--font) !important;
  letter-spacing: -0.025em !important;
  margin-top: 0 !important;
}
p, li, span { font-family: var(--font) !important; }

/* ── Buttons ── */
.stButton > button {
  font-family: var(--font) !important;
  font-weight: 600 !important;
  border-radius: var(--r) !important;
  transition: all 0.15s ease !important;
  letter-spacing: -0.01em !important;
}

/* Primary CTA */
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
  background: var(--blue) !important;
  color: #fff !important;
  border: none !important;
  padding: 0.65rem 1.5rem !important;
  font-size: 0.9rem !important;
}
.stButton > button[kind="primary"]:hover {
  background: var(--blue-h) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 20px rgba(59,130,246,.35) !important;
}

/* Secondary */
.stButton > button[kind="secondary"],
.stButton > button[data-testid="baseButton-secondary"],
.stButton > button:not([kind]) {
  background: var(--card) !important;
  color: var(--text2) !important;
  border: 1px solid var(--border2) !important;
  padding: 0.55rem 1.2rem !important;
  font-size: 0.87rem !important;
}
.stButton > button[kind="secondary"]:hover,
.stButton > button:not([kind]):hover {
  background: var(--card2) !important;
  border-color: var(--blue) !important;
  color: var(--text) !important;
  transform: none !important;
  box-shadow: none !important;
}

/* Form submit */
[data-testid="stForm"] .stButton > button {
  background: var(--blue) !important;
  color: #fff !important;
  border: none !important;
  padding: 0.65rem 2rem !important;
  font-size: 0.95rem !important;
  font-weight: 700 !important;
  width: 100% !important;
  border-radius: var(--r) !important;
}
[data-testid="stForm"] .stButton > button:hover {
  background: var(--blue-h) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 20px rgba(59,130,246,.3) !important;
}

/* Sidebar nav buttons */
[data-testid="stSidebar"] .stButton > button {
  background: transparent !important;
  border: none !important;
  color: var(--muted) !important;
  font-size: 0.855rem !important;
  font-weight: 500 !important;
  text-align: left !important;
  padding: 0.45rem 0.7rem !important;
  border-radius: 8px !important;
  width: 100% !important;
  justify-content: flex-start !important;
  box-shadow: none !important;
  transform: none !important;
  letter-spacing: 0 !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background: rgba(255,255,255,.06) !important;
  color: var(--text) !important;
  transform: none !important;
  box-shadow: none !important;
}

/* Download button */
.stDownloadButton > button {
  background: var(--surface2) !important;
  color: var(--blue) !important;
  border: 1px solid var(--border2) !important;
  border-radius: var(--r) !important;
  font-weight: 600 !important;
  font-family: var(--font) !important;
}
.stDownloadButton > button:hover {
  background: var(--blue-s) !important;
  border-color: var(--blue) !important;
  transform: none !important;
}

/* ── Form inputs ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea,
.stDateInput > div > div > input {
  background: var(--surface2) !important;
  border: 1px solid var(--border2) !important;
  border-radius: var(--r) !important;
  color: var(--text) !important;
  font-family: var(--font) !important;
  font-size: 0.9rem !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
  border-color: var(--blue) !important;
  box-shadow: 0 0 0 3px rgba(59,130,246,.15) !important;
}
.stSelectbox > div > div {
  background: var(--surface2) !important;
  border: 1px solid var(--border2) !important;
  border-radius: var(--r) !important;
  color: var(--text) !important;
  font-family: var(--font) !important;
}
.stSelectbox > div > div:focus-within {
  border-color: var(--blue) !important;
  box-shadow: 0 0 0 3px rgba(59,130,246,.15) !important;
}

/* Input labels */
.stTextInput label, .stNumberInput label, .stSelectbox label,
.stTextArea label, .stDateInput label, .stCheckbox label,
.stFileUploader label, .stSlider label, .stRadio label {
  color: var(--muted) !important;
  font-size: 0.75rem !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.07em !important;
  font-family: var(--font) !important;
}
/* Checkbox text */
.stCheckbox span { color: var(--text2) !important; font-family: var(--font) !important; }

/* ── Metrics ── */
[data-testid="metric-container"],
[data-testid="stMetric"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--rl) !important;
  padding: 1rem 1.1rem !important;
}
[data-testid="stMetricLabel"] > div {
  color: var(--muted) !important;
  font-size: 0.72rem !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
  font-family: var(--font) !important;
}
[data-testid="stMetricValue"] > div {
  color: var(--text) !important;
  font-weight: 800 !important;
  font-size: 1.5rem !important;
  letter-spacing: -0.03em !important;
  font-family: var(--font) !important;
}
[data-testid="stMetricDelta"] { font-family: var(--font) !important; }

/* ── Tables ── */
[data-testid="stDataFrame"] {
  border-radius: var(--rl) !important;
  border: 1px solid var(--border) !important;
  overflow: hidden !important;
  background: var(--card) !important;
}
[data-testid="stDataFrame"] iframe { border-radius: var(--rl) !important; }

/* ── Tabs ── */
.stTabs [role="tablist"] {
  background: var(--surface) !important;
  border-radius: var(--r) !important;
  padding: 0.2rem !important;
  gap: 0.15rem !important;
  border-bottom: none !important;
}
.stTabs [role="tab"] {
  border-radius: 8px !important;
  color: var(--muted) !important;
  font-weight: 500 !important;
  border: none !important;
  padding: 0.48rem 1rem !important;
  font-size: 0.87rem !important;
  font-family: var(--font) !important;
  background: transparent !important;
}
.stTabs [role="tab"][aria-selected="true"] {
  background: var(--card2) !important;
  color: var(--text) !important;
  font-weight: 700 !important;
}

/* ── Alerts ── */
[data-testid="stNotification"],
.stAlert {
  border-radius: var(--r) !important;
  font-family: var(--font) !important;
  font-size: 0.9rem !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r) !important;
}
[data-testid="stExpander"] summary {
  color: var(--text2) !important;
  font-weight: 600 !important;
  font-family: var(--font) !important;
}

/* ── JSON / Code ── */
[data-testid="stJson"], [data-testid="stCodeBlock"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r) !important;
  font-family: 'Fira Code', 'Cascadia Code', monospace !important;
  font-size: 0.82rem !important;
}

/* ── Slider ── */
[data-testid="stSlider"] > div > div > div {
  background: var(--blue) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
  border: 2px dashed var(--border2) !important;
  border-radius: var(--rl) !important;
  background: var(--surface) !important;
  padding: 1.5rem !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: var(--blue) !important;
  background: var(--blue-s) !important;
}

/* ── Caption / text ── */
.stCaption, [data-testid="stCaptionContainer"] {
  color: var(--muted) !important;
  font-size: 0.8rem !important;
  font-family: var(--font) !important;
}
.stMarkdown p { color: var(--text2) !important; font-family: var(--font) !important; }
.stMarkdown li { color: var(--text2) !important; font-family: var(--font) !important; }
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
  color: var(--text) !important;
}

/* ── Scrollbars ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg2); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 99px; }

/* ══════════════════════════════════════
   COMPONENT LIBRARY
   ══════════════════════════════════════ */

/* Page header */
.ig-ph { margin-bottom: 1.75rem; padding-bottom: 1.25rem; border-bottom: 1px solid var(--border); }
.ig-ph-eyebrow { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: .12em; color: var(--blue); margin-bottom: .4rem; }
.ig-ph-title { font-size: 1.65rem; font-weight: 900; color: var(--text); letter-spacing: -.04em; margin: 0 0 .25rem; line-height: 1.15; }
.ig-ph-sub { font-size: .9rem; color: var(--muted); margin: 0; line-height: 1.5; }

/* KPI card */
.ig-kpi { background: var(--card); border: 1px solid var(--border); border-radius: var(--rl); padding: 1.1rem 1.2rem; height: 100%; }
.ig-kpi-icon { font-size: 1.15rem; margin-bottom: .5rem; }
.ig-kpi-lbl { font-size: .7rem; font-weight: 700; text-transform: uppercase; letter-spacing: .09em; color: var(--muted); margin-bottom: .3rem; }
.ig-kpi-val { font-size: 1.65rem; font-weight: 900; color: var(--text); letter-spacing: -.035em; line-height: 1.05; }
.ig-kpi-sub { font-size: .78rem; color: var(--muted); margin-top: .35rem; }

/* Badge */
.ig-badge { display: inline-flex; align-items: center; gap: .25rem; padding: .22rem .65rem; border-radius: 999px; font-size: .68rem; font-weight: 700; text-transform: uppercase; letter-spacing: .07em; white-space: nowrap; }
.ig-low  { background: var(--green-s);  color: var(--green);  }
.ig-med  { background: var(--amber-s);  color: var(--amber);  }
.ig-high { background: var(--red-s);    color: var(--red);    }
.ig-crit { background: var(--crimson-s);color: var(--crimson);}
.ig-info { background: var(--blue-s);   color: var(--blue);   }
.ig-ok   { background: var(--green-s);  color: var(--green);  }
.ig-fail { background: var(--red-s);    color: var(--red);    }

/* Hero */
.ig-hero { background: linear-gradient(135deg, var(--surface2) 0%, var(--card) 100%); border: 1px solid var(--border2); border-radius: var(--rxl); padding: 2.5rem 2.75rem 2rem; margin-bottom: 1.75rem; position: relative; overflow: hidden; }
.ig-hero::after { content: ''; position: absolute; top: 0; right: 0; width: 45%; height: 100%; background: radial-gradient(ellipse at 80% 20%, rgba(59,130,246,.09) 0%, transparent 55%); pointer-events: none; }
.ig-hero-ey { font-size: .7rem; font-weight: 700; text-transform: uppercase; letter-spacing: .14em; color: var(--blue); margin-bottom: .85rem; }
.ig-hero-t  { font-size: 2.35rem; font-weight: 900; color: var(--text); letter-spacing: -.045em; line-height: 1.1; margin-bottom: .75rem; }
.ig-hero-s  { font-size: 1.05rem; color: var(--muted); max-width: 540px; line-height: 1.65; margin-bottom: 1.75rem; }
.ig-hero-actions { display: flex; gap: .75rem; flex-wrap: wrap; align-items: center; }

/* Section label */
.ig-sec { font-size: .7rem; font-weight: 700; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); margin: 1.5rem 0 .8rem; }

/* Divider */
.ig-hr { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }

/* Card */
.ig-card { background: var(--card); border: 1px solid var(--border); border-radius: var(--rl); padding: 1.25rem; }
.ig-card-title { font-size: .82rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin-bottom: .85rem; }

/* Result card */
.ig-result { border-radius: var(--rxl); padding: 2rem 2.25rem; margin: 1rem 0; border: 1px solid var(--border2); }
.ig-result-low  { background: linear-gradient(135deg, rgba(16,185,129,.08) 0%, var(--surface2) 100%); border-color: rgba(16,185,129,.25); }
.ig-result-med  { background: linear-gradient(135deg, rgba(245,158,11,.08) 0%, var(--surface2) 100%); border-color: rgba(245,158,11,.25); }
.ig-result-high { background: linear-gradient(135deg, rgba(239,68,68,.08) 0%, var(--surface2) 100%);  border-color: rgba(239,68,68,.25);  }
.ig-result-crit { background: linear-gradient(135deg, rgba(220,38,38,.12) 0%, var(--surface2) 100%);  border-color: rgba(220,38,38,.35);  }
.ig-result-risk { font-size: 3rem; font-weight: 900; letter-spacing: -.05em; line-height: 1; margin-bottom: .5rem; }
.ig-result-risk-low  { color: var(--green);  }
.ig-result-risk-med  { color: var(--amber);  }
.ig-result-risk-high { color: var(--red);    }
.ig-result-risk-crit { color: var(--crimson);}
.ig-result-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(155px, 1fr)); gap: .85rem; margin-top: 1.5rem; }
.ig-result-item { background: rgba(0,0,0,.25); border: 1px solid var(--border); border-radius: var(--r); padding: .9rem 1rem; }
.ig-result-item-lbl { font-size: .68rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin-bottom: .35rem; }
.ig-result-item-val { font-size: 1.05rem; font-weight: 700; color: var(--text); }

/* Model card */
.ig-mcard { background: var(--card); border: 1px solid var(--border); border-radius: var(--rl); padding: 1.25rem; height: 100%; }
.ig-mcard-name { font-size: .95rem; font-weight: 700; color: var(--text); margin-bottom: .15rem; }
.ig-mcard-role { font-size: .7rem; font-weight: 600; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin-bottom: 1rem; }
.ig-mcard-row { display: flex; justify-content: space-between; padding: .38rem 0; border-bottom: 1px solid var(--border); font-size: .82rem; }
.ig-mcard-row:last-child { border-bottom: none; }
.ig-mcard-k { color: var(--muted); }
.ig-mcard-v { font-weight: 700; color: var(--text); }

/* Status row */
.ig-srow { display: flex; align-items: center; justify-content: space-between; padding: .75rem 0; border-bottom: 1px solid var(--border); font-size: .87rem; }
.ig-srow:last-child { border-bottom: none; }
.ig-srow-lbl { color: var(--text2); font-weight: 500; }

/* Feature bar */
.ig-fbar { margin-bottom: .75rem; }
.ig-fbar-head { display: flex; justify-content: space-between; font-size: .8rem; margin-bottom: .3rem; }
.ig-fbar-name { color: var(--text2); font-weight: 500; }
.ig-fbar-val  { color: var(--muted); font-family: monospace; }
.ig-fbar-bg   { background: var(--border); border-radius: 999px; height: 6px; overflow: hidden; }
.ig-fbar-fill { height: 100%; border-radius: 999px; }
.ig-fbar-pos  { background: var(--red);   }
.ig-fbar-neg  { background: var(--green); }

/* Sidebar brand */
.ig-brand { padding: .25rem 0 1.1rem; border-bottom: 1px solid var(--border); margin-bottom: .5rem; }
.ig-brand-logo { font-size: 1.05rem; font-weight: 800; color: var(--text); letter-spacing: -.025em; }
.ig-brand-tag  { font-size: .66rem; color: var(--muted); text-transform: uppercase; letter-spacing: .09em; margin-top: .12rem; }
.ig-brand-st   { display: flex; align-items: center; gap: .4rem; font-size: .72rem; color: var(--muted); margin-top: .55rem; }
.ig-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
.ig-dot-on  { background: var(--green);  box-shadow: 0 0 5px var(--green);  }
.ig-dot-off { background: var(--amber);  }

/* Sidebar nav group label */
.ig-ng { font-size: .62rem; font-weight: 700; text-transform: uppercase; letter-spacing: .12em; color: var(--muted2); padding: .6rem .7rem .3rem; }

/* Sidebar perf metric */
.ig-sm { display: flex; justify-content: space-between; font-size: .78rem; padding: .3rem .7rem; color: var(--muted); }
.ig-sm strong { color: var(--text2); font-weight: 700; }

/* Alert row */
.ig-al { background: var(--card); border: 1px solid var(--border); border-left: 3px solid; border-radius: var(--r); padding: .85rem 1rem; margin-bottom: .5rem; }
.ig-al-crit { border-left-color: var(--crimson); }
.ig-al-high { border-left-color: var(--red);    }
.ig-al-med  { border-left-color: var(--amber);  }
.ig-al-info { border-left-color: var(--blue);   }
.ig-al-top  { display: flex; align-items: center; gap: .75rem; margin-bottom: .35rem; }
.ig-al-customer { font-size: .88rem; font-weight: 700; color: var(--text); }
.ig-al-invoice  { font-size: .78rem; color: var(--muted); }
.ig-al-bottom   { display: flex; gap: 1.5rem; font-size: .8rem; color: var(--muted); flex-wrap: wrap; }
.ig-al-bottom span strong { color: var(--text2); font-weight: 600; }

/* Step indicator */
.ig-steps { display: flex; align-items: center; gap: .5rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.ig-step  { display: flex; align-items: center; gap: .4rem; font-size: .8rem; font-weight: 600; color: var(--muted); }
.ig-step.active { color: var(--blue); }
.ig-step-num { width: 22px; height: 22px; border-radius: 50%; font-size: .7rem; font-weight: 700; display: flex; align-items: center; justify-content: center; background: var(--surface2); border: 1px solid var(--border2); color: var(--muted); flex-shrink: 0; }
.ig-step.active .ig-step-num { background: var(--blue-s); border-color: var(--blue); color: var(--blue); }
.ig-step-arrow { color: var(--border2); font-size: .7rem; }

/* Form section */
.ig-form-section { background: var(--surface); border: 1px solid var(--border); border-radius: var(--rl); padding: 1.25rem; margin-bottom: 1rem; }
.ig-form-section-title { font-size: .72rem; font-weight: 700; text-transform: uppercase; letter-spacing: .09em; color: var(--blue); margin-bottom: 1rem; }

/* Empty state */
.ig-empty { text-align: center; padding: 3rem 1rem; color: var(--muted); }
.ig-empty-icon { font-size: 2.5rem; margin-bottom: .75rem; }
.ig-empty-title { font-size: 1rem; font-weight: 700; color: var(--text2); margin-bottom: .4rem; }
.ig-empty-sub { font-size: .85rem; }

/* Customer profile header */
.ig-cph { background: var(--surface2); border: 1px solid var(--border2); border-radius: var(--rxl); padding: 1.75rem 2rem; margin-bottom: 1.5rem; display: flex; align-items: center; gap: 1.5rem; }
.ig-cph-avatar { width: 52px; height: 52px; border-radius: 14px; background: var(--blue-s); display: flex; align-items: center; justify-content: center; font-size: 1.35rem; flex-shrink: 0; }
.ig-cph-name { font-size: 1.25rem; font-weight: 800; color: var(--text); letter-spacing: -.03em; margin-bottom: .2rem; }
.ig-cph-meta { font-size: .8rem; color: var(--muted); }

/* Workflow stage */
.ig-wf { display: flex; align-items: center; gap: 0; margin-bottom: 1.75rem; }
.ig-wf-stage { flex: 1; text-align: center; padding: .75rem .5rem; background: var(--surface); border-top: 1px solid var(--border); border-bottom: 1px solid var(--border); border-left: 1px solid var(--border); font-size: .78rem; font-weight: 600; color: var(--muted); }
.ig-wf-stage:first-child { border-radius: var(--r) 0 0 var(--r); }
.ig-wf-stage:last-child  { border-radius: 0 var(--r) var(--r) 0; border-right: 1px solid var(--border); }
.ig-wf-stage.done { background: var(--green-s); color: var(--green); border-color: rgba(16,185,129,.25); }
.ig-wf-stage.active { background: var(--blue-s); color: var(--blue); border-color: rgba(59,130,246,.3); font-weight: 700; }

/* NLP workspace */
.ig-nlp-header { background: linear-gradient(135deg, var(--surface2) 0%, rgba(59,130,246,.06) 100%); border: 1px solid var(--border2); border-radius: var(--rxl); padding: 1.75rem 2rem; margin-bottom: 1.5rem; }
.ig-nlp-icon { font-size: 1.75rem; margin-bottom: .6rem; }
.ig-nlp-title { font-size: 1.4rem; font-weight: 800; color: var(--text); letter-spacing: -.035em; margin-bottom: .35rem; }
.ig-nlp-sub { font-size: .9rem; color: var(--muted); }

/* NLP result */
.ig-nlp-result { border-radius: var(--rxl); padding: 1.75rem; margin: 1rem 0; border: 1px solid var(--border2); }
.ig-nlp-result-high { background: linear-gradient(135deg, rgba(239,68,68,.1) 0%, var(--surface2) 100%); border-color: rgba(239,68,68,.3); }
.ig-nlp-result-med  { background: linear-gradient(135deg, rgba(245,158,11,.08) 0%, var(--surface2) 100%); border-color: rgba(245,158,11,.25); }
.ig-nlp-result-low  { background: linear-gradient(135deg, rgba(16,185,129,.08) 0%, var(--surface2) 100%); border-color: rgba(16,185,129,.25); }
.ig-nlp-label { font-size: .7rem; font-weight: 700; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); margin-bottom: .35rem; }
.ig-nlp-risk-text { font-size: 2rem; font-weight: 900; letter-spacing: -.04em; margin-bottom: .6rem; }
.ig-nlp-risk-high { color: var(--red);   }
.ig-nlp-risk-med  { color: var(--amber); }
.ig-nlp-risk-low  { color: var(--green); }

/* Financial big number */
.ig-fin-hero { text-align: center; padding: 1.5rem; }
.ig-fin-label { font-size: .72rem; font-weight: 700; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); margin-bottom: .5rem; }
.ig-fin-value { font-size: 2.5rem; font-weight: 900; color: var(--text); letter-spacing: -.05em; line-height: 1; }
.ig-fin-sub   { font-size: .82rem; color: var(--muted); margin-top: .4rem; }

</style>
""",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_RISK_CSS: Dict[str, str] = {
    "LOW": "ig-low",
    "MEDIUM": "ig-med",
    "HIGH": "ig-high",
    "CRITICAL": "ig-crit",
    "HIGH RISK": "ig-high",
    "MEDIUM RISK": "ig-med",
    "LOW RISK": "ig-low",
    "PASS": "ig-ok",
    "FAIL": "ig-fail",
    "ACTIVE": "ig-ok",
}

_RISK_RESULT_CSS: Dict[str, str] = {
    "LOW": "ig-result-low",
    "MEDIUM": "ig-result-med",
    "HIGH": "ig-result-high",
    "CRITICAL": "ig-result-crit",
}

_RISK_TEXT_CSS: Dict[str, str] = {
    "LOW": "ig-result-risk-low",
    "MEDIUM": "ig-result-risk-med",
    "HIGH": "ig-result-risk-high",
    "CRITICAL": "ig-result-risk-crit",
}


def badge(label: str) -> str:
    css = _RISK_CSS.get(label.upper(), "ig-info")
    return f'<span class="ig-badge {css}">{label}</span>'


def page_header(title: str, subtitle: str = "", eyebrow: str = "") -> None:
    ey_html = f'<div class="ig-ph-eyebrow">{eyebrow}</div>' if eyebrow else ""
    sub_html = f'<p class="ig-ph-sub">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<div class="ig-ph">{ey_html}<h1 class="ig-ph-title">{title}</h1>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def section_label(text: str) -> None:
    st.markdown(f'<div class="ig-sec">{text}</div>', unsafe_allow_html=True)


def divider() -> None:
    st.markdown('<hr class="ig-hr">', unsafe_allow_html=True)


def kpi_row(items: List[Dict[str, Any]]) -> None:
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            icon = item.get("icon", "")
            sub = item.get("sub", "")
            icon_html = f'<div class="ig-kpi-icon">{icon}</div>' if icon else ""
            sub_html = f'<div class="ig-kpi-sub">{sub}</div>' if sub else ""
            st.markdown(
                f"""<div class="ig-kpi">
                    {icon_html}
                    <div class="ig-kpi-lbl">{item['label']}</div>
                    <div class="ig-kpi-val">{item['value']}</div>
                    {sub_html}
                </div>""",
                unsafe_allow_html=True,
            )


def render_feature_bars(explanation: List[Dict[str, Any]]) -> None:
    if not explanation:
        return
    max_impact = max(e["impact"] for e in explanation) or 1.0
    for e in explanation:
        pct = min(int(e["impact"] / max_impact * 100), 100)
        is_risk = "increases risk" in e.get("direction", "")
        bar_cls = "ig-fbar-pos" if is_risk else "ig-fbar-neg"
        dir_badge = badge("HIGH RISK" if is_risk else "LOW RISK")
        st.markdown(
            f"""<div class="ig-fbar">
              <div class="ig-fbar-head">
                <span class="ig-fbar-name">{e['feature']}</span>
                <span class="ig-fbar-val">{e['impact']:.4f} &nbsp;{dir_badge}</span>
              </div>
              <div class="ig-fbar-bg"><div class="ig-fbar-fill {bar_cls}" style="width:{pct}%"></div></div>
            </div>""",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────

_NAV_GROUPS = [
    ("PLATFORM",      ["Overview", "Invoice Risk", "Customer 360", "Customer Risk"]),
    ("INTELLIGENCE",  ["Revenue Intelligence", "Retention", "Customer Segments", "Product Intelligence", "NLP Intelligence", "Explainable AI"]),
    ("OPERATIONS",    ["Risk Queue"]),
    ("SYSTEM",        ["Model Center", "Data Quality", "API / Integration", "Documentation"]),
]


def render_sidebar(artifacts: Dict[str, Any]) -> str:
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Overview"

    models = artifacts.get("models", {})
    load_errors = artifacts.get("load_errors", {})
    metadata = artifacts.get("metadata", {})
    model_count = len(models)
    healthy = not load_errors

    with st.sidebar:
        st.markdown(
            f"""<div class="ig-brand">
              <div class="ig-brand-logo">💳 InvoiceGuard AI</div>
              <div class="ig-brand-tag">Payment Risk Intelligence</div>
              <div class="ig-brand-st">
                <span class="ig-dot {'ig-dot-on' if healthy else 'ig-dot-off'}"></span>
                {model_count} models active
              </div>
            </div>""",
            unsafe_allow_html=True,
        )

        for group_name, pages in _NAV_GROUPS:
            st.markdown(f'<div class="ig-ng">{group_name}</div>', unsafe_allow_html=True)
            for page in pages:
                if st.button(page, key=f"_nav_{page}", use_container_width=True):
                    st.session_state.current_page = page
                    st.rerun()

        # Performance snapshot at the bottom
        test_metrics = metadata.get("test_metrics", {})
        if test_metrics:
            divider()
            st.markdown('<div class="ig-ng">MODEL PERFORMANCE</div>', unsafe_allow_html=True)
            acc = test_metrics.get("accuracy", 0)
            auc = test_metrics.get("roc_auc", 0)
            f1 = test_metrics.get("f1", 0)
            st.markdown(
                f"""<div class="ig-sm"><span>Accuracy</span><strong>{acc:.1%}</strong></div>
                    <div class="ig-sm"><span>ROC-AUC</span><strong>{auc:.3f}</strong></div>
                    <div class="ig-sm"><span>F1 Score</span><strong>{f1:.3f}</strong></div>""",
                unsafe_allow_html=True,
            )

    return str(st.session_state.current_page)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: EXECUTIVE DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def render_exec_dashboard(artifacts: Dict[str, Any]) -> None:
    metadata = artifacts.get("metadata", {})
    threshold = float(metadata.get("threshold", 0.5))

    # Hero
    st.markdown(
        """<div class="ig-hero">
          <div class="ig-hero-ey">AI-Powered Payment Risk Intelligence</div>
          <div class="ig-hero-t">Know which payments are at risk<br>before they become losses.</div>
          <div class="ig-hero-s">Predict late payments, quantify financial exposure, analyze customer communication, and act on AI-generated recommendations — before invoices become overdue.</div>
        </div>""",
        unsafe_allow_html=True,
    )

    analytics = build_dashboard_dataset(artifacts)

    if analytics.empty:
        st.markdown(
            """<div class="ig-empty">
              <div class="ig-empty-icon">📊</div>
              <div class="ig-empty-title">No portfolio data available</div>
              <div class="ig-empty-sub">The demo invoice dataset could not be loaded. Navigate to Single Invoice Prediction to run individual assessments.</div>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    # KPIs
    total_receivables = float(analytics["outstanding_amount"].sum())
    at_risk = float(analytics.loc[analytics["risk_level"].isin(["HIGH", "CRITICAL"]), "outstanding_amount"].sum())
    predicted_late = int((analytics["late_probability"] >= threshold).sum())
    high_risk_customers = int(analytics.loc[analytics["risk_level"].isin(["HIGH", "CRITICAL"]), "customer"].nunique())
    exposure = float(analytics["estimated_financial_exposure"].sum())
    total_invoices = len(analytics)

    section_label("Portfolio Overview")
    kpi_row([
        {"icon": "📄", "label": "Total Invoices", "value": f"{total_invoices:,}", "sub": "Active portfolio"},
        {"icon": "💰", "label": "Total Receivables", "value": f"${total_receivables:,.0f}", "sub": "Outstanding"},
        {"icon": "⚠️", "label": "Revenue at Risk", "value": f"${at_risk:,.0f}", "sub": "High + Critical"},
        {"icon": "🔴", "label": "Predicted Late", "value": f"{predicted_late}", "sub": f"Threshold {threshold:.2f}"},
        {"icon": "👥", "label": "High-Risk Customers", "value": f"{high_risk_customers}", "sub": "Requiring attention"},
        {"icon": "📉", "label": "Total Exposure", "value": f"${exposure:,.0f}", "sub": "Projected impact"},
    ])

    divider()

    # Charts
    col1, col2 = st.columns(2)
    with col1:
        section_label("Risk Distribution")
        risk_dist = analytics["risk_level"].value_counts().reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"], fill_value=0)
        st.bar_chart(risk_dist, color="#3B82F6")

    with col2:
        section_label("Financial Exposure by Month")
        if "invoice_date" in analytics.columns:
            exp_by_month = (
                analytics.assign(month=analytics["invoice_date"].dt.to_period("M").astype(str))
                .groupby("month")["estimated_financial_exposure"]
                .sum()
                .sort_index()
            )
            st.area_chart(exp_by_month, color="#EF4444")

    col3, col4 = st.columns(2)
    with col3:
        section_label("Late-Payment Trend")
        if "invoice_date" in analytics.columns:
            trend = (
                analytics.assign(month=analytics["invoice_date"].dt.to_period("M").astype(str))
                .groupby("month")["predicted_late"]
                .sum()
                .sort_index()
            )
            st.line_chart(trend, color="#F59E0B")

    with col4:
        section_label("Top Risk Customers")
        customer_risk = (
            analytics.groupby("customer")["late_probability"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )
        st.bar_chart(customer_risk, color="#EF4444")

    divider()

    # Industry breakdown
    if "industry" in analytics.columns:
        section_label("Risk by Industry")
        industry_risk = analytics.groupby("industry")["late_probability"].mean().sort_values(ascending=False)
        st.bar_chart(industry_risk, color="#3B82F6")
        divider()

    # Priority table
    section_label("Priority Attention — Highest Risk Invoices")
    display_cols = [c for c in ["invoice_id", "customer", "invoice_date", "due_date", "invoice_amount", "outstanding_amount", "risk_level", "late_probability", "estimated_financial_exposure"] if c in analytics.columns]
    top_invoices = analytics[display_cols].sort_values("late_probability", ascending=False).head(20)
    st.dataframe(top_invoices, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: SINGLE INVOICE PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def render_single_prediction(artifacts: Dict[str, Any], cold_start: bool = False) -> None:
    metadata = artifacts["metadata"]
    model_options = ["classifier"] + [
        n for n in ["logistic_regression", "random_forest", "mlp_neural_network"]
        if n in artifacts["models"]
    ]
    if "classifier" not in model_options:
        model_options = [n for n in artifacts["models"] if n != "delay_regressor"]

    demo_options = load_demo_options()

    if cold_start:
        page_header(
            "Cold Start Assessment",
            "AI risk scoring for a first-time customer with no payment history.",
            eyebrow="New Customer Mode",
        )
        st.info("ℹ️ Cold Start: All prior-history features are set to zero. The model uses only invoice and profile attributes.", icon=None)
    else:
        page_header(
            "Invoice Risk Assessment",
            "Run a complete AI risk assessment for a single invoice using the trained prediction pipeline.",
            eyebrow="Single Invoice",
        )

    # Step indicator
    st.markdown(
        """<div class="ig-steps">
          <div class="ig-step active">
            <div class="ig-step-num">1</div><span>Invoice Details</span>
          </div>
          <span class="ig-step-arrow">›</span>
          <div class="ig-step active">
            <div class="ig-step-num">2</div><span>Customer Profile</span>
          </div>
          <span class="ig-step-arrow">›</span>
          <div class="ig-step">
            <div class="ig-step-num">3</div><span>AI Assessment</span>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    with st.form("invoice_assessment_form"):
        # Section 1: Invoice
        st.markdown('<div class="ig-form-section">', unsafe_allow_html=True)
        st.markdown('<div class="ig-form-section-title">📄 Invoice Information</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            invoice_date = st.date_input("Invoice Date", value=pd.Timestamp("2026-09-01").date())
        with c2:
            due_date = st.date_input("Due Date", value=pd.Timestamp("2026-10-01").date())
        with c3:
            invoice_amount = st.number_input("Invoice Amount ($)", min_value=1.0, value=50000.0, step=1000.0)
        with c4:
            outstanding_amount = st.number_input("Outstanding Amount ($)", min_value=0.0, value=80000.0, step=1000.0)
        st.markdown("</div>", unsafe_allow_html=True)

        # Section 2: Customer History
        st.markdown('<div class="ig-form-section">', unsafe_allow_html=True)
        st.markdown('<div class="ig-form-section-title">📋 Customer Payment History</div>', unsafe_allow_html=True)
        if cold_start:
            customer_seen_before = 0
            prior_late_count = 0.0
            prior_late_ratio = 0.0
            prior_avg_delay = 0.0
            st.caption("Cold start: history fields are zeroed — this customer has no prior invoices.")
        else:
            existing_customer = st.checkbox("Existing customer (has prior invoice history)", value=True)
            h1, h2, h3, h4 = st.columns(4)
            with h1:
                customer_seen_before = st.number_input("Prior Invoices", min_value=0, value=6, step=1)
            with h2:
                prior_late_count = st.number_input("Late Payment Count", min_value=0.0, value=3.0, step=1.0)
            with h3:
                prior_late_ratio = st.number_input("Late Ratio (0–1)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
            with h4:
                prior_avg_delay = st.number_input("Avg Delay (days)", min_value=0.0, value=9.0, step=0.5)
            if not existing_customer:
                customer_seen_before = 0
                prior_late_count = 0.0
                prior_late_ratio = 0.0
                prior_avg_delay = 0.0
        st.markdown("</div>", unsafe_allow_html=True)

        # Section 3: Customer Profile
        st.markdown('<div class="ig-form-section">', unsafe_allow_html=True)
        st.markdown('<div class="ig-form-section-title">🏢 Customer Profile</div>', unsafe_allow_html=True)
        p1, p2, p3, p4, p5 = st.columns(5)
        with p1:
            industry = st.selectbox("Industry", options=demo_options.get("industry", DEFAULT_CATEGORICALS["industry"]))
        with p2:
            company_size = st.selectbox("Company Size", options=demo_options.get("company_size", DEFAULT_CATEGORICALS["company_size"]), index=1)
        with p3:
            payment_method = st.selectbox("Payment Method", options=demo_options.get("payment_method", DEFAULT_CATEGORICALS["payment_method"]))
        with p4:
            customer_segment = st.selectbox("Segment", options=demo_options.get("customer_segment", DEFAULT_CATEGORICALS["customer_segment"]))
        with p5:
            model_choice = st.selectbox("AI Model", options=model_options)
        st.markdown("</div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("🔍  Run AI Assessment", type="primary")

    if not submitted:
        return

    payload = {
        "invoice_date": str(invoice_date),
        "due_date": str(due_date),
        "invoice_amount": invoice_amount,
        "customer_seen_before": customer_seen_before if not cold_start else 0,
        "prior_late_count": prior_late_count if not cold_start else 0.0,
        "prior_late_ratio": prior_late_ratio if not cold_start else 0.0,
        "prior_avg_delay": prior_avg_delay if not cold_start else 0.0,
        "industry": industry,
        "company_size": company_size,
        "payment_method": payment_method,
        "customer_segment": customer_segment,
        "outstanding_amount": outstanding_amount,
    }

    errors = validate_single_input(payload)
    if errors:
        for msg in errors:
            st.error(msg)
        return

    with st.spinner("Running AI assessment…"):
        try:
            result = run_prediction(payload, model_choice, artifacts)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

    risk = result["risk_level"]
    risk_lower = risk.lower()
    result_cls = _RISK_RESULT_CSS.get(risk, "ig-result-low")
    text_cls = _RISK_TEXT_CSS.get(risk, "ig-result-risk-low")

    # Step 3 header
    st.markdown(
        """<div class="ig-steps">
          <div class="ig-step">
            <div class="ig-step-num">1</div><span>Invoice Details</span>
          </div>
          <span class="ig-step-arrow">›</span>
          <div class="ig-step">
            <div class="ig-step-num">2</div><span>Customer Profile</span>
          </div>
          <span class="ig-step-arrow">›</span>
          <div class="ig-step active">
            <div class="ig-step-num">3</div><span>AI Assessment</span>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""<div class="ig-result {result_cls}">
          <div style="margin-bottom:.4rem;">
            {badge(risk)}
          </div>
          <div class="ig-result-risk {text_cls}">{risk} RISK</div>
          <div class="ig-result-grid">
            <div class="ig-result-item">
              <div class="ig-result-item-lbl">Late Probability</div>
              <div class="ig-result-item-val">{result['late_probability']:.1%}</div>
            </div>
            <div class="ig-result-item">
              <div class="ig-result-item-lbl">Expected Delay</div>
              <div class="ig-result-item-val">{result['expected_delay_days']:.1f} days</div>
            </div>
            <div class="ig-result-item">
              <div class="ig-result-item-lbl">Revenue at Risk</div>
              <div class="ig-result-item-val">${result['estimated_financial_exposure']:,.2f}</div>
            </div>
            <div class="ig-result-item">
              <div class="ig-result-item-lbl">Model Used</div>
              <div class="ig-result-item-val">{model_choice.replace('_', ' ').title()}</div>
            </div>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Recommended action
    action_color = "#EF4444" if risk in ["HIGH", "CRITICAL"] else "#10B981"
    st.markdown(
        f"""<div class="ig-card" style="border-left: 3px solid {action_color}; margin-bottom: 1rem;">
          <div class="ig-card-title">Recommended Action</div>
          <p style="color:var(--text2); margin:0; font-size:.95rem;">{result['recommendation']}</p>
        </div>""",
        unsafe_allow_html=True,
    )

    # Explainability
    section_label("AI Explanation — Feature Drivers")
    explanation = explain_prediction(artifacts["models"].get(model_choice), result["feature_frame"], metadata)
    if explanation:
        render_feature_bars(explanation)
    else:
        st.caption("SHAP explainability is not available for this model. Feature importances or coefficient magnitudes are used where possible.")

    # Artifact metrics
    if "test_metrics.json" in artifacts.get("reports", {}):
        with st.expander("📊 Model Performance Metrics", expanded=False):
            st.json(artifacts["reports"]["test_metrics.json"])


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: RISK CENTER
# ─────────────────────────────────────────────────────────────────────────────

def render_risk_center(artifacts: Dict[str, Any]) -> None:
    page_header(
        "Risk Center",
        "Monitor, filter, and prioritize invoices by payment risk across your entire portfolio.",
        eyebrow="Command Center",
    )
    analytics = build_dashboard_dataset(artifacts)
    if analytics.empty:
        st.markdown('<div class="ig-empty"><div class="ig-empty-icon">🎯</div><div class="ig-empty-title">No portfolio data</div><div class="ig-empty-sub">Demo dataset unavailable. Use Single Invoice Prediction for individual assessments.</div></div>', unsafe_allow_html=True)
        return

    # Summary badges
    for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        count = int((analytics["risk_level"] == level).sum())
        b = badge(level)
        if count > 0:
            st.markdown(f'{b} &nbsp;<strong style="color:var(--text)">{count}</strong> &nbsp;<span style="color:var(--muted);font-size:.8rem;">invoices</span>', unsafe_allow_html=True)

    divider()
    section_label("Filters")

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        risk_filter = st.selectbox("Risk Level", ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"])
    with f2:
        customer_filter = st.selectbox("Customer", ["All", *sorted(analytics["customer"].dropna().unique())])
    with f3:
        industry_vals = sorted(analytics["industry"].dropna().unique()) if "industry" in analytics.columns else []
        industry_filter = st.selectbox("Industry", ["All", *industry_vals]) if industry_vals else None
    with f4:
        amount_max = max(int(analytics["invoice_amount"].quantile(0.95)), 1)
        amount_cap = st.slider("Max Invoice Amount", 0, amount_max, amount_max)

    filtered = analytics.copy()
    if risk_filter != "All":
        filtered = filtered[filtered["risk_level"] == risk_filter]
    if customer_filter != "All":
        filtered = filtered[filtered["customer"] == customer_filter]
    if industry_filter and industry_filter != "All" and "industry" in filtered.columns:
        filtered = filtered[filtered["industry"] == industry_filter]
    filtered = filtered[filtered["invoice_amount"] <= amount_cap]

    st.caption(f"Showing {len(filtered):,} of {len(analytics):,} invoices")

    display_cols = [c for c in ["invoice_id", "customer", "industry", "invoice_date", "due_date", "invoice_amount", "late_probability", "expected_delay_days", "risk_level", "estimated_financial_exposure", "recommendation"] if c in filtered.columns]
    st.dataframe(filtered[display_cols].sort_values("late_probability", ascending=False), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: CUSTOMER 360
# ─────────────────────────────────────────────────────────────────────────────

def render_customer_360(artifacts: Dict[str, Any]) -> None:
    page_header(
        "Customer 360",
        "Complete customer intelligence — payment behavior, risk trend, and invoice history.",
        eyebrow="Customer Intelligence",
    )
    analytics = build_dashboard_dataset(artifacts)
    if analytics.empty:
        st.markdown('<div class="ig-empty"><div class="ig-empty-icon">👤</div><div class="ig-empty-title">No customer data</div><div class="ig-empty-sub">Demo dataset unavailable.</div></div>', unsafe_allow_html=True)
        return

    customer_option = st.selectbox("Select Customer", sorted(analytics["customer"].dropna().unique()))
    customer_df = analytics[analytics["customer"] == customer_option].sort_values("invoice_date")

    if customer_df.empty:
        st.warning("No invoices found for the selected customer.")
        return

    total_invoices = len(customer_df)
    on_time_rate = float((customer_df["predicted_late"] == 0).mean())
    late_rate = 1 - on_time_rate
    avg_delay = float(customer_df["expected_delay_days"].mean())
    outstanding = float(customer_df["outstanding_amount"].sum())
    current_risk = customer_df.iloc[-1]["risk_level"] if not customer_df.empty else "LOW"
    avg_prob = float(customer_df["late_probability"].mean())

    # Fetch comprehensive profile
    from src.customer_360 import get_customer_profile
    profile = get_customer_profile(customer_option)
    retail = profile.get("retail_intelligence", {})
    
    # Customer profile header
    risk_badge_html = badge(current_risk)
    st.markdown(
        f"""<div class="ig-cph">
          <div class="ig-cph-avatar">🏢</div>
          <div>
            <div class="ig-cph-name">{customer_option}</div>
            <div class="ig-cph-meta">Current risk: {risk_badge_html} &nbsp;·&nbsp; {total_invoices} invoices &nbsp;·&nbsp; ${outstanding:,.0f} outstanding</div>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # KPIs (InvoiceGuard Risk + Retail Intelligence if available)
    kpis = [
        {"icon": "📄", "label": "Total Invoices",   "value": f"{total_invoices}"},
        {"icon": "⚠️", "label": "Late-Payment Rate", "value": f"{late_rate:.0%}"},
        {"icon": "💰", "label": "Outstanding",        "value": f"${outstanding:,.0f}"},
        {"icon": "🎯", "label": "Avg Risk Prob",      "value": f"{avg_prob:.1%}"},
    ]
    if retail:
        kpis.append({"icon": "⭐", "label": "Retention Risk", "value": retail.get("predictions", {}).get("retention_risk", "N/A")})
        kpis.append({"icon": "💵", "label": "Est Future Rev", "value": f"${retail.get('predictions', {}).get('expected_future_revenue_60d', 0):,.0f}"})
    
    kpi_row(kpis)

    divider()
    
    if retail:
        st.markdown("### 🛍️ Retail Intelligence (Customer 360)")
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        rfm = retail.get("rfm", {})
        col_r1.metric("Recency (Days)", rfm.get("recency_days", 0))
        col_r2.metric("Frequency (Orders)", rfm.get("frequency", 0))
        col_r3.metric("Monetary Value", f"${rfm.get('monetary', 0):,.2f}")
        col_r4.metric("Average Order Value", f"${rfm.get('avg_order_value', 0):,.2f}")
        
        recs = retail.get("recommendations", [])
        if recs:
            for r in recs:
                st.info(f"💡 Recommendation: {r}")
        divider()

    col_a, col_b = st.columns([2, 1])
    with col_a:
        section_label("Payment Risk Trend")
        if "invoice_date" in customer_df.columns:
            trend_df = customer_df.set_index("invoice_date")["late_probability"].sort_index()
            st.line_chart(trend_df, color="#3B82F6")

    with col_b:
        section_label("Risk Distribution")
        risk_dist = customer_df["risk_level"].value_counts()
        st.bar_chart(risk_dist, color="#EF4444")

    divider()
    section_label("Invoice History")
    disp_cols = [c for c in ["invoice_id", "invoice_date", "due_date", "invoice_amount", "outstanding_amount", "late_probability", "expected_delay_days", "risk_level", "estimated_financial_exposure"] if c in customer_df.columns]
    st.dataframe(customer_df[disp_cols], use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: FINANCIAL IMPACT
# ─────────────────────────────────────────────────────────────────────────────

def render_financial_impact(artifacts: Dict[str, Any]) -> None:
    page_header(
        "Financial Impact",
        "Quantify revenue at risk, financial exposure, and portfolio-level payment risk concentration.",
        eyebrow="Financial Intelligence",
    )
    analytics = build_dashboard_dataset(artifacts)
    if analytics.empty:
        st.markdown('<div class="ig-empty"><div class="ig-empty-icon">💰</div><div class="ig-empty-title">No financial data</div><div class="ig-empty-sub">Demo dataset unavailable.</div></div>', unsafe_allow_html=True)
        return

    total_outstanding = float(analytics["outstanding_amount"].sum())
    amount_at_risk = float(analytics.loc[analytics["risk_level"].isin(["HIGH", "CRITICAL"]), "outstanding_amount"].sum())
    exposure = float(analytics["estimated_financial_exposure"].sum())
    high_risk_exposure = float(analytics.loc[analytics["risk_level"].isin(["HIGH", "CRITICAL"]), "estimated_financial_exposure"].sum())
    risk_ratio = amount_at_risk / total_outstanding if total_outstanding else 0

    # Big financial numbers
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="ig-card"><div class="ig-fin-hero"><div class="ig-fin-label">Total Outstanding</div><div class="ig-fin-value">${total_outstanding:,.0f}</div><div class="ig-fin-sub">Across all invoices</div></div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="ig-card"><div class="ig-fin-hero"><div class="ig-fin-label">Revenue at Risk</div><div class="ig-fin-value" style="color:var(--red)">${amount_at_risk:,.0f}</div><div class="ig-fin-sub">High + Critical tier</div></div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="ig-card"><div class="ig-fin-hero"><div class="ig-fin-label">Total AI Exposure</div><div class="ig-fin-value" style="color:var(--amber)">${exposure:,.0f}</div><div class="ig-fin-sub">Probability-weighted</div></div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="ig-card"><div class="ig-fin-hero"><div class="ig-fin-label">Risk Concentration</div><div class="ig-fin-value" style="color:var(--red)">{risk_ratio:.1%}</div><div class="ig-fin-sub">Of portfolio at High/Critical</div></div></div>', unsafe_allow_html=True)

    divider()

    c1, c2 = st.columns(2)
    with c1:
        section_label("Exposure by Risk Level")
        exposure_by_level = analytics.groupby("risk_level")["estimated_financial_exposure"].sum().reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"], fill_value=0)
        st.bar_chart(exposure_by_level, color="#EF4444")

    with c2:
        if "industry" in analytics.columns:
            section_label("Exposure by Industry")
            exp_by_industry = analytics.groupby("industry")["estimated_financial_exposure"].sum().sort_values(ascending=False)
            st.bar_chart(exp_by_industry, color="#F59E0B")

    divider()
    section_label("Top Invoices by Financial Exposure")
    top_df = analytics.nlargest(20, "estimated_financial_exposure")
    disp = [c for c in ["invoice_id", "customer", "invoice_amount", "outstanding_amount", "estimated_financial_exposure", "risk_level", "late_probability"] if c in top_df.columns]
    st.dataframe(top_df[disp], use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: AI EXPLANATION
# ─────────────────────────────────────────────────────────────────────────────

def render_ai_explanation(artifacts: Dict[str, Any]) -> None:
    page_header(
        "AI Explanation",
        "Understand which factors drive each payment-risk prediction using SHAP explainability.",
        eyebrow="Explainable AI",
    )
    analytics = build_dashboard_dataset(artifacts)
    if analytics.empty:
        st.markdown('<div class="ig-empty"><div class="ig-empty-icon">🧠</div><div class="ig-empty-title">No prediction data</div><div class="ig-empty-sub">Demo dataset unavailable.</div></div>', unsafe_allow_html=True)
        return

    model_options = [n for n in ["classifier", "logistic_regression", "random_forest", "mlp_neural_network"] if n in artifacts["models"]]
    if not model_options:
        st.warning("No compatible classifier artifacts are available for explanation.")
        return

    col_a, col_b = st.columns([2, 1])
    with col_a:
        invoice_options = analytics["invoice_id"].tolist()
        selected_invoice = st.selectbox("Select Invoice", invoice_options)
    with col_b:
        selected_model = st.selectbox("Model", model_options)

    selected_row = analytics[analytics["invoice_id"] == selected_invoice].iloc[0]
    result = run_prediction(selected_row.to_dict(), selected_model, artifacts)
    risk = result["risk_level"]

    # Risk summary
    result_cls = _RISK_RESULT_CSS.get(risk, "ig-result-low")
    text_cls = _RISK_TEXT_CSS.get(risk, "ig-result-risk-low")
    st.markdown(
        f"""<div class="ig-result {result_cls}" style="padding:1.5rem;">
          <div style="display:flex; align-items:center; gap:1rem; flex-wrap:wrap;">
            <div>
              <div class="ig-result-risk {text_cls}" style="font-size:2rem;">{risk}</div>
              <div style="color:var(--muted);font-size:.82rem;">Invoice: {selected_invoice} &nbsp;·&nbsp; Model: {selected_model.replace('_',' ').title()}</div>
            </div>
            <div style="display:flex; gap:.75rem; flex-wrap:wrap;">
              <div class="ig-result-item"><div class="ig-result-item-lbl">Probability</div><div class="ig-result-item-val">{result['late_probability']:.1%}</div></div>
              <div class="ig-result-item"><div class="ig-result-item-lbl">Delay</div><div class="ig-result-item-val">{result['expected_delay_days']:.1f} days</div></div>
              <div class="ig-result-item"><div class="ig-result-item-lbl">Exposure</div><div class="ig-result-item-val">${result['estimated_financial_exposure']:,.0f}</div></div>
            </div>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    divider()
    section_label("Top Risk Drivers — Feature Attribution")

    explanation = explain_prediction(artifacts["models"].get(selected_model), result["feature_frame"], artifacts["metadata"])
    if explanation:
        st.markdown(
            '<div class="ig-card"><div class="ig-card-title">Why did the AI make this prediction?</div>',
            unsafe_allow_html=True,
        )
        render_feature_bars(explanation)
        st.markdown("</div>", unsafe_allow_html=True)

        divider()
        # Table view
        section_label("Full Attribution Table")
        st.dataframe(pd.DataFrame(explanation), use_container_width=True, hide_index=True)
    else:
        st.info("SHAP explainability is not available for the selected model. Feature importances are shown where possible.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: NLP PAYMENT-RISK INTELLIGENCE
# ─────────────────────────────────────────────────────────────────────────────

def render_nlp_section(artifacts: Dict[str, Any]) -> None:
    page_header(
        "Message Intelligence",
        "Analyze customer communication for payment-risk signals using the trained NLP classifier.",
        eyebrow="NLP Payment-Risk Intelligence",
    )

    nlp_model = artifacts["models"].get("nlp_payment_risk")
    if nlp_model is None:
        st.markdown('<div class="ig-empty"><div class="ig-empty-icon">🗣️</div><div class="ig-empty-title">NLP model unavailable</div><div class="ig-empty-sub">The NLP payment-risk model could not be loaded in this deployment.</div></div>', unsafe_allow_html=True)
        return

    # NLP workspace header
    st.markdown(
        """<div class="ig-nlp-header">
          <div class="ig-nlp-icon">🗣️</div>
          <div class="ig-nlp-title">Customer Payment Message Analyzer</div>
          <div class="ig-nlp-sub">Paste any customer message related to payment. The AI will classify it as HIGH, MEDIUM, or LOW risk based on language patterns.</div>
        </div>""",
        unsafe_allow_html=True,
    )

    thresholds = load_nlp_thresholds()
    text_examples = load_text_examples()

    col_a, col_b = st.columns([1, 2])
    with col_a:
        sample_text = st.selectbox("Load example message", options=text_examples)
    with col_b:
        st.caption(f"Decision thresholds — HIGH ≥ {thresholds['high']:.2f} · MEDIUM ≥ {thresholds['medium']:.2f}")

    customer_message = st.text_area(
        "Customer payment-related message",
        value=sample_text,
        height=140,
        placeholder="Paste a customer message here…",
    )

    analyze_clicked = st.button("🔍  Analyze Message", type="primary")

    if not analyze_clicked or not customer_message.strip():
        st.caption("Enter a customer message and click Analyze to run the NLP assessment.")
        return

    try:
        analysis = score_nlp_message(nlp_model, customer_message, thresholds)
    except Exception as exc:
        st.error(f"Unable to analyze message: {exc}")
        return

    risk_label = analysis["risk_label"]
    prob = analysis["probability"]
    risk_key = risk_label.split()[0].lower()  # "high", "medium", "low"
    result_cls = f"ig-nlp-result-{risk_key}"
    text_cls = f"ig-nlp-risk-{risk_key}"

    st.markdown(
        f"""<div class="ig-nlp-result {result_cls}">
          <div class="ig-nlp-label">NLP Risk Assessment</div>
          <div class="ig-nlp-risk-text {text_cls}">{risk_label}</div>
          <div style="display:flex; gap:1.5rem; flex-wrap:wrap; margin-top:.5rem;">
            <div><div class="ig-nlp-label">Confidence</div><strong style="font-size:1.15rem;color:var(--text)">{prob:.1%}</strong></div>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Probability breakdown
    divider()
    c1, c2, c3 = st.columns(3)
    probs = analysis["probabilities"]
    with c1:
        st.metric("HIGH RISK", f"{probs.get('HIGH_RISK', 0):.1%}")
    with c2:
        st.metric("MEDIUM RISK", f"{probs.get('MEDIUM_RISK', 0):.1%}")
    with c3:
        st.metric("LOW RISK", f"{probs.get('LOW_RISK', 0):.1%}")

    divider()
    st.info("The NLP model is an independent signal. It analyzes language patterns in customer messages and is separate from the invoice ML classifier.")

    # Cross-reference with invoice risk
    section_label("Cross-Reference — Baseline Invoice Risk")
    baseline_payload = {
        "invoice_date": "2026-09-01",
        "due_date": "2026-10-01",
        "invoice_amount": 50000,
        "customer_seen_before": 6,
        "prior_late_count": 3,
        "prior_late_ratio": 0.5,
        "prior_avg_delay": 9,
        "industry": "Retail",
        "company_size": "Small",
        "payment_method": "Bank Transfer",
        "customer_segment": "SME",
        "outstanding_amount": 80000,
    }
    try:
        baseline = run_prediction(baseline_payload, "classifier", artifacts)
        st.metric("Baseline Invoice Risk (reference scenario)", f"{baseline['late_probability']:.1%}")
    except Exception:
        st.caption("Baseline invoice risk could not be computed for this deployment.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: BATCH ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def render_batch_predictions(artifacts: Dict[str, Any]) -> None:
    page_header(
        "Batch Analysis",
        "Upload an invoice portfolio CSV and receive AI risk predictions for every row.",
        eyebrow="Operations",
    )

    # Workflow stages
    st.markdown(
        """<div class="ig-wf">
          <div class="ig-wf-stage active">① Upload CSV</div>
          <div class="ig-wf-stage">② Validate</div>
          <div class="ig-wf-stage">③ Analyze</div>
          <div class="ig-wf-stage">④ Review Results</div>
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """<div class="ig-card" style="margin-bottom:1.25rem;">
          <div class="ig-card-title">Required CSV Columns</div>
          <p style="color:var(--muted);font-size:.82rem;margin:0;">
          invoice_date · due_date · invoice_amount · customer_seen_before · prior_late_count · prior_late_ratio · prior_avg_delay · industry · company_size · payment_method · customer_segment · outstanding_amount
          </p>
        </div>""",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Drop your CSV file here", type=["csv"], label_visibility="collapsed")
    if uploaded_file is None:
        return

    try:
        batch_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read uploaded CSV: {exc}")
        return

    required_cols = {
        "invoice_date", "due_date", "invoice_amount", "customer_seen_before",
        "prior_late_count", "prior_late_ratio", "prior_avg_delay", "industry",
        "company_size", "payment_method", "customer_segment", "outstanding_amount",
    }
    missing = sorted(required_cols.difference(batch_df.columns))
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return

    # Stage 2-3
    st.markdown(
        """<div class="ig-wf">
          <div class="ig-wf-stage done">① Upload CSV</div>
          <div class="ig-wf-stage done">② Validate</div>
          <div class="ig-wf-stage active">③ Analyze</div>
          <div class="ig-wf-stage">④ Review Results</div>
        </div>""",
        unsafe_allow_html=True,
    )

    with st.spinner(f"Processing {len(batch_df):,} invoices…"):
        rows = []
        for idx, row in batch_df.iterrows():
            payload = row.to_dict()
            errs = validate_single_input(payload)
            if errs:
                rows.append({"row_index": idx, "status": "invalid", "errors": "; ".join(errs)})
                continue
            try:
                pred = run_prediction(payload, "classifier", artifacts)
                rows.append({
                    "row_index": idx,
                    "status": "ok",
                    "late_probability": pred["late_probability"],
                    "expected_delay_days": pred["expected_delay_days"],
                    "risk_level": pred["risk_level"],
                    "estimated_financial_exposure": pred["estimated_financial_exposure"],
                    "recommendation": pred["recommendation"],
                })
            except Exception as exc:
                rows.append({"row_index": idx, "status": "failed", "errors": str(exc)})

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        st.warning("No rows were processed from the uploaded CSV.")
        return

    # Stage 4
    st.markdown(
        """<div class="ig-wf">
          <div class="ig-wf-stage done">① Upload CSV</div>
          <div class="ig-wf-stage done">② Validate</div>
          <div class="ig-wf-stage done">③ Analyze</div>
          <div class="ig-wf-stage active">④ Review Results</div>
        </div>""",
        unsafe_allow_html=True,
    )

    ok_rows = result_df[result_df["status"] == "ok"]
    total_amount = float(batch_df.get("invoice_amount", pd.Series([0] * len(batch_df))).sum())

    if not ok_rows.empty:
        kpi_row([
            {"icon": "📄", "label": "Total Invoices", "value": f"{len(result_df):,}"},
            {"icon": "🔴", "label": "High-Risk",  "value": f"{int(ok_rows['risk_level'].isin(['HIGH','CRITICAL']).sum()):,}"},
            {"icon": "🟡", "label": "Medium-Risk", "value": f"{int((ok_rows['risk_level'] == 'MEDIUM').sum()):,}"},
            {"icon": "🟢", "label": "Low-Risk",    "value": f"{int((ok_rows['risk_level'] == 'LOW').sum()):,}"},
            {"icon": "💰", "label": "Total Amount", "value": f"${total_amount:,.0f}"},
            {"icon": "📉", "label": "Exposure",     "value": f"${float(ok_rows['estimated_financial_exposure'].sum()):,.0f}"},
        ])
        divider()

    st.dataframe(result_df, use_container_width=True, hide_index=True)

    csv_out = result_df.to_csv(index=False)
    st.download_button(
        "⬇️  Download Results CSV",
        data=csv_out,
        file_name="invoiceguard_batch_results.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ALERT CENTER
# ─────────────────────────────────────────────────────────────────────────────

def render_alert_center(artifacts: Dict[str, Any]) -> None:
    page_header(
        "Alert Center",
        "Priority actions surfaced from invoice risk, customer trends, and due-date urgency.",
        eyebrow="Operations",
    )
    analytics = build_dashboard_dataset(artifacts)
    if analytics.empty:
        st.markdown('<div class="ig-empty"><div class="ig-empty-icon">🚨</div><div class="ig-empty-title">No alert data</div><div class="ig-empty-sub">Demo dataset unavailable.</div></div>', unsafe_allow_html=True)
        return

    high_risk = analytics[analytics["risk_level"].isin(["HIGH", "CRITICAL"])].sort_values("late_probability", ascending=False).head(10)
    due_soon = analytics[analytics["due_soon"]].sort_values("days_to_due").head(10)

    # Critical alerts
    section_label("Critical & High-Risk Invoices")
    if high_risk.empty:
        st.success("✅ No high-risk invoices detected in the current dataset.")
    else:
        for _, row in high_risk.iterrows():
            rl = row.get("risk_level", "HIGH")
            al_cls = "ig-al-crit" if rl == "CRITICAL" else "ig-al-high"
            customer = row.get("customer", "—")
            inv_id = row.get("invoice_id", "—")
            amount = row.get("outstanding_amount", 0)
            prob = row.get("late_probability", 0)
            exposure = row.get("estimated_financial_exposure", 0)
            delay = row.get("expected_delay_days", 0)
            b = badge(rl)
            st.markdown(
                f"""<div class="ig-al {al_cls}">
                  <div class="ig-al-top">
                    {b}
                    <span class="ig-al-customer">{customer}</span>
                    <span class="ig-al-invoice">· {inv_id}</span>
                  </div>
                  <div class="ig-al-bottom">
                    <span>Amount: <strong>${amount:,.0f}</strong></span>
                    <span>Risk: <strong>{prob:.0%}</strong></span>
                    <span>Exposure: <strong>${exposure:,.0f}</strong></span>
                    <span>Expected delay: <strong>{delay:.1f} days</strong></span>
                  </div>
                </div>""",
                unsafe_allow_html=True,
            )

    divider()

    # Due soon
    section_label("Invoices Approaching Due Date (≤ 7 days)")
    if due_soon.empty:
        st.success("✅ No invoices are approaching their due date within the current threshold.")
    else:
        for _, row in due_soon.iterrows():
            rl = row.get("risk_level", "LOW")
            al_cls = "ig-al-crit" if rl == "CRITICAL" else ("ig-al-high" if rl == "HIGH" else ("ig-al-med" if rl == "MEDIUM" else "ig-al-info"))
            b = badge(rl)
            st.markdown(
                f"""<div class="ig-al {al_cls}">
                  <div class="ig-al-top">
                    {b}
                    <span class="ig-al-customer">{row.get('customer', '—')}</span>
                    <span class="ig-al-invoice">· {row.get('invoice_id', '—')}</span>
                  </div>
                  <div class="ig-al-bottom">
                    <span>Due in: <strong>{row.get('days_to_due', 0)} days</strong></span>
                    <span>Amount: <strong>${row.get('outstanding_amount', 0):,.0f}</strong></span>
                    <span>Risk: <strong>{row.get('late_probability', 0):.0%}</strong></span>
                  </div>
                </div>""",
                unsafe_allow_html=True,
            )

    divider()

    # Rising customers
    section_label("Customers with Elevated Risk")
    cust_summary = (
        analytics.groupby("customer")
        .agg(
            invoice_count=("invoice_id", "count"),
            avg_probability=("late_probability", "mean"),
            latest_risk=("risk_level", lambda s: s.iloc[-1] if len(s) else "LOW"),
        )
        .reset_index()
    )
    rising = cust_summary[cust_summary["avg_probability"] >= 0.6].sort_values("avg_probability", ascending=False)
    if rising.empty:
        st.success("✅ No customers currently show elevated rising-risk conditions in the available dataset.")
    else:
        st.dataframe(rising, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MODEL CENTER
# ─────────────────────────────────────────────────────────────────────────────

def render_model_center(artifacts: Dict[str, Any]) -> None:
    page_header(
        "Model Center",
        "AI model inventory, performance metrics, and artifact configuration.",
        eyebrow="System",
    )
    reports = artifacts.get("reports", {})
    metadata = artifacts.get("metadata", {})
    models = artifacts.get("models", {})

    test_m = reports.get("test_metrics.json", metadata.get("test_metrics", {}))
    reg_m  = reports.get("regression_metrics.json", metadata.get("regression_metrics", {}))
    nlp_m  = reports.get("nlp_metrics.json", {})

    # Model cards
    section_label("Classifier Models")
    model_cards = [
        {
            "name": "XGBoost Classifier",
            "role": "Primary Risk Classifier",
            "key": "classifier",
            "metrics": {
                "Accuracy":  f"{test_m.get('accuracy', 0):.1%}" if test_m else "—",
                "Precision": f"{test_m.get('precision', 0):.1%}" if test_m else "—",
                "Recall":    f"{test_m.get('recall', 0):.1%}"    if test_m else "—",
                "F1 Score":  f"{test_m.get('f1', 0):.3f}"        if test_m else "—",
                "ROC-AUC":   f"{test_m.get('roc_auc', 0):.3f}"   if test_m else "—",
                "Threshold": f"{test_m.get('threshold', 0):.3f}" if test_m else "—",
            },
        },
        {
            "name": "Delay Regressor",
            "role": "Payment Delay Estimator",
            "key": "delay_regressor",
            "metrics": {
                "MAE (days)": f"{reg_m.get('mae_days', 0):.1f}"    if reg_m else "—",
                "RMSE (days)":f"{reg_m.get('rmse_days', 0):.1f}"   if reg_m else "—",
                "Test samples":f"{reg_m.get('n_test_delayed', '—')}" if reg_m else "—",
            },
        },
        {
            "name": "NLP Risk Classifier",
            "role": "Message Risk Intelligence",
            "key": "nlp_payment_risk",
            "metrics": {
                "Best model": nlp_m.get("best_model", "—"),
                "High thresh": f"{nlp_m.get('thresholds', {}).get('high', '—')}",
                "Med thresh":  f"{nlp_m.get('thresholds', {}).get('medium', '—')}",
            },
        },
    ]

    cols = st.columns(3)
    for col, card in zip(cols, model_cards):
        with col:
            status = "ACTIVE" if card["key"] in models else "UNAVAILABLE"
            s_badge = badge(status)
            metrics_html = "".join(
                f'<div class="ig-mcard-row"><span class="ig-mcard-k">{k}</span><span class="ig-mcard-v">{v}</span></div>'
                for k, v in card["metrics"].items()
            )
            st.markdown(
                f"""<div class="ig-mcard">
                  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:.2rem;">
                    <div class="ig-mcard-name">{card['name']}</div>
                    {s_badge}
                  </div>
                  <div class="ig-mcard-role">{card['role']}</div>
                  {metrics_html}
                </div>""",
                unsafe_allow_html=True,
            )

    # Additional models
    divider()
    section_label("Alternative Classifiers")
    alt_models = [
        ("Logistic Regression", "logistic_regression"),
        ("Random Forest", "random_forest"),
        ("MLP Neural Network", "mlp_neural_network"),
    ]
    alt_cols = st.columns(3)
    for col, (name, key) in zip(alt_cols, alt_models):
        with col:
            status = "ACTIVE" if key in models else "NOT LOADED"
            s_badge = badge("ACTIVE" if key in models else "FAIL")
            st.markdown(
                f"""<div class="ig-mcard">
                  <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                    <div class="ig-mcard-name">{name}</div>
                    {s_badge}
                  </div>
                  <div class="ig-mcard-role">Alternative Classifier</div>
                  <div class="ig-mcard-row"><span class="ig-mcard-k">Status</span><span class="ig-mcard-v">{status}</span></div>
                </div>""",
                unsafe_allow_html=True,
            )

    # Model comparison
    comparison_df, nlp_compare_df = load_comparison_reports()
    if comparison_df is not None:
        divider()
        section_label("Model Comparison Report")
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    if nlp_compare_df is not None:
        divider()
        section_label("NLP Model Comparison")
        st.dataframe(nlp_compare_df, use_container_width=True, hide_index=True)


@st.cache_data(show_spinner=False)
def load_comparison_reports() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    model_compare_path = REPORTS / "model_comparison.csv"
    nlp_compare_path = REPORTS / "nlp_model_comparison.csv"
    comp_df = pd.read_csv(model_compare_path) if model_compare_path.exists() else None
    nlp_df = pd.read_csv(nlp_compare_path) if nlp_compare_path.exists() else None
    return comp_df, nlp_df



# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ARTIFACT DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def render_artifact_diagnostics(artifacts: Dict[str, Any]) -> None:
    page_header(
        "Artifact Diagnostics",
        "System health, loaded artifact status, thresholds, and runtime configuration.",
        eyebrow="System",
    )
    models = artifacts.get("models", {})
    load_errors = artifacts.get("load_errors", {})
    metadata = artifacts.get("metadata", {})
    reports = artifacts.get("reports", {})

    all_healthy = not load_errors
    system_badge = badge("PASS" if all_healthy else "FAIL")
    st.markdown(
        f"""<div class="ig-card" style="margin-bottom:1.25rem;">
          <div style="display:flex;align-items:center;gap:1rem;">
            <div>
              <div class="ig-card-title">System Health</div>
              <div style="font-size:1.1rem;font-weight:700;color:var(--text);">
                {'All systems operational' if all_healthy else f'{len(load_errors)} artifact warning(s)'}
              </div>
            </div>
            {system_badge}
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Model artifacts
    section_label("Loaded Model Artifacts")
    artifact_defs = [
        ("XGBoost Classifier",   "classifier"),
        ("Delay Regressor",      "delay_regressor"),
        ("Logistic Regression",  "logistic_regression"),
        ("Random Forest",        "random_forest"),
        ("MLP Neural Network",   "mlp_neural_network"),
        ("NLP Classifier",       "nlp_payment_risk"),
    ]
    col1, col2 = st.columns(2)
    for i, (label, key) in enumerate(artifact_defs):
        col = col1 if i % 2 == 0 else col2
        with col:
            loaded = key in models
            s_badge = badge("PASS" if loaded else "FAIL")
            detail = "Loaded — available for inference" if loaded else load_errors.get(key, "File not found in /models")
            path = MODELS / f"{key}.joblib"
            size_str = f"{path.stat().st_size / 1024 / 1024:.1f} MB" if path.exists() else "—"
            st.markdown(
                f"""<div class="ig-card" style="margin-bottom:.6rem;">
                  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:.4rem;">
                    <strong style="color:var(--text);font-size:.9rem;">{label}</strong>
                    {s_badge}
                  </div>
                  <div class="ig-srow" style="padding:.2rem 0;">
                    <span class="ig-srow-lbl">File</span>
                    <code style="font-size:.75rem;color:var(--muted)">{key}.joblib</code>
                  </div>
                  <div class="ig-srow" style="padding:.2rem 0;border-bottom:none;">
                    <span class="ig-srow-lbl">Size</span>
                    <span style="color:var(--muted);font-size:.8rem;">{size_str}</span>
                  </div>
                  <div style="font-size:.75rem;color:var(--muted);margin-top:.3rem;">{detail}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    divider()

    # Reports
    section_label("Report Artifacts")
    report_files = [
        ("test_metrics.json", "Classifier Test Metrics"),
        ("regression_metrics.json", "Regression Metrics"),
        ("nlp_metrics.json", "NLP Model Metrics"),
        ("nlp_thresholds.json", "NLP Decision Thresholds"),
        ("all_model_artifacts.json", "Model Artifacts Summary"),
    ]
    st.markdown('<div class="ig-card">', unsafe_allow_html=True)
    for fname, label in report_files:
        present = fname in reports
        s_badge = badge("PASS" if present else "FAIL")
        st.markdown(
            f"""<div class="ig-srow">
              <span class="ig-srow-lbl">{label} <code style="font-size:.72rem;color:var(--muted2)">({fname})</code></span>
              {s_badge}
            </div>""",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    divider()

    # Configuration
    section_label("Model Configuration")
    if metadata:
        best_model = metadata.get("best_model", "—")
        threshold = metadata.get("threshold", "—")
        features = metadata.get("features", [])
        st.markdown(
            f"""<div class="ig-card">
              <div class="ig-srow"><span class="ig-srow-lbl">Best Model</span><strong style="color:var(--text)">{best_model}</strong></div>
              <div class="ig-srow"><span class="ig-srow-lbl">Decision Threshold</span><strong style="color:var(--text)">{threshold}</strong></div>
              <div class="ig-srow" style="border-bottom:none;"><span class="ig-srow-lbl">Feature Count</span><strong style="color:var(--text)">{len(features)}</strong></div>
            </div>""",
            unsafe_allow_html=True,
        )
        if features:
            with st.expander("Feature Order", expanded=False):
                st.code("\n".join(features), language="text")

    if load_errors:
        divider()
        section_label("Load Warnings")
        for key, err in load_errors.items():
            st.warning(f"**{key}**: {err}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    from src.ui.css import inject_css
    from src.ui.navigation import render_sidebar
    from src.ui.views.overview import render_overview
    from src.ui.views.receivables import render_collections
    from src.ui.views.customer_search import render_customer_search
    from src.ui.views.revenue_intelligence import render_revenue_forecast, render_repurchase_risk, render_revenue_at_risk
    from src.ui.views.ai_insights import render_ai_recommendations, render_risk_signals
    from src.ui.views.system import render_data_quality
    
    # We use inject_css from src.ui.css to get the new enterprise styles, but keep old ones if needed
    inject_css()

    artifacts = load_project_artifacts(get_artifact_signature())
    models = artifacts.get("models", {})
    load_errors = artifacts.get("load_errors", {})

    # Sidebar navigation using the new router
    current_page = render_sidebar()

    # Startup guard
    required = ["classifier", "delay_regressor"]
    failed = [n for n in required if n not in models]
    if failed:
        st.error("⚠️ Fatal startup error: required artifacts failed to load.")
        for n in failed:
            st.error(f"Required artifact **{n}** — {load_errors.get(n, 'missing from /models')}")
        st.stop()

    # Page routing
    page_map = {
        "Overview":             render_overview,
        "Invoice Risk":         lambda a: render_single_prediction(a, cold_start=False),
        "Collections":          render_collections,
        "Customer 360":         render_customer_360,
        "Customer Search":      render_customer_search,
        "Customer Risk":        render_risk_center,
        "Revenue Forecast":     render_revenue_forecast,
        "Repurchase Risk":      render_repurchase_risk,
        "Revenue at Risk":      render_revenue_at_risk,
        "AI Recommendations":   render_ai_recommendations,
        "Risk Signals":         render_risk_signals,
        "Explainable AI":       render_ai_explanation,
        "Model Performance":    render_model_center,
        "Data Quality":         render_data_quality,
        "System Health":        render_artifact_diagnostics,
    }

    render_fn = page_map.get(current_page, render_overview)
    render_fn(artifacts)


if __name__ == "__main__":
    main()
