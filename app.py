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

st.set_page_config(page_title="InvoiceGuard AI", page_icon="💳", layout="wide")


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
            samples = data.get("samples", 0)
            if samples:
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
        return [str(value) for value in model.classes_]

    if hasattr(model, "named_steps"):
        classifier = model.named_steps.get("classifier")
        if classifier is not None and hasattr(classifier, "classes_"):
            return [str(value) for value in classifier.classes_]

    return ["HIGH_RISK", "LOW_RISK", "MEDIUM_RISK"]


def score_nlp_message(model: Any, customer_message: str, thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    message = (customer_message or "").strip()
    if not message:
        raise ValueError("Customer message cannot be empty.")

    try:
        probabilities = np.asarray(model.predict_proba([message])[0], dtype=float)
    except Exception as exc:  # pragma: no cover - defensive guard for optional model artifacts
        raise RuntimeError(f"Unable to evaluate NLP message: {exc}") from exc

    class_order = get_nlp_class_order(model)
    probability_map = dict(zip(class_order, probabilities.tolist()))

    if len(probabilities) < 3:
        raise RuntimeError("NLP model did not return the expected three-class probability vector.")

    resolved_thresholds = thresholds or {"high": 0.5, "medium": 0.45}
    high_probability = float(probability_map.get("HIGH_RISK", 0.0))
    medium_probability = float(probability_map.get("MEDIUM_RISK", 0.0))
    low_probability = float(probability_map.get("LOW_RISK", 0.0))

    if high_probability >= float(resolved_thresholds.get("high", 0.5)):
        risk_label = "HIGH RISK"
        selected_probability = high_probability
    elif medium_probability >= float(resolved_thresholds.get("medium", 0.45)):
        risk_label = "MEDIUM RISK"
        selected_probability = medium_probability
    else:
        risk_label = "LOW RISK"
        selected_probability = low_probability

    return {
        "risk_label": risk_label,
        "probability": selected_probability,
        "probabilities": probability_map,
        "thresholds": resolved_thresholds,
    }


def inject_fintech_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #06131f;
            --bg-2: #0b1724;
            --panel: rgba(15, 23, 42, 0.88);
            --panel-soft: rgba(17, 24, 39, 0.7);
            --panel-border: rgba(148, 163, 184, 0.18);
            --text: #ecfeff;
            --muted: #a5b4c9;
            --primary: #7dd3fc;
            --primary-2: #38bdf8;
            --success: #34d399;
            --warning: #fbbf24;
            --danger: #f87171;
            --accent: #a78bfa;
        }

        html, body, [data-testid="stAppViewContainer"] {
            background: linear-gradient(180deg, #081822 0%, #0b1724 100%);
            color: var(--text);
        }

        .main .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
            max-width: 1500px;
        }

        .stApp {
            background: linear-gradient(180deg, #081822 0%, #0b1724 100%);
        }

        h1, h2, h3, h4 {
            letter-spacing: -0.03em;
        }

        .app-shell {
            background: rgba(10, 20, 31, 0.66);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 22px;
            padding: 0.9rem 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 16px 30px rgba(2, 6, 23, 0.18);
        }

        .topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
        }

        .brand-stack {
            display: flex;
            flex-direction: column;
            gap: 0.1rem;
        }

        .brand-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 1.6rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            color: #f8fafc;
        }

        .brand-subtitle {
            font-size: 0.82rem;
            color: var(--muted);
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .header-actions {
            display: flex;
            flex-wrap: wrap;
            justify-content: flex-end;
            gap: 0.55rem;
        }

        .header-chip {
            display: inline-flex;
            align-items: center;
            padding: 0.38rem 0.7rem;
            border-radius: 999px;
            border: 1px solid rgba(125, 211, 252, 0.2);
            background: rgba(15, 23, 42, 0.72);
            color: #dbeafe;
            font-size: 0.74rem;
            font-weight: 600;
        }

        .header-chip.success {
            color: #a7f3d0;
            border-color: rgba(52, 211, 153, 0.22);
            background: rgba(16, 185, 129, 0.08);
        }

        .header-chip.warning {
            color: #fcd34d;
            border-color: rgba(251, 191, 36, 0.24);
            background: rgba(245, 158, 11, 0.08);
        }

        .hero-shell {
            background: linear-gradient(135deg, rgba(15, 118, 110, 0.18), rgba(59, 130, 246, 0.12), rgba(168, 85, 247, 0.16));
            border: 1px solid rgba(125, 211, 252, 0.22);
            border-radius: 24px;
            padding: 1.4rem 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 20px 35px rgba(2, 6, 23, 0.28);
        }

        .hero-title {
            font-size: 2.15rem;
            font-weight: 800;
            margin: 0;
            color: #f8fafc;
        }

        .hero-subtitle {
            font-size: 1rem;
            color: var(--muted);
            margin-top: 0.2rem;
        }

        .section-header {
            margin: 0.8rem 0 0.9rem;
            padding: 0.15rem 0;
        }

        .section-title {
            font-size: 1.2rem;
            font-weight: 700;
            color: #f8fafc;
            margin-bottom: 0.15rem;
        }

        .section-caption {
            font-size: 0.86rem;
            color: var(--muted);
        }

        .section-shell {
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.92), rgba(15, 23, 42, 0.74));
            border: 1px solid var(--panel-border);
            border-radius: 20px;
            padding: 1.15rem;
            margin-bottom: 1rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.12);
        }

        .kpi-card {
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.95), rgba(15, 23, 42, 0.9));
            border: 1px solid rgba(96, 165, 250, 0.25);
            border-radius: 18px;
            padding: 1rem 1rem 0.9rem;
            height: 100%;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
        }

        .kpi-label {
            font-size: 0.72rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            margin-bottom: 0.45rem;
        }

        .kpi-value {
            font-size: 1.62rem;
            font-weight: 800;
            color: #f8fafc;
            line-height: 1.1;
            margin-bottom: 0.2rem;
        }

        .kpi-sub {
            font-size: 0.82rem;
            color: #cbd5e1;
        }

        .badge {
            display: inline-block;
            padding: 0.28rem 0.68rem;
            border-radius: 999px;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            border: 1px solid rgba(255,255,255,0.12);
        }

        .badge-low { background: rgba(34, 197, 94, 0.18); color: #86efac; }
        .badge-medium { background: rgba(250, 204, 21, 0.18); color: #facc15; }
        .badge-high { background: rgba(251, 146, 60, 0.18); color: #fdba74; }
        .badge-critical { background: rgba(239, 68, 68, 0.20); color: #fca5a5; }

        .status-panel {
            border-left: 4px solid #38bdf8;
            background: rgba(14, 116, 144, 0.1);
            padding: 0.8rem 1rem;
            border-radius: 14px;
            margin: 0.5rem 0 1rem;
        }

        .status-panel strong {
            color: #f8fafc;
        }

        .sidebar-shell {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 18px;
            padding: 0.85rem 0.95rem;
            margin-bottom: 0.8rem;
        }

        .sidebar-brand {
            font-size: 1rem;
            font-weight: 750;
            color: #f8fafc;
            letter-spacing: -0.02em;
        }

        .sidebar-subtitle {
            font-size: 0.72rem;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.07em;
            margin-top: 0.1rem;
        }

        .sidebar-status {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.5rem;
            font-size: 0.78rem;
            color: #cbd5e1;
            margin-top: 0.3rem;
        }

        .sidebar-status strong {
            color: #f8fafc;
        }

        .focus-panel {
            background: rgba(14, 116, 144, 0.08);
            border: 1px solid rgba(125, 211, 252, 0.18);
            border-radius: 18px;
            padding: 1rem;
        }

        .dataframe {
            background: rgba(15, 23, 42, 0.6);
            border-radius: 12px;
            border: 1px solid rgba(148, 163, 184, 0.12);
        }

        div[data-testid="stSidebar"] {
            background: rgba(6, 19, 31, 0.92);
            border-right: 1px solid rgba(148, 163, 184, 0.12);
        }

        .sidebar .block-container {
            padding-top: 1.2rem;
        }

        .stButton > button {
            border-radius: 12px;
            font-weight: 600;
            border: 1px solid rgba(125, 211, 252, 0.2);
            background: linear-gradient(180deg, rgba(59, 130, 246, 0.18), rgba(59, 130, 246, 0.08));
            color: #e2e8f0;
        }

        .stButton > button:hover {
            border-color: rgba(125, 211, 252, 0.45);
        }

        .stSelectbox > div > div,
        .stNumberInput > div > div,
        .stTextInput > div > div,
        .stTextArea > div > div,
        .stDateInput > div > div,
        .stFileUploader > div {
            border-radius: 12px;
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.18);
        }

        .stMetric {
            background: rgba(15, 23, 42, 0.7);
            border-radius: 14px;
            padding: 0.8rem 0.9rem;
            border: 1px solid rgba(148, 163, 184, 0.18);
        }

        [data-testid="metric-container"] {
            background: rgba(15, 23, 42, 0.7);
            border-radius: 14px;
            border: 1px solid rgba(148, 163, 184, 0.18);
            padding: 0.6rem 0.75rem;
        }

        .result-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.96), rgba(15, 23, 42, 0.92));
            border: 1px solid rgba(125, 211, 252, 0.24);
            border-radius: 22px;
            padding: 1.25rem;
            margin: 0.75rem 0 1rem;
            box-shadow: 0 18px 30px rgba(2, 6, 23, 0.2);
        }

        .result-card h3 {
            margin-top: 0;
            margin-bottom: 0.2rem;
            font-size: 1rem;
            color: var(--muted);
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        .result-card .score {
            font-size: 2rem;
            font-weight: 800;
            color: #f8fafc;
            line-height: 1.1;
            margin-bottom: 0.4rem;
        }

        .result-card .score-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 0.75rem;
        }

        .result-card .score-item {
            background: rgba(15, 23, 42, 0.7);
            border-radius: 12px;
            border: 1px solid rgba(148, 163, 184, 0.18);
            padding: 0.75rem 0.8rem;
        }

        .result-card .score-item-label {
            font-size: 0.7rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.2rem;
        }

        .result-card .score-item-value {
            font-size: 1.05rem;
            font-weight: 700;
            color: #f8fafc;
        }

        .stApp [data-testid="stSidebar"] {
            background: rgba(6, 19, 31, 0.97);
            border-right: 1px solid rgba(148, 163, 184, 0.14);
        }

        .stApp [data-testid="stSidebar"] .block-container {
            padding-top: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .stApp [data-testid="stSidebarUserContent"] {
            gap: 0.75rem;
        }

        [data-testid="stDataFrame"] {
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid rgba(148, 163, 184, 0.14);
            background: rgba(15, 23, 42, 0.76);
        }

        [data-testid="stDataFrame"] .dataframe-container {
            border-radius: 16px;
            overflow: hidden;
        }

        [data-testid="stDataFrame"] table {
            background: rgba(15, 23, 42, 0.78);
            border-collapse: separate;
            border-spacing: 0;
        }

        [data-testid="stDataFrame"] thead th {
            background: rgba(15, 118, 110, 0.18);
            color: #e2e8f0;
            font-weight: 700;
            border-bottom: 1px solid rgba(148, 163, 184, 0.18);
        }

        [data-testid="stDataFrame"] tbody td {
            border-bottom: 1px solid rgba(148, 163, 184, 0.08);
            color: #cbd5e1;
        }

        [data-testid="stDataFrame"] tbody tr:hover {
            background: rgba(59, 130, 246, 0.08);
        }

        [data-testid="stCodeBlock"] {
            border-radius: 14px;
            border: 1px solid rgba(148, 163, 184, 0.12);
            background: rgba(15, 23, 42, 0.75);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
        }

        div[data-testid="stJson"] {
            border-radius: 14px;
            border: 1px solid rgba(148, 163, 184, 0.12);
            background: rgba(15, 23, 42, 0.7);
            padding: 0.5rem 0.6rem;
        }

        .stAlert {
            border-radius: 14px;
            border: 1px solid rgba(148, 163, 184, 0.12);
            background: rgba(15, 23, 42, 0.72);
        }

        [data-testid="stNotification"] {
            border-radius: 14px;
            border: 1px solid rgba(148, 163, 184, 0.12);
        }

        [data-testid="stMetric"] {
            border-radius: 14px;
            border: 1px solid rgba(148, 163, 184, 0.14);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
        }

        [data-testid="stMetricLabel"] {
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.7rem;
            font-weight: 700;
        }

        [data-testid="stMetricValue"] {
            color: #f8fafc;
            font-weight: 800;
        }

        .stDownloadButton > button,
        .stButton > button {
            border-radius: 12px;
            font-weight: 700;
            letter-spacing: -0.01em;
            transition: transform 150ms ease, box-shadow 150ms ease, border-color 150ms ease;
        }

        .stDownloadButton > button:hover,
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 18px rgba(56, 189, 248, 0.12);
        }

        div[data-testid="stBlockContainer"] {
            min-height: 0;
        }

        .stTabs [role="tablist"] {
            gap: 0.45rem;
        }

        .stTabs [role="tab"] {
            border-radius: 10px 10px 0 0;
            border: 1px solid rgba(148, 163, 184, 0.12);
            background: rgba(15, 23, 42, 0.66);
            color: #cbd5e1;
            padding: 0.55rem 0.9rem;
        }

        .stTabs [role="tab"][aria-selected="true"] {
            background: rgba(59, 130, 246, 0.18);
            border-color: rgba(125, 211, 252, 0.28);
            color: #f8fafc;
        }

        .stCheckbox,
        .stRadio,
        .stSelectbox,
        .stNumberInput,
        .stTextInput,
        .stTextArea {
            color: #f8fafc;
        }

        .stSelectbox label,
        .stNumberInput label,
        .stTextInput label,
        .stTextArea label,
        .stDateInput label,
        .stFileUploader label {
            color: #dbeafe;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_dashboard_dataset(artifacts: Dict[str, Any]) -> pd.DataFrame:
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
        df["invoice_id"] = [f"INV-{index:06d}" for index in range(len(df))]

    if "customer_id" in df.columns:
        df["customer"] = df["customer_id"]
    elif "customer" in df.columns:
        df["customer"] = df["customer"]
    else:
        df["customer"] = "UNKNOWN"

    for column in ["invoice_date", "due_date"]:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")

    for column in ["invoice_amount", "outstanding_amount"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    predictions: List[Dict[str, Any]] = []
    metadata = artifacts.get("metadata", {})
    selected_model = "classifier"

    for _, row in df.iterrows():
        payload = row.to_dict()
        try:
            prediction = run_prediction(payload, selected_model, artifacts)
        except Exception:
            continue

        record = row.to_dict()
        record["late_probability"] = float(prediction["late_probability"])
        record["expected_delay_days"] = float(prediction["expected_delay_days"])
        record["risk_level"] = prediction["risk_level"]
        record["estimated_financial_exposure"] = float(prediction["estimated_financial_exposure"])
        record["recommendation"] = prediction["recommendation"]
        record["predicted_late"] = int(prediction["late_probability"] >= metadata.get("threshold", 0.5))
        record["days_to_due"] = int((pd.Timestamp(record["due_date"]) - pd.Timestamp(record["invoice_date"])).days)
        predictions.append(record)

    if not predictions:
        return pd.DataFrame()

    analytics = pd.DataFrame(predictions)
    analytics["risk_level"] = analytics["risk_level"].fillna("LOW")
    analytics["predicted_late"] = analytics["predicted_late"].fillna(0).astype(int)
    analytics["amount_at_risk"] = analytics["outstanding_amount"].where(analytics["risk_level"].isin(["HIGH", "CRITICAL"]), 0)
    analytics["due_soon"] = analytics["days_to_due"].le(7)
    return analytics


def render_kpi_cards(kpis: List[Dict[str, Any]]) -> None:
    cols = st.columns(len(kpis))
    for idx, item in enumerate(kpis):
        with cols[idx]:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{item['label']}</div>
                    <div class="kpi-value">{item['value']}</div>
                    <div class="kpi-sub">{item.get('subtext', '')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_section_header(title: str, subtitle: str = "", badge: Optional[str] = None) -> None:
    badge_html = f'<span class="badge badge-medium">{badge}</span>' if badge else ""
    subtitle_html = f'<div class="section-caption">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="section-header">
            <div class="section-title">{title} {badge_html}</div>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_app_header(metadata: Dict[str, Any], models: Dict[str, Any], load_errors: Dict[str, Any]) -> None:
    best_model = metadata.get("best_model", "classifier")
    model_count = len(models)
    warning_count = len(load_errors)

    warning_chip = f'<span class="header-chip warning">{warning_count} warning(s)</span>' if warning_count else ''
    st.markdown(
        f"""
        <div class="app-shell">
            <div class="topbar">
                <div class="brand-stack">
                    <div class="brand-pill">InvoiceGuard AI</div>
                    <div class="brand-subtitle">AI-Powered Payment Risk Intelligence</div>
                </div>
                <div class="header-actions">
                    <span class="header-chip">Best model: {best_model}</span>
                    <span class="header-chip success">{model_count} models loaded</span>
                    {warning_chip}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_exec_dashboard(artifacts: Dict[str, Any]) -> None:
    analytics = build_dashboard_dataset(artifacts)
    metadata = artifacts.get("metadata", {})
    threshold = float(metadata.get("threshold", 0.5))

    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-title">InvoiceGuard AI</div>
            <div class="hero-subtitle">AI-powered payment risk & financial intelligence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if analytics.empty:
        st.warning("No demo invoice dataset is available for dashboard analytics.")
        return

    total_receivables = float(analytics["outstanding_amount"].sum())
    at_risk_amount = float(analytics.loc[analytics["risk_level"].isin(["HIGH", "CRITICAL"]), "outstanding_amount"].sum())
    predicted_late_invoices = int((analytics["late_probability"] >= threshold).sum())
    high_risk_customers = int(analytics.loc[analytics["risk_level"].isin(["HIGH", "CRITICAL"]), "customer"].nunique())
    expected_delayed_cash = float(analytics.loc[analytics["risk_level"].isin(["HIGH", "CRITICAL"]), "estimated_financial_exposure"].sum())

    render_section_header("Executive Overview", "Portfolio health, risk concentration, and model-driven financial signal summary.", "Live")
    render_kpi_cards(
        [
            {"label": "Total Receivables", "value": f"${total_receivables:,.2f}", "subtext": f"{len(analytics)} invoices"},
            {"label": "At-Risk Amount", "value": f"${at_risk_amount:,.2f}", "subtext": "High + critical risk"},
            {"label": "Predicted Late Invoices", "value": f"{predicted_late_invoices}", "subtext": f"Threshold {threshold:.2f}"},
            {"label": "High-Risk Customers", "value": f"{high_risk_customers}", "subtext": "Customers needing attention"},
            {"label": "Expected Delayed Cash", "value": f"${expected_delayed_cash:,.2f}", "subtext": "Projected exposure"},
        ]
    )

    st.markdown("---")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        risk_distribution = analytics["risk_level"].value_counts().reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"], fill_value=0)
        render_section_header("Risk Overview", "Current risk distribution across the active portfolio.")
        st.bar_chart(risk_distribution)

    with chart_col2:
        trend = analytics.assign(month=analytics["invoice_date"].dt.to_period("M").astype(str)).groupby("month")["predicted_late"].sum().sort_index()
        render_section_header("Risk Trend", "Late-payment momentum over time.")
        st.line_chart(trend)

    chart_col3, chart_col4 = st.columns(2)
    with chart_col3:
        amount_by_month = analytics.assign(month=analytics["invoice_date"].dt.to_period("M").astype(str)).groupby("month")["estimated_financial_exposure"].sum().sort_index()
        render_section_header("Financial Exposure", "Projected revenue exposure by month.")
        st.area_chart(amount_by_month)

    with chart_col4:
        customer_risk = analytics.groupby("customer")["late_probability"].mean().sort_values(ascending=False).head(10)
        render_section_header("Top Risk Drivers", "Highest-risk customers by average probability.")
        st.bar_chart(customer_risk)

    if "industry" in analytics.columns:
        st.markdown("---")
        render_section_header("Risk by Industry", "Cross-industry payment risk signal benchmark.")
        industry_risk = analytics.groupby("industry")["late_probability"].mean().sort_values(ascending=False)
        st.bar_chart(industry_risk)

    st.markdown("---")
    render_section_header("Portfolio Snapshot", "Most relevant invoices ranked by payment risk and financial impact.")
    display_cols = ["invoice_id", "customer", "invoice_date", "due_date", "invoice_amount", "outstanding_amount", "risk_level", "late_probability", "estimated_financial_exposure"]
    st.dataframe(analytics[display_cols].sort_values("late_probability", ascending=False).head(20), use_container_width=True)


def render_risk_center(artifacts: Dict[str, Any]) -> None:
    analytics = build_dashboard_dataset(artifacts)
    if analytics.empty:
        st.warning("No dashboard dataset is available for the risk center.")
        return

    render_section_header("Risk Center", "Monitor and prioritize payment risk across your portfolio.")

    col1, col2, col3 = st.columns(3)
    with col1:
        risk_filter = st.selectbox("Risk level", ["All", *sorted(analytics["risk_level"].dropna().unique())])
    with col2:
        customer_filter = st.selectbox("Customer", ["All", *sorted(analytics["customer"].dropna().unique())])
    with col3:
        amount_max = int(analytics["invoice_amount"].quantile(0.95))
        amount_cap = st.slider("Invoice amount max", min_value=0, max_value=max(amount_max, 1), value=max(amount_max, 1))

    filtered = analytics.copy()
    if risk_filter != "All":
        filtered = filtered[filtered["risk_level"] == risk_filter]
    if customer_filter != "All":
        filtered = filtered[filtered["customer"] == customer_filter]
    filtered = filtered[filtered["invoice_amount"] <= amount_cap]

    if "industry" in filtered.columns:
        industry_filter = st.selectbox("Industry", ["All", *sorted(filtered["industry"].dropna().unique())])
        if industry_filter != "All":
            filtered = filtered[filtered["industry"] == industry_filter]

    st.write(f"Showing {len(filtered)} invoice(s) matching the current filters.")

    display_cols = ["invoice_id", "customer", "industry", "invoice_date", "due_date", "invoice_amount", "late_probability", "expected_delay_days", "risk_level", "estimated_financial_exposure", "recommendation"]
    filtered = filtered[display_cols]
    st.dataframe(filtered.sort_values("late_probability", ascending=False), use_container_width=True)


def render_customer_360(artifacts: Dict[str, Any]) -> None:
    analytics = build_dashboard_dataset(artifacts)
    if analytics.empty:
        st.warning("No customer data is available for the customer 360 view.")
        return

    render_section_header("Customer 360", "Customer intelligence with payment behavior, risk history, and invoice context.")
    customer_option = st.selectbox("Select customer", sorted(analytics["customer"].dropna().unique()))
    customer_df = analytics[analytics["customer"] == customer_option].sort_values("invoice_date")

    if customer_df.empty:
        st.warning("No invoices found for the selected customer.")
        return

    total_invoices = len(customer_df)
    on_time_rate = float((customer_df["predicted_late"] == 0).mean())
    late_rate = 1 - on_time_rate
    avg_delay = float(customer_df["expected_delay_days"].mean())
    outstanding_amount = float(customer_df["outstanding_amount"].sum())
    current_risk = customer_df.iloc[-1]["risk_level"] if not customer_df.empty else "LOW"

    kpi_columns = st.columns(6)
    kpi_columns[0].metric("Invoices", total_invoices)
    kpi_columns[1].metric("On-time rate", f"{on_time_rate:.1%}")
    kpi_columns[2].metric("Late-payment rate", f"{late_rate:.1%}")
    kpi_columns[3].metric("Average delay", f"{avg_delay:.1f} days")
    kpi_columns[4].metric("Outstanding amount", f"${outstanding_amount:,.2f}")
    kpi_columns[5].metric("Current risk", current_risk)

    st.markdown("---")
    st.subheader("Historical risk trend")
    trend_df = customer_df.set_index("invoice_date")["late_probability"].sort_index()
    st.line_chart(trend_df)

    display_cols = ["invoice_id", "invoice_date", "due_date", "invoice_amount", "outstanding_amount", "late_probability", "expected_delay_days", "risk_level", "estimated_financial_exposure"]
    st.dataframe(customer_df[display_cols], use_container_width=True)


def render_financial_impact(artifacts: Dict[str, Any]) -> None:
    analytics = build_dashboard_dataset(artifacts)
    if analytics.empty:
        st.warning("No financial dataset is available.")
        return

    render_section_header("Financial Impact", "Quantify revenue at risk, exposure, and recovery potential.")
    total_outstanding = float(analytics["outstanding_amount"].sum())
    amount_at_risk = float(analytics.loc[analytics["risk_level"].isin(["HIGH", "CRITICAL"]), "outstanding_amount"].sum())
    expected_delayed_cash = float(analytics.loc[analytics["risk_level"].isin(["HIGH", "CRITICAL"]), "estimated_financial_exposure"].sum())
    estimated_exposure = float(analytics["estimated_financial_exposure"].sum())

    cols = st.columns(4)
    cols[0].metric("Total Outstanding", f"${total_outstanding:,.2f}")
    cols[1].metric("Amount at Risk", f"${amount_at_risk:,.2f}")
    cols[2].metric("Expected Delayed Cash", f"${expected_delayed_cash:,.2f}")
    cols[3].metric("Estimated Exposure", f"${estimated_exposure:,.2f}")

    st.markdown("---")
    st.subheader("Top invoices by financial exposure")
    top_df = analytics.nlargest(15, "estimated_financial_exposure")[["invoice_id", "customer", "invoice_amount", "outstanding_amount", "estimated_financial_exposure", "risk_level"]]
    st.dataframe(top_df, use_container_width=True)


def render_ai_explanation(artifacts: Dict[str, Any]) -> None:
    analytics = build_dashboard_dataset(artifacts)
    if analytics.empty:
        st.warning("No prediction data is available for explanation analysis.")
        return

    render_section_header("AI Explanation", "Explainable AI view showing the strongest drivers behind each prediction.")
    invoice_options = [row["invoice_id"] for _, row in analytics.iterrows()]
    selected_invoice = st.selectbox("Select invoice", invoice_options)
    selected_row = analytics[analytics["invoice_id"] == selected_invoice].iloc[0]
    model_options = [name for name in ["classifier", "logistic_regression", "random_forest", "mlp_neural_network"] if name in artifacts["models"]]

    if not model_options:
        st.warning("No compatible classifier artifacts are available for model explanation.")
        return

    selected_model = st.selectbox("Model for explanation", model_options, index=0)
    result = run_prediction(selected_row.to_dict(), selected_model, artifacts)

    st.info(f"Selected invoice: {selected_invoice} | Model: {selected_model}")

    explanation = explain_prediction(artifacts["models"].get(selected_model), result["feature_frame"], artifacts["metadata"])
    if explanation:
        explanation_df = pd.DataFrame(explanation)
        st.dataframe(explanation_df, use_container_width=True)
        st.bar_chart(explanation_df.set_index("feature")["impact"])
    else:
        st.warning("No SHAP or model-compatible feature importance data is available for this model in the current deployment.")


def render_alert_center(artifacts: Dict[str, Any]) -> None:
    analytics = build_dashboard_dataset(artifacts)
    if analytics.empty:
        st.warning("No alert data is available.")
        return

    render_section_header("Alert Center", "Priority actions surfaced from invoice risk, customer trends, and due-date urgency.")

    high_risk_invoices = analytics[analytics["risk_level"].isin(["HIGH", "CRITICAL"])].sort_values("late_probability", ascending=False).head(10)
    due_soon = analytics[analytics["due_soon"]].sort_values("days_to_due").head(10)

    st.subheader("Actionable alerts")
    if high_risk_invoices.empty:
        st.info("No high-risk invoices were detected in the current dataset.")
    else:
        st.dataframe(high_risk_invoices[["invoice_id", "customer", "risk_level", "late_probability", "estimated_financial_exposure", "days_to_due"]], use_container_width=True)

    if due_soon.empty:
        st.info("No invoices are approaching due date within the current threshold.")
    else:
        st.subheader("Invoices approaching due date")
        st.dataframe(due_soon[["invoice_id", "customer", "due_date", "days_to_due", "risk_level", "estimated_financial_exposure"]], use_container_width=True)

    customer_summary = analytics.groupby("customer").agg(
        invoice_count=("invoice_id", "count"),
        avg_probability=("late_probability", "mean"),
        latest_risk=("risk_level", lambda s: s.iloc[-1] if len(s) else "LOW")
    ).reset_index()

    st.subheader("Customers with rising risk")
    rising_customers = customer_summary[customer_summary["avg_probability"] >= 0.6]
    if rising_customers.empty:
        st.info("No customers currently show elevated rising-risk conditions in the available demo dataset.")
    else:
        st.dataframe(rising_customers.sort_values("avg_probability", ascending=False), use_container_width=True)


def render_model_center(artifacts: Dict[str, Any]) -> None:
    render_section_header("Model Center", "Model inventory, outputs, and saved artifact context.")

    reports = artifacts.get("reports", {})
    model_compare_path = REPORTS / "model_comparison.csv"
    if model_compare_path.exists():
        comparison_df = pd.read_csv(model_compare_path)
        st.caption("Stored validation metrics from the project reports directory.")
        st.dataframe(comparison_df, use_container_width=True)

    if "test_metrics.json" in reports:
        st.subheader("Classifier metrics")
        st.json(reports["test_metrics.json"])

    if "regression_metrics.json" in reports:
        st.subheader("Delay-regression metrics")
        st.json(reports["regression_metrics.json"])

    if "nlp_metrics.json" in reports:
        st.subheader("NLP model metrics")
        st.json(reports["nlp_metrics.json"])


def render_artifact_diagnostics(artifacts: Dict[str, Any]) -> None:
    render_section_header("Artifact Diagnostics", "Technical system health, loaded artifacts, and configuration status.")
    models = artifacts.get("models", {})
    load_errors = artifacts.get("load_errors", {})

    artifact_names = [
        ("Classifier", "classifier"),
        ("Delay regressor", "delay_regressor"),
        ("Logistic Regression", "logistic_regression"),
        ("Random Forest", "random_forest"),
        ("MLP Neural Network", "mlp_neural_network"),
        ("NLP model", "nlp_payment_risk"),
    ]

    cols = st.columns(3)
    for index, (label, name) in enumerate(artifact_names):
        status = "PASS" if name in models else "FAIL"
        with cols[index % 3]:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{status}</div>
                    <div class="kpi-sub">{load_errors.get(name, 'Loaded successfully') if status == 'FAIL' else 'Available for inference'}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_batch_predictions(artifacts: Dict[str, Any]) -> None:
    render_section_header("Batch Analysis", "Upload, analyze, and review invoice batches with the existing prediction pipeline.")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is None:
        st.caption("Expected columns: invoice_date, due_date, invoice_amount, customer_seen_before, prior_late_count, prior_late_ratio, prior_avg_delay, industry, company_size, payment_method, customer_segment, outstanding_amount")
        return

    try:
        batch_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read uploaded CSV: {exc}")
        return

    required_cols = {
        "invoice_date",
        "due_date",
        "invoice_amount",
        "customer_seen_before",
        "prior_late_count",
        "prior_late_ratio",
        "prior_avg_delay",
        "industry",
        "company_size",
        "payment_method",
        "customer_segment",
        "outstanding_amount",
    }

    missing = sorted(required_cols.difference(batch_df.columns))
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return

    rows = []
    for index, row in batch_df.iterrows():
        payload = row.to_dict()
        errors = validate_single_input(payload)
        if errors:
            rows.append({"row_index": index, "status": "invalid", "errors": "; ".join(errors)})
            continue

        model_choice = "classifier"
        try:
            prediction = run_prediction(payload, model_choice, artifacts)
            rows.append(
                {
                    "row_index": index,
                    "status": "ok",
                    "late_probability": prediction["late_probability"],
                    "expected_delay_days": prediction["expected_delay_days"],
                    "risk_level": prediction["risk_level"],
                    "estimated_financial_exposure": prediction["estimated_financial_exposure"],
                    "recommendation": prediction["recommendation"],
                }
            )
        except Exception as exc:
            rows.append({"row_index": index, "status": "failed", "errors": str(exc)})

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        st.warning("No rows were processed from the uploaded CSV.")
        return

    ok_rows = result_df[result_df["status"] == "ok"].copy()
    if not ok_rows.empty:
        summary_values = {
            "Total invoices": len(result_df),
            "High-risk invoices": int((ok_rows["risk_level"].isin(["HIGH", "CRITICAL"])).sum()),
            "Medium-risk invoices": int((ok_rows["risk_level"] == "MEDIUM").sum()),
            "Low-risk invoices": int((ok_rows["risk_level"] == "LOW").sum()),
            "Total amount": float(batch_df.get("invoice_amount", pd.Series([0] * len(batch_df))).sum()),
            "Amount at risk": float(ok_rows["estimated_financial_exposure"].sum()),
        }

        st.markdown("### Batch summary")
        cols = st.columns(6)
        for idx, (label, value) in enumerate(summary_values.items()):
            with cols[idx]:
                st.metric(label, value if isinstance(value, str) else f"{value:,.2f}" if label in {"Total amount", "Amount at risk"} else value)

    st.dataframe(result_df, use_container_width=True)

    if not result_df.empty:
        csv_buffer = result_df.to_csv(index=False)
        st.download_button(
            label="Download batch predictions",
            data=csv_buffer,
            file_name="invoiceguard_batch_predictions.csv",
            mime="text/csv",
        )


def render_nlp_section(artifacts: Dict[str, Any]) -> None:
    render_section_header("NLP Payment-Risk Intelligence", "Analyze customer communication for payment-risk signals using the existing NLP model.")

    nlp_model = artifacts["models"].get("nlp_payment_risk")
    if nlp_model is None:
        st.warning("NLP model unavailable in this deployment.")
        return

    text_examples = load_text_examples()
    sample_text = st.selectbox("Example payment message", options=text_examples)
    customer_message = st.text_area("Customer payment-related message", value=sample_text, height=150)

    if st.button("Analyze message") and customer_message.strip():
        try:
            thresholds = load_nlp_thresholds()
            analysis = score_nlp_message(nlp_model, customer_message, thresholds)
        except Exception as exc:
            st.error(f"Unable to analyze payment message: {exc}")
            return

        col_a, col_b = st.columns(2)
        col_a.metric("NLP payment-risk probability", f"{analysis['probability']:.2%}")
        col_b.metric("NLP risk level", analysis["risk_label"])

        st.caption(
            f"NLP thresholds: HIGH >= {thresholds['high']:.2f}, MEDIUM >= {thresholds['medium']:.2f}."
        )

        st.json(analysis["probabilities"])
        st.info("The NLP model is an independent signal. It does not automatically combine with the invoice classifier in this deployment.")

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
            baseline_prediction = run_prediction(baseline_payload, "classifier", artifacts)
            st.metric("Baseline invoice risk", f"{baseline_prediction['late_probability']:.2%}")
        except Exception:
            st.warning("Baseline invoice risk could not be generated for this deployment.")


def render_app_nav() -> str:
    sections = [
        "Overview → Executive Dashboard",
        "Risk & Predictions → Single Invoice Prediction",
        "Risk & Predictions → Cold Start Mode",
        "Risk & Predictions → Risk Center",
        "Risk & Predictions → Customer 360",
        "Intelligence → Financial Impact",
        "Intelligence → AI Explanation",
        "Intelligence → NLP Payment-Risk Intelligence",
        "Operations → Batch Analysis",
        "Operations → Alert Center",
        "System → Model Center",
        "System → Artifact Diagnostics",
    ]

    selected = st.sidebar.selectbox("Navigation", options=sections, index=0)
    return selected.split(" → ", 1)[1]


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

    invoice_amount = float(payload.get("invoice_amount", 0) or 0)
    if invoice_amount <= 0:
        errors.append("invoice_amount must be greater than zero.")

    outstanding_amount = float(payload.get("outstanding_amount", 0) or 0)
    if outstanding_amount < 0:
        errors.append("outstanding_amount cannot be negative.")

    customer_seen_before = int(payload.get("customer_seen_before", 0) or 0)
    if customer_seen_before < 0:
        errors.append("customer_seen_before cannot be negative.")

    for field in ["prior_late_count", "prior_avg_delay"]:
        value = float(payload.get(field, 0) or 0)
        if value < 0:
            errors.append(f"{field} cannot be negative.")

    prior_late_ratio = float(payload.get("prior_late_ratio", 0) or 0)
    if prior_late_ratio < 0 or prior_late_ratio > 1:
        errors.append("prior_late_ratio must be between 0 and 1.")

    for field in ["industry", "company_size", "payment_method", "customer_segment"]:
        val = payload.get(field)
        if val is not None and (isinstance(val, str) and not val.strip()):
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
        value = payload.get(col)
        row[col] = value if value not in (None, "") else np.nan

    feature_order = metadata.get("features") or [
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

    ordered_row = {col: row.get(col, np.nan) for col in feature_order}
    return ordered_row


def run_prediction(payload: Dict[str, Any], selected_model: str, artifacts: Dict[str, Any]) -> Dict[str, Any]:
    metadata = artifacts["metadata"]
    models = artifacts["models"]

    if "classifier" not in models and selected_model == "classifier":
        raise RuntimeError("No trained classifier was found in /models. Run python -m src.train --source demo first.")

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

    risk_level = "LOW"
    if probability < 0.35:
        risk_level = "LOW"
    elif probability < 0.65:
        risk_level = "MEDIUM"
    elif probability < 0.85:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"

    exposure = float((payload.get("invoice_amount", 0) or 0) * probability)

    if risk_level in ["HIGH", "CRITICAL"]:
        recommendation = "Prioritize collection follow-up before the due date."
    else:
        recommendation = "Monitor payment and schedule routine follow-up."

    return {
        "late_probability": probability,
        "risk_level": risk_level,
        "expected_delay_days": delay_days,
        "estimated_financial_exposure": exposure,
        "recommendation": recommendation,
        "feature_frame": feature_frame,
    }


def format_top_features(model: Any, feature_frame: pd.DataFrame, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
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
            top = []
            for idx in top_indices:
                top.append(
                    {
                        "feature": transformed_names[idx],
                        "impact": float(np.abs(values[idx])),
                        "direction": "increases risk" if values[idx] >= 0 else "reduces risk",
                    }
                )
            return top

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
                "feature": names[idx],
                "impact": float(np.abs(values[idx])),
                "direction": "increases risk" if values[idx] >= 0 else "reduces risk",
            }
            for idx in top_indices
        ]
    except Exception:
        return []


def generate_fallback_top_features(model: Any, feature_frame: pd.DataFrame) -> List[Dict[str, Any]]:
    if hasattr(model, "feature_importances_"):
        estimator = model.feature_importances_
        names = list(feature_frame.columns)
        scores = np.asarray(estimator, dtype=float)
        top_indices = np.argsort(scores)[::-1][:5]
        return [
            {
                "feature": names[idx],
                "impact": float(scores[idx]),
                "direction": "increases risk" if idx < len(names) else "feature importance",
            }
            for idx in top_indices
        ]

    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("model")
        if estimator is not None and hasattr(estimator, "coef_"):
            coef = np.asarray(estimator.coef_).reshape(-1)
            names = feature_frame.columns
            top_indices = np.argsort(np.abs(coef))[::-1][:5]
            return [
                {
                    "feature": names[idx],
                    "impact": float(np.abs(coef[idx])),
                    "direction": "increases risk" if coef[idx] >= 0 else "reduces risk",
                }
                for idx in top_indices
            ]

    return []


def explain_prediction(model: Any, feature_frame: pd.DataFrame, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    top = format_top_features(model, feature_frame, metadata)
    if top:
        return top

    fallback = generate_fallback_top_features(model, feature_frame)
    if fallback:
        return fallback

    return []


def render_single_prediction(artifacts: Dict[str, Any]) -> None:
    metadata = artifacts["metadata"]
    model_options = ["classifier"] + [name for name in ["logistic_regression", "random_forest", "mlp_neural_network"] if name in artifacts["models"]]
    if "classifier" not in model_options:
        model_options = [name for name in artifacts["models"] if name != "delay_regressor"]

    loaded_models = artifacts["models"]
    demo_options = load_demo_options()

    st.markdown(
        """
        <div class="section-shell">
            <div class="section-title">Single Invoice Prediction</div>
            <div class="section-caption">Professional AI assessment workflow for one invoice. The prediction pipeline remains unchanged and uses the existing saved artifacts.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("single_prediction_form"):
        st.markdown("### Invoice Information")
        col1, col2 = st.columns(2)
        with col1:
            invoice_date = st.date_input("Invoice date", value=pd.Timestamp("2026-09-01").date())
            due_date = st.date_input("Due date", value=pd.Timestamp("2026-10-01").date())
            invoice_amount = st.number_input("Invoice amount", min_value=1.0, value=50000.0, step=1000.0)
            outstanding_amount = st.number_input("Outstanding amount", min_value=0.0, value=80000.0, step=1000.0)
            model_choice = st.selectbox("Model for main prediction", options=model_options, index=0)

        with col2:
            existing_customer = st.checkbox("Existing customer (has prior invoice history)", value=True)
            customer_seen_before = st.number_input("Customer seen before (prior invoices)", min_value=0, value=6, step=1)
            prior_late_count = st.number_input("Prior late count", min_value=0.0, value=3.0, step=1.0)
            prior_late_ratio = st.number_input("Prior late ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
            prior_avg_delay = st.number_input("Prior average delay (days)", min_value=0.0, value=9.0, step=0.5)

        st.markdown("### Customer profile")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            industry = st.selectbox("Industry", options=demo_options.get("industry", DEFAULT_CATEGORICALS["industry"]), index=0)
        with c2:
            company_size = st.selectbox("Company size", options=demo_options.get("company_size", DEFAULT_CATEGORICALS["company_size"]), index=1)
        with c3:
            payment_method = st.selectbox("Payment method", options=demo_options.get("payment_method", DEFAULT_CATEGORICALS["payment_method"]), index=0)
        with c4:
            customer_segment = st.selectbox("Customer segment", options=demo_options.get("customer_segment", DEFAULT_CATEGORICALS["customer_segment"]), index=0)

        if not existing_customer:
            st.info("Cold-start mode: historical customer fields will be set to zero because this customer has no prior payment history.")
            customer_seen_before = 0
            prior_late_count = 0.0
            prior_late_ratio = 0.0
            prior_avg_delay = 0.0

        submitted = st.form_submit_button("Run AI Assessment")

    if not submitted:
        return

    payload = {
        "invoice_date": str(invoice_date),
        "due_date": str(due_date),
        "invoice_amount": invoice_amount,
        "customer_seen_before": customer_seen_before,
        "prior_late_count": prior_late_count,
        "prior_late_ratio": prior_late_ratio,
        "prior_avg_delay": prior_avg_delay,
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

    try:
        result = run_prediction(payload, model_choice, artifacts)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    risk_label = result["risk_level"]
    risk_badge = risk_label.lower().replace(" ", "-")
    st.markdown(
        f"""
        <div class="result-card">
            <h3>Payment Risk</h3>
            <div class="score">{risk_label}</div>
            <div class="badge badge-{risk_badge}">{risk_label}</div>
            <div class="score-grid" style="margin-top: 1rem;">
                <div class="score-item">
                    <div class="score-item-label">Probability</div>
                    <div class="score-item-value">{result['late_probability']:.2%}</div>
                </div>
                <div class="score-item">
                    <div class="score-item-label">Expected Delay</div>
                    <div class="score-item-value">{result['expected_delay_days']:.1f} days</div>
                </div>
                <div class="score-item">
                    <div class="score-item-label">Revenue at Risk</div>
                    <div class="score-item-value">${result['estimated_financial_exposure']:.2f}</div>
                </div>
                <div class="score-item">
                    <div class="score-item-label">Recommended Action</div>
                    <div class="score-item-value">{result['recommendation']}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### AI Assessment")
    explanation = explain_prediction(loaded_models.get(model_choice), result["feature_frame"], metadata)
    if explanation:
        for item in explanation:
            st.write(f"- {item['feature']}: {item['direction']} (impact={item['impact']:.4f})")
    else:
        st.warning("SHAP explainability is not available for the selected model, so this view shows no factor breakdown. The safest available explainability method has been used where possible.")

    st.markdown("### Recommended Business Action")
    st.info(result["recommendation"])

    if "test_metrics" in artifacts["reports"]:
        st.markdown("### Current Artifact Metrics")
        st.json(artifacts["reports"]["test_metrics.json"])


def main() -> None:
    artifacts = load_project_artifacts(get_artifact_signature())
    metadata = artifacts.get("metadata", {})
    models = artifacts.get("models", {})
    load_errors = artifacts.get("load_errors", {})

    inject_fintech_css()
    render_app_header(metadata, models, load_errors)

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-shell">
                <div class="sidebar-brand">InvoiceGuard AI</div>
                <div class="sidebar-subtitle">Runtime Status</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="sidebar-status"><span>Best model</span><strong>{metadata.get('best_model', 'unknown')}</strong></div>
            <div class="sidebar-status"><span>Models loaded</span><strong>{len(models)}</strong></div>
            <div class="sidebar-status"><span>Artifacts</span><strong>{'Healthy' if not load_errors else 'Warnings'}</strong></div>
            """,
            unsafe_allow_html=True,
        )
        st.success("Artifacts loaded from /models and /reports")

        if metadata.get("test_metrics"):
            st.subheader("Saved test metrics")
            st.json(metadata.get("test_metrics", {}))

        st.markdown("---")
        render_artifact_diagnostics(artifacts)

        if load_errors:
            st.warning("Some optional artifacts were skipped because they were not usable in this runtime.")

    required_artifacts = ["classifier", "delay_regressor"]
    failed_required = [name for name in required_artifacts if name not in models]
    if failed_required:
        st.error("Fatal startup error: required artifacts failed to load.")
        for name in failed_required:
            st.error(f"Required artifact '{name}' failed to load: {load_errors.get(name, 'missing from /models')}")
        st.stop()

    section = render_app_nav()

    section_map = {
        "Executive Dashboard": render_exec_dashboard,
        "Single Invoice Prediction": render_single_prediction,
        "Cold Start Mode": render_single_prediction,
        "Risk Center": render_risk_center,
        "Customer 360": render_customer_360,
        "Financial Impact": render_financial_impact,
        "AI Explanation": render_ai_explanation,
        "NLP Payment-Risk Intelligence": render_nlp_section,
        "Batch Analysis": render_batch_predictions,
        "Alert Center": render_alert_center,
        "Model Center": render_model_center,
        "Artifact Diagnostics": render_artifact_diagnostics,
    }

    section_map.get(section, render_exec_dashboard)(artifacts)


if __name__ == "__main__":
    main()
