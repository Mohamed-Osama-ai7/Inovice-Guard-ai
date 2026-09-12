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

st.set_page_config(page_title="InvoiceGuard AI", page_icon="💳", layout="wide")


@st.cache_resource
def load_project_artifacts() -> Dict[str, Any]:
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
    for name, path in model_files.items():
        if path.exists():
            loaded_models[name] = joblib.load(path)

    reports = {}
    for report_name in ["test_metrics.json", "regression_metrics.json", "all_model_artifacts.json", "nlp_metrics.json"]:
        path = REPORTS / report_name
        if path.exists():
            reports[report_name] = json.loads(path.read_text(encoding="utf-8"))

    return {
        "metadata": metadata,
        "models": loaded_models,
        "reports": reports,
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

    st.subheader("Single invoice prediction")
    st.caption("This dashboard never uses payment_date or post-outcome fields as predictors.")

    with st.form("single_prediction_form"):
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

        st.write("### Customer profile")
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

        submitted = st.form_submit_button("Run prediction")

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

    st.success("Prediction generated successfully.")
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    metrics_col1.metric("Late probability", f"{result['late_probability']:.2%}")
    metrics_col2.metric("Expected delay", f"{result['expected_delay_days']:.1f} days")
    metrics_col3.metric("Risk level", result["risk_level"])
    metrics_col4.metric("Financial exposure", f"${result['estimated_financial_exposure']:.2f}")

    st.subheader("Recommended business action")
    st.info(result["recommendation"])

    model_name = metadata.get("best_model", "classifier")
    if "test_metrics" in artifacts["reports"]:
        st.subheader("Current artifact metrics")
        st.json(artifacts["reports"]["test_metrics.json"])

    st.subheader("Most important factors")
    explanation = explain_prediction(loaded_models.get(model_choice), result["feature_frame"], metadata)
    if explanation:
        for item in explanation:
            st.write(f"- {item['feature']}: {item['direction']} (impact={item['impact']:.4f})")
    else:
        st.warning("SHAP explainability is not available for the selected model, so this view shows no factor breakdown. The safest available explainability method has been used where possible.")


def render_batch_predictions(artifacts: Dict[str, Any]) -> None:
    st.subheader("CSV batch prediction")
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
            rows.append(
                {
                    "row_index": index,
                    "status": "invalid",
                    "errors": "; ".join(errors),
                }
            )
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
            rows.append({
                "row_index": index,
                "status": "failed",
                "errors": str(exc),
            })

    result_df = pd.DataFrame(rows)
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
    st.subheader("NLP payment-risk analysis")
    nlp_model = artifacts["models"].get("nlp_payment_risk")
    if nlp_model is None:
        st.warning("The NLP demo model was not found in /models, so the NLP analysis section is unavailable.")
        return

    text_examples = load_text_examples()
    sample_text = st.selectbox("Example payment message", options=text_examples)
    customer_message = st.text_area("Customer payment-related message", value=sample_text, height=150)

    if st.button("Analyze message") and customer_message.strip():
        probability = float(nlp_model.predict_proba([customer_message])[0, 1])
        risk_label = "HIGH RISK" if probability >= 0.5 else "LOW RISK"
        st.metric("NLP payment-risk probability", f"{probability:.2%}")
        st.info(f"Detected {risk_label} based on the message content.")


def main() -> None:
    artifacts = load_project_artifacts()
    metadata = artifacts.get("metadata", {})
    models = artifacts.get("models", {})

    st.title("InvoiceGuard AI")
    st.caption("Invoice payment risk prediction and cash-flow protection dashboard")

    with st.sidebar:
        st.header("Project status")
        st.success("Artifacts loaded from /models and /reports")
        st.write(f"Best model: {metadata.get('best_model', 'unknown')}")
        st.write(f"Classifier loaded: {'yes' if 'classifier' in models else 'no'}")
        st.write(f"Delay regressor loaded: {'yes' if 'delay_regressor' in models else 'no'}")
        st.write(f"NLP model loaded: {'yes' if 'nlp_payment_risk' in models else 'no'}")

        if metadata.get("test_metrics"):
            st.subheader("Saved test metrics")
            st.json(metadata.get("test_metrics", {}))

    if not models.get("classifier"):
        st.error("No trained model artifacts were found in /models. Run python -m src.train --source demo first.")
        st.stop()

    render_single_prediction(artifacts)
    st.markdown("---")
    render_batch_predictions(artifacts)
    st.markdown("---")
    render_nlp_section(artifacts)


if __name__ == "__main__":
    main()
