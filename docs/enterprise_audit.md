# Enterprise System Audit: InvoiceGuard AI

> **Historical Baseline Note:** This document captures the system architecture and gap analysis recorded at an early project milestone. Many items listed as gaps (calibration, security, file handling, responsive UI) have since been addressed. Refer to `README.md`, `docs/model_card.md`, and `docs/model_quality_report.md` for the current verified state of the system.


## 1. ARCHITECTURE
The current architecture is a hybrid monolithic Streamlit application with a FastAPI backend (`src/api.py`) for programmatic access. 
- **Frontend**: Streamlit-based UI recently refactored into a `src/ui/` package with views (`overview.py`, `customer_360.py`, etc.) and a master router.
- **Backend/Inference**: Models are loaded at startup via `joblib`. Predictions are generated synchronously.
- **Data Engineering**: Data pipelines for the Invoice domain (`src/pipeline.py`) and Retail domain (`src/retail_pipeline.py`) run as offline batch scripts.
- **Explainability**: SHAP is used for tree-based models, falling back to coefficient magnitudes or feature importances for others.

## 2. DATA
- **InvoiceGuard Data**: Uses a synthetic/demo CSV (`demo_invoices.csv`) for invoice payment data. It lacks true scale and temporal realism.
- **Retail Data**: Built to ingest the real UCI Online Retail II dataset (`online_retail_II.xlsx`). Contains real transactions from 2009–2011, over 1M records.
- **Quality**: Basic validation exists in `validate_single_input`, but there is no enterprise-grade schema validation, data quality reporting, or robust file-upload handling layer for end-users.

## 3. MODELS
The system currently maintains multiple models across two distinct domains:
**Invoice Payment Risk**
- Primary: XGBoost/Random Forest (`classifier.joblib`), Logistic Regression, MLP Neural Network.
- NLP Risk: TF-IDF + Logistic Regression (`nlp_payment_risk.joblib`) for payment communications.
- Delay Regressor: Gradient-boosted regressor for expected delay days.

**Retail & Customer Intelligence**
- Repurchase Risk: Logistic Regression/HistGradientBoosting (`retail_repurchase_model.pkl`).
- Future Revenue: HistGradientBoosting Regressor (`retail_future_revenue_model.pkl`).

## 4. FEATURES
- **Invoice Features**: Log-transformed amounts, temporal components (year, month, quarter), historical customer behavior (prior late count, ratio, avg delay). Uses a `shift(1)` approach to prevent leakage.
- **Retail Features**: Point-in-time RFM (Recency, Frequency, Monetary), cancellation rates, customer age, and short-term behavior (90-day revenue).
- **Leakage Risk**: The pipelines correctly use temporal splits (e.g., `build_temporal_dataset`), but the strict adherence across all features needs a formal audit.

## 5. TRAINING
- **Retail**: Uses a proper temporal split (Train -> Validation -> Out-Of-Time Test) in `src/retail_train.py`.
- **Invoice**: Uses a chronological 70/15/15 split.
- **Model Selection**: Currently hardcoded scripts (`scripts/build_all_models.py`, `scripts/build_retail_models.py`). Benchmarking against baselines is not formalized into reports.

## 6. EVALUATION
- **Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC for classification. MAE, RMSE for regression.
- **Imbalance**: Uses `class_weight='balanced'` in some models, but threshold calibration and Brier scores are missing.
- **Overfitting**: Basic train/test comparison exists, but a formal overfitting audit and generalization gap report is missing.

## 7. DEPLOYMENT
- **Streamlit Cloud**: Deployed at `https://inovice-guard-ai-17.streamlit.app/`. Uses `requirements.txt`.
- **Docker/Render**: Configured for the FastAPI backend (`Dockerfile`, `render.yaml`).
- **State**: Currently live and functioning, but needs to be rigorously protected during this massive upgrade.

## 8. SECURITY
- **Secrets**: Uses `os.getenv("GROQ_API_KEY")`.
- **File Uploads**: Currently lacking a robust, secure file-handling layer (e.g., path traversal protection, size limits, extension validation).
- **Data Privacy**: No obvious PII leakage, but real customer names in the UCI dataset must be handled securely.

## 9. UX
- **Current State**: Transitioning from a basic dashboard to an enterprise layout.
- **Gaps**: Lacks empty states, error boundaries, proper loading skeletons, an "Action Center", global search, and a seamless "Run Prediction" workflow for non-technical users.

## 10. KNOWN RISKS
- **Monolithic Bottlenecks**: Heavy datasets loaded in Streamlit can cause Out-Of-Memory (OOM) errors or slow reruns.
- **Data Quality**: User-uploaded CSVs without strict schema validation will crash the prediction pipelines.
- **Calibration**: Probabilities are used for "Exposure" calculations, but they are not strictly calibrated (e.g., Isotonic/Platt scaling).
- **Over-promising**: Must ensure the UI clearly indicates confidence levels and doesn't present heuristic risk as absolute certainty.
