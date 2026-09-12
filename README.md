# InvoiceGuard AI

**Intelligent Invoice Payment Risk Prediction & Cash-Flow Protection System**

InvoiceGuard AI predicts whether an invoice is likely to be paid late, estimates expected delay, calculates financial exposure, explains risk drivers, and recommends a business action.

## Why this project is technically sound

- Classification: Logistic Regression, Random Forest, XGBoost, MLP Neural Network
- Regression: Gradient-boosted regressor for positive delay days
- Cold start: historical customer features are created only from earlier invoices using `shift(1)`; first-ever invoices have no personal history
- Leakage prevention: `payment_date`, target-derived delay fields, and post-payment information are excluded from X
- Time-aware evaluation: chronological 70/15/15 train/validation/test split
- Explainability-ready: feature structure and API explanations; SHAP can be layered in for the final presentation
- Deployment: FastAPI + Docker + Render configuration

## 1. Local setup

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Run with the official Kaggle dataset

```bash
python scripts/download_data.py
python -m src.train --source kaggle
```

The training script writes:
- `reports/data_audit.json`
- `reports/model_comparison.csv`
- `reports/test_metrics.json`
- `reports/regression_metrics.json`
- `reports/test_predictions.csv`
- `models/classifier.joblib`
- `models/delay_regressor.joblib` (when enough delayed invoices exist)
- `models/metadata.json`

## 3. Offline smoke test

Useful when Kaggle is unavailable from the current machine:

```bash
python scripts/generate_demo.py
python -m src.train --source demo
```

The demo is synthetic and exists only to verify that the complete pipeline works. **Do not present demo metrics as Kaggle results.**

## 4. Start the API

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`.

## 5. API example

POST `/predict` with JSON:

```json
{
  "invoice_date":"2026-09-01",
  "due_date":"2026-10-01",
  "invoice_amount":50000,
  "customer_seen_before":6,
  "prior_late_count":3,
  "prior_late_ratio":0.5,
  "prior_avg_delay":9,
  "industry":"Construction",
  "company_size":"Medium",
  "payment_method":"Bank Transfer",
  "customer_segment":"SME",
  "outstanding_amount":80000
}
```

## 6. Deployment

### Docker

```bash
docker build -t invoiceguard-ai .
docker run -p 8000:8000 invoiceguard-ai
```

### Render

The repository includes `render.yaml` and a Dockerfile. Push the folder to GitHub, create a Render Web Service from the repo, and deploy with Docker.

**Important:** the deployed service must have the trained model artifacts under `models/`. For a real deployment, train using the Kaggle dataset first and commit only artifacts that you are allowed to redistribute, or mount them through a release/storage workflow.

## Accuracy policy

The project **does not hard-code or fake a >90% accuracy claim**. The real Kaggle test accuracy is computed after the dataset is downloaded and audited. The training script prints a PASS only when the untouched chronological test set actually reaches 90% or higher.

This is deliberate: using `payment_date` as an input feature or tuning only on the test set would create leakage and make the reported number invalid.
