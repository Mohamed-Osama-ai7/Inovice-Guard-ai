# InvoiceGuard AI
**Enterprise Receivables & Customer Intelligence Platform**

InvoiceGuard AI is an AI-powered receivables and customer intelligence platform that combines payment-risk prediction, revenue analytics, customer-level risk monitoring, explainable machine learning, and optional NLP-based payment-risk analysis.

---

## Live Product & Links

- **Live Application:** [https://inovice-guard-ai-17.streamlit.app/](https://inovice-guard-ai-17.streamlit.app/)
- **GitHub Repository:** [https://github.com/Mohamed-Osama-ai7/Inovice-Guard-ai](https://github.com/Mohamed-Osama-ai7/Inovice-Guard-ai)

*Note on Live Access: The application is deployed on Streamlit Community Cloud. Access permissions are managed via Streamlit workspace settings.*

---

## Product Overview

InvoiceGuard AI addresses critical financial operations challenges faced by B2B organizations:
- **Identifying Late Payment Risk:** Predicting which invoices are likely to become past due before the payment due date.
- **Quantifying Financial Exposure:** Calculating estimated dollar exposure at risk across open receivables.
- **Prioritizing Collections:** Enabling collections teams to focus outreach on high-exposure and high-risk accounts.
- **Customer 360 Intelligence:** Unifying payment history, behavioral RFM metrics, churn risk, and revenue forecasts per customer.
- **Explainability & Transparency:** Providing feature attribution scores via SHAP so users understand prediction drivers.

---

## Core Capabilities

- **Invoice Payment-Risk Prediction:** Binary classification scoring late payment probability ($P(\text{risk}) \in [0, 1]$).
- **Expected Payment Delay Estimation:** Regression modeling of anticipated delay days for past-due invoices.
- **Financial Exposure Estimation:** Dollar-weighted exposure calculation combining invoice amounts and late probabilities.
- **Customer 360 Analysis:** CRM-style unified view merging B2B invoice histories with retail behavioral metrics.
- **Customer Search:** Sub-10ms customer lookup and search across all intelligence domains.
- **Revenue Intelligence:** 60-day customer future revenue forecasting and repurchase churn risk classification.
- **Risk & Action Center:** Prioritized queue of high-risk accounts with recommended operational follow-ups.
- **Explainable AI:** Feature attribution breakdown showing top positive and negative risk factors per prediction.
- **NLP Payment-Risk Analysis:** Optional text classification of customer emails and payment communications.
- **Batch Processing & Export:** Vectorized batch inference supporting CSV and XLSX file uploads up to 50 MB with exportable results.
- **Data Quality & Validation:** Schema enforcement, missing value handling, and data profiling diagnostics.
- **Responsive Interface:** Full design system support across Desktop, Laptop, Tablet, and Mobile viewports.

---

## System Architecture

```text
User Input / CSV Batch / Customer Search
       │
       ▼
Data Contract & Input Validation (Pydantic / Pandas Schema)
       │
       ▼
Feature Engineering Pipeline (Point-in-Time Lag Aggregations & Preprocessing)
       │
       ▼
Machine Learning Inference Engine
  ├── Invoice Risk Classifier (Calibrated Logistic Regression)
  ├── Expected Delay Regressor (Regression Pipeline)
  ├── Retail Repurchase Risk Model (Calibrated Classifier)
  ├── Retail Revenue Regressor (HistGradientBoosting)
  └── NLP Payment-Risk Model (TF-IDF + Calibrated Logistic Regression)
       │
       ▼
Risk Probabilities, Expected Delays & Financial Exposure Calculations
       │
       ▼
Explainability Engine (SHAP Feature Attribution)
       │
       ▼
Responsive Enterprise SaaS UI (Streamlit View Router & Custom CSS System)
```

### NLP Pipeline Flow

```text
Customer Communication Text
       │
       ▼
Text Preprocessing & TF-IDF Vectorization (1-2 N-grams)
       │
       ▼
Binary Payment-Risk Classifier -> Probability P(risk) in [0, 1]
       │
       ▼
Business Alert Threshold Mapping:
  ├── LOW RISK    : P(risk) < 0.30
  ├── MEDIUM RISK : 0.30 <= P(risk) < 0.50
  └── HIGH RISK   : P(risk) >= 0.50
```

---

## Machine Learning & Evaluation

### Tabular Invoice Risk Model
- **Algorithm:** Calibrated Logistic Regression
- **Target Variable:** `late_payment` (Binary: 0 = On Time, 1 = Late)
- **Data Splitting:** Strict chronological split (60% Train, 20% Validation, 10% Test, 10% Out-Of-Time).
- **Leakage Prevention:** Post-payment fields (`payment_date`, `delay_days`) are excluded from predictor matrix $X$. Historical customer features (`prior_late_count`, `prior_late_ratio`, `prior_avg_delay`) use point-in-time expanding window aggregation (`cumsum().shift(1)`).
- **Probability Calibration:** Isotonic regression calibration fitted exclusively on the independent Validation split via `CalibratedClassifierCV(cv="prefit")`.
- **Audited Metrics:**
  - **Test Set:** PR-AUC: 0.9751 | ROC-AUC: 0.9893 | Brier Score: 0.0372
  - **Out-Of-Time (OOT) Set:** PR-AUC: 0.9700 | ROC-AUC: 0.9878 | Brier Score: 0.0363
- **Generalization Assessment:** Strong temporal generalization on the evaluated OOT split.

### NLP Payment-Risk Model
- **Algorithm:** TF-IDF (1-2 N-grams) + Calibrated Logistic Regression
- **Contract:** Binary classification (Class 0 = Low/No Payment Risk, Class 1 = High Payment Risk).
- **Threshold Mapping:** The UI layer maps predicted probability $P(\text{risk})$ to business alert bands: LOW ($< 0.30$), MEDIUM ($0.30 - 0.49$), and HIGH ($\ge 0.50$). **MEDIUM is a business alert band, not a third learned class.**
- **Evaluation Redesign:** Evaluated using a **group-disjoint template split** (0% template overlap between Train, Validation, and Test splits) and an **independent held-out real-world out-of-domain (OOD) evaluation corpus**.
- **Audited NLP Metrics:**
  - **Validation (Unseen Templates):** Precision: 0.5250 | Recall: 1.0000 | F1: 0.6885 | Macro F1: 0.5875 | PR-AUC: 0.7050 | ROC-AUC: 0.8401 | Brier: 0.2227
  - **Test (Unseen Templates):** Precision: 0.4375 | Recall: 1.0000 | F1: 0.6087 | Macro F1: 0.4904 | PR-AUC: 0.6829 | ROC-AUC: 0.8259 | Brier: 0.2742
  - **Held-Out Real-World OOD Corpus:** Precision: 0.4286 | Recall: 0.9000 | F1: 0.5806 | Macro F1: 0.5662 | PR-AUC: 0.8118 | ROC-AUC: 0.8800 | Brier: 0.2318
- **Evaluation Note:** The OOD evaluation provides evidence of generalization beyond training templates, but it remains a held-out synthetic/OOD evaluation and should not be interpreted as broad real-world validation.

---

## Dataset Source & Attribution

The customer analytics and revenue intelligence modules use the official **UCI Online Retail II** dataset:
- **Official Source:** [https://archive.ics.uci.edu/dataset/502/online+retail+ii](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
- **DOI:** [https://doi.org/10.24432/C5CG6D](https://doi.org/10.24432/C5CG6D)
- **Citation:** Chen, D. (2012). *Online Retail II*. UCI Machine Learning Repository.
- **Dataset Specification:** Contains 1,067,371 transaction instances covering all transactions occurring between 01/12/2009 and 09/12/2011 for a UK-based online retailer.
- **Licensing:** Licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).

*Data Acquisition:* Raw datasets are downloaded locally via repository-relative setup scripts (`python scripts/download_retail_data.py` and `python scripts/download_data.py`).

---

## Measured Benchmark Performance

All core operations have been benchmarked in the development/test environment:

| Operation | Benchmark Timing | Optimization Description |
|---|---|---|
| **Overview Dataset (First Uncached Run)** | **0.313 seconds** | Vectorized batch inference (down from ~345 seconds) |
| **Overview Dataset (Cached Run)** | **0.012 seconds** | Fast Streamlit session rendering |
| **Customer 360 Lookup** | **2.49 ms** | O(1) indexed customer profile retrieval |
| **Customer Search** | **6.99 ms** | Sub-10ms substring matching |
| **Single Invoice Prediction** | **19.66 ms** | Real-time classification & SHAP scoring |
| **Batch Prediction (12,000 rows)** | **34.30 ms** | Fully vectorized batch inference pipeline |
| **NLP Single Message Prediction** | **3.64 ms** | Real-time TF-IDF vectorization and scoring |

*Note: Benchmark measurements reflect server-side execution timing in the test environment and do not include browser rendering or network transfer latency.*

---

## Responsive Interface

The application UI supports Desktop, Laptop, Tablet, and Mobile devices:
- **Desktop (>= 1200px):** Multi-column KPI rows, full sidebar navigation, side-by-side analytical cards, interactive Plotly charts, and expanded dataframes.
- **Tablet (768px - 1199px):** Adaptive 2x2 grid layouts, auto-scaling charts, and touch-friendly controls.
- **Mobile (< 768px):** Collapsed single-column vertical card stacks, fluid typography (`clamp()`), horizontal table scrolling, touch targets with 44px minimum height, and zero page-wide horizontal overflow.

*Validation Note: Automated browser/device interaction was not available in the validation environment; responsive behavior was audited through CSS/layout inspection and application-level verification.*

---

## File Handling, Safety & Security

- **File Upload Limits:** Enforces a maximum file size of **50 MB** and a maximum dataset length of **1,000,000 rows**.
- **Supported Formats:** Validated CSV and XLSX files.
- **Input Validation:** Rejects unsupported extensions (.exe, .sh, etc.), corrupted files, empty files, malformed column schemas, and invalid data types gracefully without exposing Python stack traces.
- **File Safety:** Uploaded files are processed in-memory. No uploaded files are written to arbitrary local disk paths, preventing path traversal vulnerabilities.
- **Security & Secrets:** Repository contains zero hardcoded API keys, passwords, or tokens. Environment configuration uses `.env.example` placeholder templates. `.gitignore` excludes `.env`, virtual environments, caches, and local data files.

---

## Testing & Quality Assurance

- **Pytest Suite:** 25 core tests PASSED. An additional 7 FastAPI HTTP integration tests in `tests/test_api.py` require `httpx>=0.23.0` (listed in `requirements.txt`); they are skipped gracefully when `httpx` is unavailable in restricted environments and pass fully when it is installed.
- **Python Compilation:** `python -m compileall src/ app.py scripts/` completed cleanly with 0 errors.
- **Startup Smoke Test:** `python -c "from src.ui.views.overview import render_overview; from app import main; print('STARTUP OK')"` output verified (`STARTUP OK`).
- **Vectorized Equivalence:** Equivalence check verified exact prediction equality ($0.0000$ max difference) between vectorized batch inference and legacy implementations.

---

## Repository Structure

```text
InvoiceGuard_AI/
├── app.py                      # Main Streamlit application router & UI views
├── requirements.txt            # Pinned runtime dependencies
├── .env.example                # Environment variables template
├── src/
│   ├── api.py                  # FastAPI REST endpoints
│   ├── customer_360.py         # Customer 360 profile builder & aggregator
│   ├── data_quality.py         # Schema validation & profiling tools
│   ├── model_registry.py       # Model metadata loader & registry
│   ├── pipeline.py             # Invoice model inference & feature pipeline
│   ├── retail_pipeline.py      # UCI retail dataset pipeline & feature engineering
│   └── ui/
│       ├── components.py       # Reusable UI component library
│       ├── css.py              # Responsive CSS design system
│       ├── data.py             # Vectorized dashboard dataset loader
│       ├── navigation.py       # Sidebar & navigation router
│       └── views/              # Page view modules (Overview, Customer 360, etc.)
├── models/                     # Serialized model artifacts (.joblib & metadata .json)
├── scripts/                    # Data download, model training, and evaluation scripts
├── reports/                    # Generated JSON metrics & evaluation reports
├── docs/                       # Model cards, quality reports, and architecture documentation
└── tests/                      # Automated unit & integration test suite
```

---

## Quickstart & Installation

### 1. Clone Repository
```bash
git clone https://github.com/Mohamed-Osama-ai7/Inovice-Guard-ai.git
cd Inovice-Guard-ai
```

### 2. Environment Setup
Create and activate a virtual environment (Python 3.9+ recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Optional Environment Configuration
Copy `.env.example` to `.env` if external services (such as optional Groq API features) are used:
```bash
cp .env.example .env
```

### 5. Download Datasets

Two separate datasets power different parts of the application:

**UCI Online Retail II** (customer analytics and revenue intelligence modules):
```bash
python scripts/download_retail_data.py
```
Downloads `online_retail_II.xlsx` from the UCI ML Repository into `data/raw/`.

**Kaggle Invoice Dataset** (invoice payment-risk model retraining):
```bash
python scripts/download_data.py
```
Downloads the B2B invoice dataset via `kagglehub` (requires Kaggle API credentials). The application ships with pre-trained demo artifacts and does not require this step to run.

### 6. Run Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Verification & Testing Commands

Run the full automated test suite:
```bash
python -m pytest tests/ -v
```

Verify Python source compilation:
```bash
python -m compileall src/ app.py scripts/
```

Run application startup check:
```bash
python -c "from src.ui.views.overview import render_overview; from app import main; print('STARTUP OK')"
```

---

## License & Attribution

- **Source Code:** Released under repository terms.
- **UCI Online Retail II Dataset:** Licensed under Creative Commons Attribution 4.0 International (CC BY 4.0). Attribution to Daqing Chen (2012), UCI Machine Learning Repository (DOI: 10.24432/C5CG6D).