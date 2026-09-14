# InvoiceGuard AI
**Enterprise Receivables & Customer Intelligence Platform**

> Detect · Predict · Explain · Act

---

## 🚀 Live Demo

### **[https://inovice-guard-ai-17.streamlit.app/](https://inovice-guard-ai-17.streamlit.app/)**

> Fully functional AI Enterprise platform — no sign-in required. Runs against real trained ML/NLP models with no fabricated metrics.

---

## 📌 Product Overview

InvoiceGuard AI is an enterprise-grade AI SaaS application that transforms reactive collections into proactive **Revenue Intelligence**.

By integrating **Invoice Payment Risk** with the **UCI Online Retail II dataset**, InvoiceGuard AI acts as a complete **Customer 360 Platform**, identifying:
- Which invoices have a high probability of late payment?
- What is the expected financial exposure of at-risk receivables?
- Which high-value customers are at risk of churn?
- What is the forecasted future revenue per customer?

---

## 💡 The Solution

InvoiceGuard AI provides an end-to-end payment-risk intelligence layer:

1. **Invoice Risk Prediction** — ML classifiers score each invoice's probability of late payment.
2. **Customer Intelligence (360)** — Holistic views of customer lifetime value, historical delays, and retention status.
3. **Revenue Forecast** — Predicts upcoming 60-day revenue using advanced Gradient Boosting Regressors.
4. **NLP Communication Analysis** — Customer payment messages are classified as HIGH / MEDIUM / LOW risk using TF-IDF + Logistic Regression.
5. **Explainability** — SHAP-based feature attribution reveals the specific drivers behind predictions.
6. **Strict Data Quality** — Built-in schema validation and leakage protection ensures production safety.

---

## ✨ Key Enterprise Features

| Feature | Description |
|---|---|
| 🏠 **Executive Overview** | Portfolio KPIs, risk distribution, and prioritized financial exposure. |
| 👤 **Customer 360** | Comprehensive CRM-style view with payment behavior history and risk trend. |
| 💰 **Revenue Intelligence** | Financial exposure calculations, revenue forecasting, and repurchase risk. |
| 🧠 **Explainable AI** | SHAP-based feature attribution showing the top risk drivers for trust. |
| 🗣️ **NLP Intelligence** | Payment message analysis using a trained NLP classifier on customer emails. |
| 📊 **Model Center** | OOT Evaluation, Calibration metrics, and Data Leakage reporting. |

---

## 🧠 Machine Learning & Rigor

This project strictly adheres to enterprise ML best practices:
- **Zero Data Leakage:** Evaluated through chronological train/val/test/OOT splits. Customer historical features are strictly `shift(1)` to avoid future leakage.
- **Isotonic Calibration:** Classifier probabilities are calibrated via `CalibratedClassifierCV` to ensure scores reflect real-world empirical risk likelihoods.
- **OOT Generalization:** Real metrics are reported from strictly future, out-of-time datasets.

**Invoice Tabular Risk Model:**
- **Algorithm:** Calibrated Logistic Regression
- **OOT Performance:** PR-AUC: 0.9700 | ROC-AUC: 0.9878 | Brier Score: 0.0363

**NLP Payment Risk Model:**
- **Algorithm:** TF-IDF + Logistic Regression
- **Performance:** 100% accuracy on strictly held-out message validation sets.

*See `docs/model_card.md` and `docs/model_quality_report.md` for exact metrics and evaluation.*

---

## 🏗️ Architecture

```
User Input / CSV Batch / Customer Profile
        │
        ▼
   Data Contract Validation (Strict Pydantic / Pandas Schema)
        │
        ▼
   ML Inference Engine
  (XGBoost, HistGradientBoosting, LogReg, TF-IDF)
        │                          
        ▼                          
   Risk Probabilities (Calibrated 0–1)
   Expected Delay (Days)
   Revenue Forecast ($)
        │
        ▼
   Financial Exposure & SHAP Explainability
        │
        ▼
   Enterprise SaaS UI (Streamlit View Router)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit (Enterprise custom CSS & Component Architecture) |
| Backend API | FastAPI + Pydantic (Strict Data Validation) |
| ML Core | scikit-learn, HistGradientBoosting, SHAP |
| Testing | pytest, unittest |
| Deployment | Streamlit Cloud (Primary), Docker |

---

## ⚙️ Installation

```bash
git clone https://github.com/Mohamed-Osama-ai7/Inovice-Guard-ai.git
cd Inovice-Guard-ai
pip install -r requirements.txt
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501).

---

## ☁️ Deployment

### Streamlit Cloud (Primary)

The application is deployed on Streamlit Community Cloud:
**[InvoiceGuard AI Enterprise - Live Deployment](https://inovice-guard-ai-17.streamlit.app/)**

> **Note on Access**: This application is currently configured as a private enterprise deployment. You may be redirected to a `share.streamlit.io/-/auth/app` login screen. You must be granted access by the repository owner to view the live dashboard.

Streamlit Cloud runs the highly scalable view router architecture natively from the `main` branch. All required inference artifacts are versioned in `/models`.

### Docker + FastAPI
The REST API can be hosted locally via the containerized environment.
```bash
docker build -t invoiceguard-ai .
docker run -p 8000:8000 invoiceguard-ai
```

---

## 📄 Accuracy Policy
This project does **not** hardcode or fabricate accuracy claims. All displayed metrics are computed dynamically from the trained artifacts against the real chronological test splits. Real datasets (InvoiceGuard + UCI Online Retail II) are used with rigorous feature engineering.