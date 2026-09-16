# InvoiceGuard AI — Results

## Current Deployed Model: Invoice Payment-Risk Classifier

The deployed inference artifact is `models/classifier.joblib` — a Calibrated Logistic Regression pipeline (isotonic regression via `CalibratedClassifierCV(cv="prefit")`).

### Test-Set and OOT Metrics (Current Deployed Classifier)

Metrics are from `models/classifier_metadata.json`.

| Split | Precision | Recall | F1 | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|---|---|---|
| Train | 0.9553 | 0.9036 | 0.9287 | 0.9782 | 0.9898 | 0.0340 |
| Validation | 0.9491 | 0.9031 | 0.9255 | 0.9750 | 0.9892 | 0.0344 |
| Test | 0.9616 | 0.8870 | 0.9228 | 0.9751 | 0.9893 | 0.0372 |
| OOT | 0.9491 | 0.8809 | 0.9137 | 0.9700 | 0.9878 | 0.0363 |

Generalization assessment: Strong temporal generalization on the evaluated OOT split.

### Payment Delay Regression

Source: `models/metadata.json`.

- MAE: 5.90 days
- RMSE: 7.96 days
- Evaluated on: delayed invoices in the test split (n=529)

---

## Retail & Customer Intelligence Models (UCI Online Retail II)

Evaluated under strict point-in-time snapshot generation with chronological forward splits.

### 1. Repurchase Risk Classifier (`retail_repurchase_model.pkl`)

- **Algorithm**: Logistic Regression with Isotonic Probability Calibration (`CalibratedClassifierCV`)
- **Target**: `repurchase_60d` (binary indicator: $\ge 1$ order in $[T, T+60\text{d})$)
- **Source Metadata**: `models/retail_repurchase_model_metadata.json`

| Split | Precision | Recall | F1 | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|---|---|---|
| Train | 0.7843 | 0.2628 | 0.3937 | 0.6529 | 0.7406 | 0.2026 |
| Validation | 0.7139 | 0.3889 | 0.5035 | 0.6298 | 0.7906 | 0.1501 |
| Test | 0.7175 | 0.4088 | 0.5208 | 0.6478 | 0.8133 | 0.1460 |
| OOT | 0.7387 | 0.3959 | 0.5155 | 0.6668 | 0.7976 | 0.1682 |

### 2. Future Revenue Regressor (`retail_future_revenue_model.pkl`)

- **Algorithm**: Histogram-based Gradient Boosting Regressor (`HistGradientBoostingRegressor`)
- **Target**: `future_revenue_60d` (continuous forward spending in GBP)
- **Source Metadata**: `models/retail_future_revenue_model_metadata.json`

| Split | MAE (£) | RMSE (£) | R² Score |
|---|---|---|---|
| Train | £279.71 | £764.26 | 0.8054 |
| Validation | £242.47 | £784.89 | 0.3304 |
| Test | £236.21 | £895.87 | 0.5617 |
| OOT | £288.84 | £1,301.68 | 0.5744 |

### 3. Customer Cohort Segmentation (KMeans)

- **Algorithm**: KMeans clustering on normalized RFM and cancellation rates ($K=4$)
- **Evaluated Customers**: 5,499
- **Quality Metrics**: Silhouette Score: **0.3231**, Davies-Bouldin Index: **0.9930**
- **Identified Segments**:
  - Champions / High Value (1,580 customers, 28.7% share)
  - Promising / Growing (1,795 customers, 32.6% share)
  - At-Risk High Value (589 customers, 10.7% share)
  - Dormant / Low Activity (1,535 customers, 27.9% share)

---

## Historical / Experimental Benchmark

> **Note:** The results below are from an earlier experimental benchmark run against the **synthetic demo dataset** using an XGBoost classifier during model selection. These are NOT the current deployed inference artifact.
>
> They are preserved for reference only and should not be cited as current performance.

### Demo Dataset Experimental Benchmark (Historical, XGBoost, Model Selection)

- Best classifier in selection: XGBoost
- Accuracy: 94.78%
- Precision: 90.96%
- Recall: 91.30%
- F1: 91.13%
- ROC-AUC: 98.89%
- PR-AUC: 97.79%
- Delay-regression MAE: 5.90 days
- Delay-regression RMSE: 7.96 days

These results confirm the end-to-end pipeline runs correctly on the demo dataset. They are not performance claims for the deployed model.

---

## Data Sources

### UCI Online Retail II (Retail & Customer Intelligence Modules)
- License: CC BY 4.0
- Source: https://archive.ics.uci.edu/dataset/502/online+retail+ii
- DOI: https://doi.org/10.24432/C5CG6D

### Kaggle Invoice Dataset (Invoice Payment-Risk Model)
- License: CC BY-NC 4.0 (Kaggle listing)
- Source: https://www.kaggle.com/datasets/pradumn203/payment-date-prediction-for-invoices-dataset
- Not included in the repository. Download requires Kaggle API credentials.
- The application ships with a pre-trained model; retraining requires downloading this dataset separately.
