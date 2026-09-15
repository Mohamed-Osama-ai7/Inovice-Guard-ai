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
