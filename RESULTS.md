# Verified local result

The complete pipeline was executed locally on the included **synthetic demo dataset** to verify that preprocessing, leakage controls, model selection, regression, artifact saving, and API serving are wired correctly.

### Demo test-set metrics (chronological, untouched)

- Best classifier: **XGBoost**
- Accuracy: **94.78%**
- Precision: **90.96%**
- Recall: **91.30%**
- F1: **91.13%**
- ROC-AUC: **98.89%**
- PR-AUC: **97.79%**
- Delay-regression MAE: **5.90 days**
- Delay-regression RMSE: **7.96 days**

**These are not Kaggle results.** They only certify that the project runs end-to-end and that the demonstration dataset is learnable. After downloading the real Kaggle dataset, rerun training and use the generated `reports/test_metrics.json` for the thesis/report.

## Main Kaggle source
Customer Invoices Dataset — Payment Date Prediction on Open Invoices:
https://www.kaggle.com/datasets/pradumn203/payment-date-prediction-for-invoices-dataset

This is the source for the **invoice payment-risk model**. The Kaggle page lists the license as CC BY-NC 4.0. This dataset is not included in the repository; it must be downloaded separately via `python scripts/download_data.py` (requires Kaggle API credentials).

**Note:** The UCI Online Retail II dataset (used for customer analytics and revenue intelligence modules) is a separate dataset licensed under CC BY 4.0. See README.md for attribution and DOI.
