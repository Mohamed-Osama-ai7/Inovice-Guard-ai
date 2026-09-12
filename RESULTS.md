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

Kaggle shows the dataset as B2B invoice/accounts-receivable data with payment due date, invoice date, invoice amount, baseline date and payment date, and the page currently shows CC BY-NC 4.0.
