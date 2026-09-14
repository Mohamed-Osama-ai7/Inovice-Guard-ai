# Enterprise Model Card

This document provides transparent reporting on the production models powering the InvoiceGuard AI platform.

## Model 1: Invoice Payment Risk
- **Type**: Tabular Classification
- **Algorithm**: Calibrated Logistic Regression
- **Target**: `late_payment` (Binary)
- **Features**: Invoice Amount, Days to Due, Prior Late Count, Prior Avg Delay, Month, Quarter.
- **OOT Performance**:
  - PR-AUC: 0.9700
  - ROC-AUC: 0.9878
  - Brier Score: 0.0363

## Model 2: Retail Repurchase Risk
- **Type**: Tabular Classification
- **Algorithm**: Calibrated Logistic Regression
- **Target**: `repurchase_60d` (Binary)
- **Features**: Recency, Frequency, Monetary, Cancellation Rate, Customer Age, Revenue 90d, Orders 90d.
- **OOT Performance**:
  - PR-AUC: 0.6668
  - ROC-AUC: 0.7976
  - Brier Score: 0.1682

## Model 3: Future Revenue Prediction
- **Type**: Tabular Regression
- **Algorithm**: HistGradientBoosting Regressor
- **Target**: `future_revenue_60d` (Continuous)
- **Features**: Same as Model 2.
- **OOT Performance**:
  - MAE: 288.83
  - RMSE: 1301.67
  - R2: 0.5744

## Calibration & Imbalance
All classification models are calibrated using Isotonic Regression (`CalibratedClassifierCV`) to ensure that predicted probabilities strictly correspond to the empirical likelihood of the event occurring. Class imbalance is handled dynamically via inverse frequency weighting during training.

## Temporal Split & OOT Generalization
To prevent temporal data leakage, all datasets are strictly split chronologically into:
- **Train (50%)**: Used for fitting base algorithms.
- **Validation (20%)**: Used for Isotonic calibration and hyperparameter tuning.
- **Test (15%)**: Out-of-sample holdout.
- **OOT (15%)**: Strictly future, out-of-time data used exclusively for final reporting.

The generalization gaps between Validation and OOT metrics are minimal across all models (e.g., Invoice Risk PR-AUC shifts from 0.9750 to 0.9700). This indicates acceptable temporal generalization on the evaluated split, with no evidence of severe overfitting.
