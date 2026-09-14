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

## Model 2: NLP Payment Risk
- **Type**: Text Classification
- **Algorithm**: Calibrated Logistic Regression with TF-IDF (1-2 ngrams)
- **Target**: `risk_level` (Binary: 0 = LOW RISK, 1 = HIGH RISK)
- **Features**: Text strings.
- **Contract**: The model is strictly binary. The UI layer takes the predicted positive probability and applies business thresholds (e.g., `< 0.3` is LOW, `>= 0.3` is MEDIUM, `>= 0.5` is HIGH). MEDIUM is a derived UI category, not a learned third class.
- **OOT Performance**:
  - Perfect separation (PR-AUC: 1.0) on the small synthetic NLP holdout dataset.

## Model 3: Retail Repurchase Risk
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
All classification models are strictly calibrated using Isotonic Regression (`CalibratedClassifierCV` with `cv="prefit"`).
**Critical Validation:** The isotonic calibration mapping is fit on a genuinely independent **Validation Split (20%)**, while the base estimator is strictly fit on the **Train Split (50%)**. This completely prevents base-estimator leakage during calibration, guaranteeing that predicted probabilities correspond to true empirical likelihoods.

## Temporal Split & OOT Generalization
To prevent temporal data leakage, all datasets are strictly split chronologically into:
- **Train (50%)**: Used exclusively for fitting base algorithms.
- **Validation (20%)**: Used exclusively for Isotonic calibration of the base estimator.
- **Test (15%)**: Out-of-sample holdout for tuning.
- **OOT (15%)**: Strictly future, out-of-time data used exclusively for final reporting.

The generalization gaps between Validation and OOT metrics are minimal across all models (e.g., Invoice Risk PR-AUC shifts from 0.9750 to 0.9700). This indicates acceptable temporal generalization on the evaluated split, with no evidence of severe overfitting.
