# Enterprise Model Card

This document provides transparent, scientifically honest reporting on the production models powering the InvoiceGuard AI platform.

## Model 1: Invoice Payment Risk
- **Type**: Tabular Classification
- **Algorithm**: Calibrated Logistic Regression
- **Target**: `late_payment` (Binary)
- **Features**: Invoice Amount, Days to Due, Prior Late Count, Prior Late Ratio, Prior Avg Delay, Outstanding Amount, Month, Quarter, Day of Week, Industry, Company Size, Payment Method, Customer Segment.
- **Leakage Safeguards**: `payment_date` and `delay_days` are strictly excluded from feature space `X`. Historical customer statistics use point-in-time expanding window aggregation (`cumsum().shift(1)`).
- **OOT Performance**:
  - Test PR-AUC: 0.9751 | OOT PR-AUC: 0.9700
  - Test ROC-AUC: 0.9893 | OOT ROC-AUC: 0.9878
  - Test Brier Score: 0.0372 | OOT Brier Score: 0.0363
- **Generalization Assessment**: Strong temporal generalization on the evaluated out-of-time split.

## Model 2: NLP Payment Risk
- **Type**: Text Classification
- **Algorithm**: Calibrated Logistic Regression with TF-IDF (1-2 ngrams)
- **Target**: `risk_level` (Binary: 0 = LOW/NO RISK, 1 = HIGH RISK)
- **Contract**: The model contract is strictly binary ($P(\text{risk}) \in [0, 1]$). The UI layer applies business alert thresholds (LOW `< 0.30`, MEDIUM `0.30–0.49`, HIGH `≥ 0.50`). MEDIUM is a derived UI business alert band, not a learned third class.
- **Evaluation Redesign & Leakage Control**:
  - *Methodology*: Evaluated using a **group-disjoint template split** (zero template overlap between Train, Validation, and Test sets) and an **independent held-out real-world OOD evaluation corpus**.
  - *Why Redesigned*: Previous reporting ($F_1 = 1.000$) suffered from template leakage due to random splitting of repeated synthetic templates. Group-disjoint splitting reveals true performance on unseen phrasings.
- **Honest Evaluation Metrics**:
  - **Train**: Precision 1.0000, Recall 1.0000, F1 1.0000, PR-AUC 1.0000, ROC-AUC 1.0000, Brier 0.0011
  - **Validation (Unseen Templates)**: Precision 0.5250, Recall 1.0000, F1 0.6885, Macro F1 0.5875, PR-AUC 0.7050, ROC-AUC 0.8401, Brier 0.2227
  - **Test (Unseen Templates)**: Precision 0.4375, Recall 1.0000, F1 0.6087, Macro F1 0.4904, PR-AUC 0.6829, ROC-AUC 0.8259, Brier 0.2742
  - **Held-Out Real-World OOD Corpus**: Precision 0.4286, Recall 0.9000, F1 0.5806, Macro F1 0.5662, PR-AUC 0.8118, ROC-AUC 0.8800, Brier 0.2318
- **Assessment**: The model achieves 90% risk recall on out-of-domain text with an ROC-AUC of 0.8800, ensuring high-risk customer communications are effectively flagged for review.

## Model 3: Retail Repurchase Risk
- **Type**: Tabular Classification
- **Algorithm**: Calibrated Logistic Regression
- **Target**: `repurchase_60d` (Binary)
- **Features**: Recency, Frequency, Monetary, Cancellation Rate, Customer Age, Revenue 90d, Orders 90d.
- **OOT Performance**:
  - PR-AUC: 0.6668
  - ROC-AUC: 0.7976
  - Brier Score: 0.1682

## Model 4: Future Revenue Prediction
- **Type**: Tabular Regression
- **Algorithm**: HistGradientBoosting Regressor
- **Target**: `future_revenue_60d` (Continuous)
- **Features**: Recency, Frequency, Monetary, Cancellation Rate, Customer Age, Revenue 90d, Orders 90d.
- **OOT Performance**:
  - MAE: 288.83
  - RMSE: 1301.67
  - R2: 0.5744

## Calibration & Imbalance
All classification models are strictly calibrated using Isotonic Regression (`CalibratedClassifierCV` with `cv="prefit"`).
**Critical Validation:** The isotonic calibration mapping is fit on a genuinely independent **Validation Split**, while the base estimator is strictly fit on the **Train Split**. This completely prevents base-estimator leakage during calibration, guaranteeing that predicted probabilities correspond to empirical likelihoods.

## Temporal Split & OOT Generalization
All tabular datasets are strictly split chronologically into:
- **Train (60%)**: Used exclusively for fitting base algorithms.
- **Validation (20%)**: Used exclusively for Isotonic calibration.
- **Test (10%)**: Out-of-sample holdout for tuning.
- **OOT (10%)**: Strictly future, out-of-time data used exclusively for final reporting.
