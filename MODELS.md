# InvoiceGuard AI — Model Artifacts

## Tabular Invoice Models

- `models/classifier.joblib`: Deployed production classifier pipeline. The active artifact is an isotonically calibrated Logistic Regression pipeline (`CalibratedClassifierCV` over `LogisticRegression`) with `ColumnTransformer` preprocessing. (Note: `models/metadata.json: best_model` records `random_forest` from historical raw-AUC benchmark selection, whereas `classifier.joblib` is the calibrated production model).

- `models/logistic_regression.joblib`: Logistic Regression baseline.
- `models/random_forest.joblib`: Random Forest baseline.
- `models/mlp_neural_network.joblib`: Feed-forward MLP neural network pipeline.
- `models/hist_gradient_boosting.joblib`: HistGradientBoosting pipeline.
- `models/delay_regressor.joblib`: Expected payment delay regression model (positive delay-days).

## NLP Models

- `models/nlp_payment_risk.joblib`: TF-IDF + Calibrated Logistic Regression binary payment-risk model. This is an optional component because the primary invoice dataset does not contain customer message text; it operates on separately supplied communications.
- `models/nlp_vectorizer.joblib`: TF-IDF vectorizer companion to the NLP model.

## Retail Intelligence Models

- `models/retail_repurchase_model.pkl`: Calibrated Logistic Regression for 60-day customer repurchase risk.
- `models/retail_future_revenue_model.pkl`: HistGradientBoosting Regressor for 60-day customer future revenue.

## Important Notes

- Tabular invoice artifacts (`classifier.joblib`, `delay_regressor.joblib`) are trained on the included synthetic demo dataset (`data/demo/`).
- Retail intelligence models (`retail_repurchase_model.pkl`, `retail_future_revenue_model.pkl`) are trained on the official UCI Online Retail II dataset (`data/raw/online_retail_II.xlsx` / `data/processed/retail_customer_snapshots.csv`).
- To retrain on the official Kaggle invoice dataset, run `python scripts/download_data.py` then `python -m src.train`.
- To retrain retail models on UCI Online Retail II, run `python scripts/download_retail_data.py` then `PYTHONPATH=. python scripts/build_retail_models.py`.
- Never use post-payment fields such as `payment_date` or `delay_days` as predictors; these are strictly excluded from feature space during training.
