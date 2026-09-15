# InvoiceGuard AI — Model Artifacts

## Tabular Invoice Models

- `models/classifier.joblib`: Selected classifier pipeline. The current artifact uses Random Forest as the best-performing model (see `models/metadata.json: best_model`).
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

- All serialized artifacts in `models/` are trained on the included synthetic demo dataset (`data/demo/`).
- To retrain on the official Kaggle invoice dataset, run `python scripts/download_data.py` then `python -m src.train`.
- To retrain retail models on UCI Online Retail II, run `python scripts/download_retail_data.py` then `PYTHONPATH=. python scripts/build_retail_models.py`.
- Never use post-payment fields such as `payment_date` or `delay_days` as predictors; these are strictly excluded from feature space during training.
