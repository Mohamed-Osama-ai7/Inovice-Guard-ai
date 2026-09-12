# InvoiceGuard AI — Model Artifacts

## Tabular ML
- `models/classifier.joblib`: selected end-to-end classifier (XGBoost pipeline in current demo build)
- `models/logistic_regression.joblib`: Logistic Regression baseline
- `models/random_forest.joblib`: Random Forest baseline
- `models/mlp_neural_network.joblib`: Feed-forward MLP neural network pipeline
- `models/delay_regressor.joblib`: expected positive delay-days regression model

## NLP
- `models/nlp_payment_risk.joblib`: TF-IDF + Logistic Regression payment-communication risk model. This is an optional/demo NLP component because the selected Kaggle invoice dataset does not contain customer email/message text.

## Important
The included trained artifacts are demo-build artifacts. Run `python scripts/download_data.py`, then `python -m src.train` and `PYTHONPATH=. python scripts/build_all_models.py` against the real Kaggle data to produce final reportable metrics. Never use post-payment fields such as `payment_date` as predictors.
