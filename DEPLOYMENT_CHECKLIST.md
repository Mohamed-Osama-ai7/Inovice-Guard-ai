# Deployment Checklist for InvoiceGuard AI

## Files created/modified
- `app.py` — main Streamlit dashboard entrypoint
- `src/audit_data.py` — audit script compatible with `python -m src.audit_data`
- `requirements.txt` — includes Streamlit and project runtime dependencies
- `.gitignore` — updated for repo safety and deployment readiness
- `.streamlit/config.toml` — Streamlit Cloud-compatible config
- `src/api.py` — updated for Python 3.9 compatibility
- `src/pipeline.py` — updated for Python 3.9 compatibility

## Models found
- `models/classifier.joblib`
- `models/delay_regressor.joblib`
- `models/logistic_regression.joblib`
- `models/random_forest.joblib`
- `models/mlp_neural_network.joblib`
- `models/nlp_payment_risk.joblib`
- `models/metadata.json`

## Models successfully loaded
Verified in the current environment:
- classifier.joblib — loaded
- delay_regressor.joblib — loaded
- logistic_regression.joblib — loaded
- random_forest.joblib — loaded
- mlp_neural_network.joblib — loaded
- nlp_payment_risk.joblib — loaded
- metadata.json — loaded

## Tests performed
1. `py -m src.audit_data --source demo`
   - Passed
   - Generated `reports/data_audit.json`
2. `py -m src.train --source demo`
   - Passed
   - Best model: random_forest
   - Accuracy measured: 0.945
3. `py -m src.train`
   - Passed using auto dataset selection
4. `py -m streamlit run app.py --server.headless true`
   - Streamlit started successfully on local port 8501
5. Health check for FastAPI API server
   - `http://127.0.0.1:8000/health`
   - Returned `200 OK`

## Errors fixed
- Python 3.9 compatibility issue in `src/api.py` related to `str | None`
- Python 3.9 compatibility issue in `src/pipeline.py` related to `str | None`
- Missing `app.py` for Streamlit deployment
- Missing `src/audit_data.py` for requested audit command
- Streamlit startup path issue caused by running from the wrong working directory
- Missing `streamlit` dependency in requirements
- Updated `.gitignore` to avoid committing secrets and credentials

## Remaining limitations
- The current `classifier.joblib` artifact is trained on the demo dataset, not the real Kaggle dataset.
- The project currently does not use the real Kaggle dataset unless it is downloaded by the user.
- The NLP model is a demo/optional component and is not a production-grade customer-message model.
- SHAP is included as an optional explainability path, but the safest reliable explanation method in this codebase is the feature-importance fallback when SHAP is not compatible with the selected model.
- The reported 94.5% accuracy is from the demo dataset and must not be claimed as Kaggle results.

## Exact Streamlit Cloud deployment steps
1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create a new app from the repo.
3. Set the app path to `app.py`.
4. Choose Python version 3.9 or 3.10.
5. Use the repository root as the app directory.
6. Ensure the following files are present in the repo:
   - `app.py`
   - `requirements.txt`
   - `models/`
   - `reports/`
   - `.streamlit/config.toml`
7. Deploy.
8. If you want a stronger production setup for a final thesis/demo, retrain on the official Kaggle dataset locally first and replace the demo artifacts in `models/` with approved artifacts.

## Local run commands
- `cd InvoiceGuard_AI`
- `py -m src.audit_data --source demo`
- `py -m src.train --source demo`
- `py -m streamlit run app.py --server.headless true`

## Git commands
```bash
git init
git add .
git commit -m "Add InvoiceGuard AI Streamlit deployment"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```
