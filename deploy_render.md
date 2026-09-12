# Render deployment

1. Create a GitHub repository and upload this project.
2. In Render, create a **Web Service** from the repository.
3. Choose Docker; `Dockerfile` is already included.
4. Deploy.
5. Open `/health` to verify the model is loaded.

The included container ships with the verified demo model so the service starts without downloading Kaggle at runtime.

For a production/thesis deployment, retrain on the official Kaggle data locally, review the data license/redistribution requirements, then replace the demo model artifacts with the approved real-data artifacts.
