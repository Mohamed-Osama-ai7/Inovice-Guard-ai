@echo off
python scripts\download_data.py
python -m src.train --source kaggle
uvicorn src.api:app --host 0.0.0.0 --port 8000
