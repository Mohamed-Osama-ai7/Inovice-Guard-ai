@echo off
python scripts\generate_demo.py
python -m src.train --source demo
uvicorn src.api:app --host 0.0.0.0 --port 8000
