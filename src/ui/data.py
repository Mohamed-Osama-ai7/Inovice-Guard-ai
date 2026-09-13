import streamlit as st
import pandas as pd
from typing import Dict, Any

# We reuse the cached function from the original implementation
# However, we must ensure we import it or redefine it safely.
# To keep it clean, we redefine the cache logic here calling into the original logic if needed,
# or just redefine it fully.
# The original logic requires `src.pipeline` and models.
# It's better to just use the one defined in app.py to avoid duplication.
from app import load_project_artifacts as orig_load
from app import _cached_build_dashboard_dataset, get_artifact_signature

def load_project_artifacts(signature: str) -> Dict[str, Any]:
    return orig_load(signature)

def get_dashboard_dataset(artifacts: Dict[str, Any]) -> pd.DataFrame:
    signature = get_artifact_signature()
    return _cached_build_dashboard_dataset(signature, artifacts)
