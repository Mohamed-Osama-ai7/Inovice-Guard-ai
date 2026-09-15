import streamlit as st
from typing import Dict, Any
from src.ui.components import page_header, kpi_row, divider
from app import render_artifact_diagnostics


def render_system_status(artifacts: Dict[str, Any]) -> None:
    """Delegates to the main artifact diagnostics view, branded as System Status."""
    render_artifact_diagnostics(artifacts)


def render_data_quality(artifacts: Dict[str, Any]) -> None:
    page_header("Data Quality", "Automated diagnostics and artifact validation.")
    render_artifact_diagnostics(artifacts)
