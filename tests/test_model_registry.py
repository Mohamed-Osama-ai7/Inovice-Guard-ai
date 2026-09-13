import pytest
from src.model_registry import ModelRegistry

def test_get_invoice_model_metadata():
    metadata = ModelRegistry.get_invoice_model_metadata()
    assert isinstance(metadata, dict)
    
def test_get_retail_repurchase_metadata():
    metadata = ModelRegistry.get_retail_repurchase_metadata()
    assert isinstance(metadata, dict)

def test_get_retail_revenue_metadata():
    metadata = ModelRegistry.get_retail_revenue_metadata()
    assert isinstance(metadata, dict)

def test_get_all_metrics():
    metrics = ModelRegistry.get_all_metrics()
    assert isinstance(metrics, dict)
    assert "invoice_payment_risk" in metrics
    assert "retail_repurchase" in metrics
    assert "retail_future_revenue" in metrics
    assert "nlp_payment_risk" in metrics
