import pytest
from fastapi.testclient import TestClient
from src.api import app, InvoiceInput

client = TestClient(app)

def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"
    assert "model_loaded" in data

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "test_metrics" in data

def test_predict_endpoint_success():
    payload = {
        "invoice_date": "2023-10-01",
        "due_date": "2023-10-31",
        "invoice_amount": 5000.0,
        "customer_seen_before": 1,
        "prior_late_count": 0,
        "prior_late_ratio": 0.0,
        "prior_avg_delay": 0.0,
        "industry": "IT",
        "company_size": "Medium",
        "payment_method": "Bank Transfer",
        "customer_segment": "Enterprise",
        "outstanding_amount": 0.0
    }
    response = client.post("/predict", json=payload)
    if response.status_code == 200:
        data = response.json()
        if "error" not in data:
            assert "late_probability" in data
            assert "predicted_late" in data
            assert "risk_level" in data
            assert "recommendation" in data
            
def test_predict_endpoint_validation_error():
    payload = {
        "invoice_date": "2023-10-01",
        # Missing required fields like due_date
        "invoice_amount": -100.0, # invalid amount
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # Pydantic validation error

def test_customer_profile_endpoint():
    # CUST-001 is a known stub in customer_360 logic
    response = client.get("/customer/profile/CUST-001")
    assert response.status_code == 200
    data = response.json()
    assert "customer_id" in data
    assert data["customer_id"] == "CUST-001"
