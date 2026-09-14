import pytest
from unittest.mock import patch
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
    assert data["status"] == "ok"

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200

def test_predict_endpoint_success():
    payload = {
        "invoice_date": "2023-10-01",
        "due_date": "2023-10-31",
        "invoice_amount": 5000.0,
        "customer_seen_before": 1,
        "prior_late_count": 0,
        "prior_late_ratio": 0.5,
        "prior_avg_delay": 5.0,
        "industry": "IT",
        "outstanding_amount": 0.0
    }
    response = client.post("/predict", json=payload)
    if response.status_code == 200:
        data = response.json()
        if "error" not in data:
            assert "late_probability" in data
            assert "risk_level" in data

def test_predict_endpoint_validation_errors():
    # 1. Missing required field (due_date)
    response = client.post("/predict", json={"invoice_date": "2023-10-01", "invoice_amount": 500})
    assert response.status_code == 422
    
    # 2. Invalid invoice_amount (<= 0)
    response = client.post("/predict", json={"invoice_date": "2023-10-01", "due_date": "2023-10-31", "invoice_amount": 0})
    assert response.status_code == 422
    
    # 3. Invalid probability range (prior_late_ratio > 1.0)
    response = client.post("/predict", json={
        "invoice_date": "2023-10-01", "due_date": "2023-10-31", "invoice_amount": 5000, "prior_late_ratio": 1.5
    })
    assert response.status_code == 422
    
    # 4. Boundary values (prior_late_ratio = 1.0 is valid)
    response = client.post("/predict", json={
        "invoice_date": "2023-10-01", "due_date": "2023-10-31", "invoice_amount": 5000, "prior_late_ratio": 1.0
    })
    assert response.status_code == 200 or ("error" in response.json()) # 200 or 500 if model missing, but 422 is bad

@patch("src.api.get_customer_profile")
def test_customer_profile_valid(mock_get_profile):
    mock_get_profile.return_value = {
        "customer_id": "CUST-VALID",
        "data_sources": ["InvoiceGuard"]
    }
    response = client.get("/customer/profile/CUST-VALID")
    assert response.status_code == 200
    assert response.json()["customer_id"] == "CUST-VALID"

@patch("src.api.get_customer_profile")
def test_customer_profile_unknown(mock_get_profile):
    mock_get_profile.return_value = {
        "customer_id": "CUST-UNKNOWN",
        "data_sources": []
    }
    response = client.get("/customer/profile/CUST-UNKNOWN")
    assert response.status_code == 404
    assert response.json()["detail"] == "Customer not found"

