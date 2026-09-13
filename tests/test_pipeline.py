from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import load_project_artifacts, run_prediction, validate_single_input, get_artifact_signature
from src.api import InvoiceInput, row, predict, health, metrics


class TestInvoiceGuardPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts = load_project_artifacts(get_artifact_signature())
        cls.metadata = cls.artifacts.get("metadata", {})
        cls.models = cls.artifacts.get("models", {})
        assert "classifier" in cls.models, "Expected classifier to be loaded."

    def test_artifacts_load_successfully(self):
        self.assertIn("classifier", self.models)
        self.assertIn("delay_regressor", self.models)
        self.assertIn("nlp_payment_risk", self.models)
        self.assertGreater(len(self.metadata), 0)

    def test_known_customer_prediction(self):
        payload = {
            "invoice_date": "2024-01-15",
            "due_date": "2024-02-14",
            "invoice_amount": 25000.0,
            "customer_seen_before": 5,
            "prior_late_count": 0,
            "prior_late_ratio": 0.0,
            "prior_avg_delay": 0.0,
            "outstanding_amount": 0.0,
            "industry": "Technology",
            "company_size": "Enterprise",
            "payment_method": "Bank Transfer",
            "customer_segment": "Tier 1",
        }
        res = run_prediction(payload, "classifier", self.artifacts)
        self.assertIn("late_probability", res)
        self.assertGreaterEqual(res["late_probability"], 0.0)
        self.assertLessEqual(res["late_probability"], 1.0)
        self.assertIn(res["risk_level"], {"LOW", "MEDIUM", "HIGH", "CRITICAL"})
        self.assertGreaterEqual(res["expected_delay_days"], 0.0)

    def test_unseen_customer_cold_start_prediction(self):
        payload = {
            "invoice_date": "2024-03-01",
            "due_date": "2024-03-31",
            "invoice_amount": 15000.0,
            "customer_seen_before": 0,
            "prior_late_count": 0,
            "prior_late_ratio": 0.0,
            "prior_avg_delay": 0.0,
            "outstanding_amount": 0.0,
            "industry": "Unknown New Industry",
            "company_size": "Unknown Size",
            "payment_method": "Credit Card",
            "customer_segment": "New Client",
        }
        res = run_prediction(payload, "classifier", self.artifacts)
        self.assertIn("late_probability", res)
        self.assertIn(res["risk_level"], {"LOW", "MEDIUM", "HIGH", "CRITICAL"})

    def test_missing_and_unknown_fields_handling(self):
        payload = {
            "invoice_date": "2024-05-10",
            "due_date": "2024-06-09",
            "invoice_amount": None,
            "industry": None,
            "company_size": "NonExistentSize",
        }
        res = run_prediction(payload, "classifier", self.artifacts)
        self.assertIn("late_probability", res)
        self.assertFalse(np.isnan(res["late_probability"]))

    def test_input_validation(self):
        errors = validate_single_input({"invoice_amount": -500})
        self.assertTrue(any("greater than zero" in e for e in errors))

        errors_dates = validate_single_input({
            "invoice_amount": 1000,
            "invoice_date": "2024-05-10",
            "due_date": "2024-05-01"
        })
        self.assertTrue(any("cannot be earlier than" in e for e in errors_dates))


    def test_api_endpoints(self):
        h = health()
        self.assertEqual(h["status"], "ok")
        self.assertTrue(h["model_loaded"])

        m = metrics()
        self.assertIn("test_metrics", m)

        inv = InvoiceInput(
            invoice_date="2024-02-01",
            due_date="2024-03-02",
            invoice_amount=5000.0,
            customer_seen_before=2,
            prior_late_count=1.0,
            prior_late_ratio=0.5,
            prior_avg_delay=4.0
        )
        res = predict(inv)
        self.assertIn("late_probability", res)
        self.assertIn("predicted_late", res)
        self.assertIn(res["predicted_late"], {0, 1})


if __name__ == "__main__":
    unittest.main()
