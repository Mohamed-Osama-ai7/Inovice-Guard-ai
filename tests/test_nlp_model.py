from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from app import load_nlp_thresholds, score_nlp_message

MODEL_PATH = ROOT / "models" / "nlp_payment_risk.joblib"
UNSEEN_PATH = ROOT / "data" / "nlp_unseen_test.csv"


class TestNlpModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert MODEL_PATH.exists(), "Expected models/nlp_payment_risk.joblib to exist."
        cls.nlp_model = joblib.load(MODEL_PATH)

    def test_nlp_model_loads_and_produces_probabilities(self):
        probabilities = self.nlp_model.predict_proba(["Payment has been scheduled and will be completed on the agreed date."])[0]

        self.assertEqual(probabilities.shape, (2,))
        self.assertTrue(np.allclose(probabilities.sum(), 1.0, atol=1e-6))
        self.assertTrue(np.all(probabilities >= 0.0))

    def test_nlp_risk_labeling_for_known_messages(self):
        thresholds = load_nlp_thresholds()

        low_risk = score_nlp_message(
            self.nlp_model,
            "Payment has already been completed and the invoice is fully settled.",
            thresholds,
        )
        self.assertEqual(low_risk["risk_label"], "LOW RISK")

        high_risk = score_nlp_message(
            self.nlp_model,
            "Payment may be delayed and we expect to settle it soon.",
            thresholds,
        )
        self.assertEqual(high_risk["risk_label"], "HIGH RISK")

    def test_nlp_model_handles_unseen_messages(self):
        self.assertTrue(UNSEEN_PATH.exists(), "Expected data/nlp_unseen_test.csv to exist.")

        unseen = pd.read_csv(UNSEEN_PATH)
        self.assertFalse(unseen.empty)

        sample = unseen.iloc[0].to_dict()
        analysis = score_nlp_message(self.nlp_model, sample["text"], load_nlp_thresholds())

        self.assertIn(analysis["risk_label"], {"LOW RISK", "MEDIUM RISK", "HIGH RISK"})
        self.assertGreaterEqual(analysis["probability"], 0.0)

    def test_nlp_model_rejects_empty_input(self):
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            score_nlp_message(self.nlp_model, "   ")


if __name__ == "__main__":
    unittest.main()
