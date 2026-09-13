import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

class Customer360Service:
    def __init__(self):
        self.invoice_df = None
        self.retail_df = None
        self.retail_repurchase_model = None
        self.retail_revenue_model = None
        self.retail_features = None
        self._load_data_and_models()

    def _load_data_and_models(self):
        # Load InvoiceGuard Data (using demo_invoices as the source of truth for customer history)
        invoice_path = DATA_DIR / "demo" / "demo_invoices.csv"
        if invoice_path.exists():
            self.invoice_df = pd.read_csv(invoice_path)
        
        # Load Retail Data (latest snapshot per customer)
        retail_path = DATA_DIR / "processed" / "retail_customer_snapshots.csv"
        if retail_path.exists():
            # Get only the latest snapshot per customer
            df = pd.read_csv(retail_path)
            df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
            self.retail_df = df.sort_values('snapshot_date').groupby('customer_id').tail(1).set_index('customer_id')

        # Load Retail Models
        try:
            self.retail_repurchase_model = joblib.load(MODELS_DIR / "retail_repurchase_model.pkl")
            self.retail_revenue_model = joblib.load(MODELS_DIR / "retail_future_revenue_model.pkl")
            self.retail_features = joblib.load(MODELS_DIR / "retail_features.pkl")
        except FileNotFoundError:
            pass # Models not built yet

    def get_invoiceguard_profile(self, customer_id: str) -> Dict[str, Any]:
        if self.invoice_df is None:
            return {}
        
        cust_data = self.invoice_df[self.invoice_df['customer_id'] == customer_id]
        if cust_data.empty:
            return {}
            
        # Aggregate invoice history
        total_invoices = len(cust_data)
        total_amount = cust_data['invoice_amount'].sum()
        
        # Determine risk based on latest prior_late_ratio (as a heuristic for the customer profile)
        latest_record = cust_data.sort_values('invoice_date').iloc[-1]
        late_ratio = latest_record.get('prior_late_ratio', 0)
        avg_delay = latest_record.get('prior_avg_delay', 0)
        
        risk = "LOW"
        if late_ratio > 0.5 or avg_delay > 15:
            risk = "HIGH"
        elif late_ratio > 0.2 or avg_delay > 5:
            risk = "MEDIUM"

        return {
            "source": "InvoiceGuard",
            "total_invoices": total_invoices,
            "total_invoice_amount": float(total_amount),
            "historical_late_ratio": float(late_ratio),
            "historical_avg_delay_days": float(avg_delay),
            "payment_risk_level": risk,
            "industry": str(latest_record.get('industry', 'Unknown')),
            "segment": str(latest_record.get('customer_segment', 'Unknown'))
        }

    def get_retail_profile(self, customer_id: str) -> Dict[str, Any]:
        if self.retail_df is None or customer_id not in self.retail_df.index:
            return {}
            
        row = self.retail_df.loc[customer_id]
        
        profile = {
            "source": "UCI Retail",
            "rfm": {
                "recency_days": int(row.get('recency_days', 0)),
                "frequency": int(row.get('frequency', 0)),
                "monetary": float(row.get('monetary', 0)),
                "avg_order_value": float(row.get('avg_order_value', 0))
            },
            "behavior": {
                "customer_age_days": int(row.get('customer_age_days', 0)),
                "cancellation_rate": float(row.get('cancellation_rate', 0)),
                "revenue_90d": float(row.get('revenue_90d', 0)),
                "orders_90d": float(row.get('orders_90d', 0))
            }
        }

        # Predict if models are loaded
        if self.retail_repurchase_model and self.retail_features:
            x = pd.DataFrame([row])[self.retail_features].fillna(0)
            repurchase_prob = float(self.retail_repurchase_model.predict_proba(x)[:, 1][0])
            pred_revenue = float(self.retail_revenue_model.predict(x)[0])
            
            profile["predictions"] = {
                "repurchase_probability_60d": round(repurchase_prob, 4),
                "retention_risk": "HIGH" if repurchase_prob < 0.3 else ("MEDIUM" if repurchase_prob < 0.7 else "LOW"),
                "expected_future_revenue_60d": round(max(0, pred_revenue), 2)
            }
            
            # Recommendations
            recs = []
            if profile["predictions"]["retention_risk"] == "HIGH" and profile["rfm"]["monetary"] > 1000:
                recs.append("High churn risk + high customer value -> Retention intervention needed.")
            if profile["rfm"]["recency_days"] > 90:
                recs.append("Customer inactive for over 90 days. Send re-engagement campaign.")
            profile["recommendations"] = recs

        return profile

    def get_customer_profile(self, customer_id: str) -> Dict[str, Any]:
        """
        Returns a Customer 360 profile combining data from all available sources.
        """
        profile = {
            "customer_id": customer_id,
            "invoice_intelligence": self.get_invoiceguard_profile(customer_id),
            "retail_intelligence": self.get_retail_profile(customer_id)
        }
        
        # Summary Status
        has_invoice = bool(profile["invoice_intelligence"])
        has_retail = bool(profile["retail_intelligence"])
        
        if has_invoice and has_retail:
            profile["data_sources"] = ["InvoiceGuard", "UCI Retail"]
        elif has_invoice:
            profile["data_sources"] = ["InvoiceGuard"]
        elif has_retail:
            profile["data_sources"] = ["UCI Retail"]
        else:
            profile["data_sources"] = []
            profile["error"] = "Customer not found in any data source."

        return profile

# Singleton instance
customer_service = Customer360Service()

def get_customer_profile(customer_id: str):
    return customer_service.get_customer_profile(customer_id)
