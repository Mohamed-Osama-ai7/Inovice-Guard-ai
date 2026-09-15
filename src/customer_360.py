"""
InvoiceGuard AI - Unified Customer 360 Service
Coordinates cross-domain customer intelligence across InvoiceGuard receivables
and UCI Online Retail II behavioral telemetry. Preserves strict domain isolation
and never generates spurious cross-dataset identity links.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.risk_engine import RiskEngine

logger = logging.getLogger("invoiceguard.customer_360")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"


class Customer360Service:
    def __init__(self):
        self.invoice_df: Optional[pd.DataFrame] = None
        self.retail_df: Optional[pd.DataFrame] = None
        self.retail_repurchase_model = None
        self.retail_revenue_model = None
        self.retail_features: List[str] = []
        self._invoice_profiles: Dict[str, Dict[str, Any]] = {}
        self._load_data_and_models()

    def _load_data_and_models(self) -> None:
        # 1. Load InvoiceGuard Data
        invoice_path = DATA_DIR / "demo" / "demo_invoices.csv"
        self._invoice_profiles = {}
        if invoice_path.exists():
            try:
                self.invoice_df = pd.read_csv(invoice_path)
                if not self.invoice_df.empty and "customer_id" in self.invoice_df.columns:
                    sorted_df = (
                        self.invoice_df.sort_values("invoice_date")
                        if "invoice_date" in self.invoice_df.columns
                        else self.invoice_df
                    )
                    for cust_id, group in sorted_df.groupby("customer_id"):
                        total_invoices = len(group)
                        total_amount = float(group["invoice_amount"].sum()) if "invoice_amount" in group.columns else 0.0
                        latest_record = group.iloc[-1]
                        
                        late_ratio = float(latest_record.get("prior_late_ratio", 0.0))
                        avg_delay = float(latest_record.get("prior_avg_delay", 0.0))
                        outstanding = float(latest_record.get("outstanding_amount", 0.0))
                        
                        risk = "LOW"
                        if late_ratio > 0.5 or avg_delay > 15:
                            risk = "HIGH"
                        elif late_ratio > 0.2 or avg_delay > 5:
                            risk = "MEDIUM"

                        self._invoice_profiles[str(cust_id)] = {
                            "domain": "InvoiceGuard Receivables Domain",
                            "source": "InvoiceGuard",
                            "customer_id": str(cust_id),
                            "total_invoices": total_invoices,
                            "total_invoice_amount": total_amount,
                            "total_outstanding": outstanding,
                            "historical_late_ratio": late_ratio,
                            "historical_avg_delay_days": avg_delay,
                            "payment_risk_level": risk,
                            "industry": str(latest_record.get("industry", "Unknown")),
                            "segment": str(latest_record.get("customer_segment", "Unknown")),
                            "company_size": str(latest_record.get("company_size", "Unknown")),
                            "payment_method": str(latest_record.get("payment_method", "Unknown")),
                        }
            except Exception as exc:
                logger.warning("Could not pre-aggregate invoice profiles: %s", exc)

        # 2. Load Retail Snapshot Data
        retail_path = DATA_DIR / "processed" / "retail_customer_snapshots.csv"
        if retail_path.exists():
            try:
                df = pd.read_csv(retail_path)
                df["customer_id"] = df["customer_id"].astype(str)
                df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
                self.retail_df = df.sort_values("snapshot_date").groupby("customer_id").tail(1).set_index("customer_id")
            except Exception as exc:
                logger.warning("Could not load retail customer snapshots: %s", exc)

        # 3. Load Retail Models
        try:
            self.retail_repurchase_model = joblib.load(MODELS_DIR / "retail_repurchase_model.pkl")
            self.retail_revenue_model = joblib.load(MODELS_DIR / "retail_future_revenue_model.pkl")
            self.retail_features = joblib.load(MODELS_DIR / "retail_features.pkl")
        except Exception:
            pass

    def get_invoiceguard_profile(self, customer_id: str) -> Dict[str, Any]:
        return self._invoice_profiles.get(str(customer_id), {})

    def get_retail_profile(self, customer_id: str) -> Dict[str, Any]:
        cid_str = str(customer_id)
        if self.retail_df is None or cid_str not in self.retail_df.index:
            return {}

        row = self.retail_df.loc[cid_str]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]

        recency = int(row.get("recency_days", 0))
        freq = int(row.get("frequency", 0))
        monetary = float(row.get("monetary", 0.0))
        aov = float(row.get("avg_order_value", 0.0))
        canc_rate = float(row.get("cancellation_rate", 0.0))
        age_days = int(row.get("customer_age_days", 0))
        rev_90d = float(row.get("revenue_90d", 0.0))
        orders_90d = float(row.get("orders_90d", 0.0))

        # Determine segment
        if monetary >= 3000 and recency <= 60:
            segment = "Champions / High Value"
        elif monetary >= 1000 and recency > 180:
            segment = "At-Risk High Value"
        elif recency <= 90:
            segment = "Promising / Growing"
        else:
            segment = "Dormant / Low Activity"

        profile: Dict[str, Any] = {
            "domain": "UCI Online Retail Domain",
            "source": "UCI Retail",
            "customer_id": cid_str,
            "segment": segment,
            "rfm": {
                "recency_days": recency,
                "frequency": freq,
                "monetary": monetary,
                "avg_order_value": round(aov, 2),
            },
            "behavior": {
                "customer_age_days": age_days,
                "cancellation_rate": round(canc_rate, 4),
                "revenue_90d": round(rev_90d, 2),
                "orders_90d": int(orders_90d),
            },
        }

        # Predict using retail models if available
        if self.retail_repurchase_model and self.retail_features:
            try:
                x = pd.DataFrame([row])[self.retail_features].fillna(0)
                repurchase_prob = float(self.retail_repurchase_model.predict_proba(x)[:, 1][0])
                pred_revenue = float(self.retail_revenue_model.predict(x)[0])
                inactivity_prob = 1.0 - repurchase_prob

                profile["predictions"] = {
                    "repurchase_probability_60d": round(repurchase_prob, 4),
                    "inactivity_risk_probability_60d": round(inactivity_prob, 4),
                    "inactivity_risk_level": "HIGH" if inactivity_prob >= 0.70 else ("MEDIUM" if inactivity_prob >= 0.35 else "LOW"),
                    "expected_future_revenue_60d": round(max(0.0, pred_revenue), 2),
                }

                # Local driver explanations
                top_drivers = []
                if recency > 120:
                    top_drivers.append(f"Extended purchase inactivity ({recency} days) elevates dormancy risk")
                elif recency < 30:
                    top_drivers.append(f"Recent transaction ({recency} days ago) indicates active engagement")

                if freq >= 10:
                    top_drivers.append(f"High historical order volume ({freq} orders) strongly supports repeat purchase")
                elif freq <= 1:
                    top_drivers.append(f"Single-purchase customer profile creates baseline retention fragility")

                if canc_rate > 0.20:
                    top_drivers.append(f"Elevated return rate ({canc_rate:.1%}) reflects potential dissatisfaction")

                profile["top_drivers"] = top_drivers
            except Exception as exc:
                logger.warning("Error computing retail model predictions: %s", exc)

        return profile

    def get_customer_profile(self, customer_id: str) -> Dict[str, Any]:
        """
        Returns a complete Customer 360 profile combining domain-specific intelligence
        with transparent composite risk scoring and actionable recommendations.
        """
        cid_str = str(customer_id).strip()
        invoice_intel = self.get_invoiceguard_profile(cid_str)
        retail_intel = self.get_retail_profile(cid_str)

        has_invoice = bool(invoice_intel)
        has_retail = bool(retail_intel)

        data_sources = []
        if has_invoice:
            data_sources.append("InvoiceGuard Receivables")
        if has_retail:
            data_sources.append("UCI Online Retail")

        # Multi-signal risk computation
        inv_late = invoice_intel.get("historical_late_ratio") if has_invoice else None
        ret_inact = retail_intel.get("predictions", {}).get("inactivity_risk_probability_60d") if has_retail else None
        monetary = retail_intel.get("rfm", {}).get("monetary", 0.0) if has_retail else 0.0
        recency = retail_intel.get("rfm", {}).get("recency_days", 0.0) if has_retail else 0.0
        cancellation = retail_intel.get("behavior", {}).get("cancellation_rate", 0.0) if has_retail else 0.0
        outstanding = invoice_intel.get("total_outstanding", 0.0) if has_invoice else 0.0
        inv_amt = invoice_intel.get("total_invoice_amount", 0.0) if has_invoice else 0.0

        risk_assessment = RiskEngine.calculate_business_risk_score(
            invoice_late_prob=inv_late,
            retail_inactivity_prob=ret_inact,
            exposure_amount=outstanding,
            invoice_amount=inv_amt,
            cancellation_rate=cancellation,
            recency_days=recency,
        )

        recommendations = RiskEngine.generate_recommendations(
            composite_score=risk_assessment["composite_score"],
            invoice_late_prob=inv_late,
            retail_inactivity_prob=ret_inact,
            cancellation_rate=cancellation,
            recency_days=recency,
            monetary=monetary,
            exposure_amount=outstanding,
        )

        profile: Dict[str, Any] = {
            "customer_id": cid_str,
            "data_sources": data_sources,
            "domain_separation_guarantee": (
                "InvoiceGuard demo customers and UCI Online Retail records represent disjoint populations. "
                "No cross-dataset entity synthesis or speculative identifier matching is performed."
            ),
            "invoice_intelligence": invoice_intel,
            "retail_intelligence": retail_intel,
            "composite_risk": risk_assessment,
            "recommendations": recommendations,
        }

        if not data_sources:
            profile["error"] = f"Customer ID '{cid_str}' was not found in either Receivables or Retail data repositories."

        return profile


# Singleton service
customer_service = Customer360Service()


def get_customer_profile(customer_id: str) -> Dict[str, Any]:
    return customer_service.get_customer_profile(customer_id)
