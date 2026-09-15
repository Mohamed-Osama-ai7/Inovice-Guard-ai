"""
InvoiceGuard AI - Multi-Signal Business Risk & Recommendation Engine
Combines payment risk, customer behavioral signals, financial exposure,
and message intelligence into a transparent, calibrated Business Risk Score.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class RiskEngine:
    """
    Enterprise composite scoring layer.
    Explicitly labeled as a structured business scoring model rather than
    a synthetic single probability to maintain scientific integrity.
    """

    @staticmethod
    def calculate_business_risk_score(
        invoice_late_prob: Optional[float] = None,
        retail_inactivity_prob: Optional[float] = None,
        exposure_amount: float = 0.0,
        invoice_amount: float = 0.0,
        cancellation_rate: float = 0.0,
        recency_days: float = 0.0,
        nlp_risk_prob: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Computes a deterministic composite risk score (0-100) and risk tier.
        Gracefully handles customers present in Invoice, Retail, or both domains.
        """
        weights: Dict[str, float] = {}
        sub_scores: Dict[str, float] = {}

        # 1. Invoice Payment Signal
        if invoice_late_prob is not None:
            sub_scores["payment_risk"] = float(np_clip(invoice_late_prob, 0.0, 1.0))
            weights["payment_risk"] = 0.40

        # 2. Behavioral / Inactivity Signal
        if retail_inactivity_prob is not None:
            sub_scores["inactivity_risk"] = float(np_clip(retail_inactivity_prob, 0.0, 1.0))
            weights["inactivity_risk"] = 0.35
        elif recency_days > 0:
            recency_normalized = float(np_clip(recency_days / 180.0, 0.0, 1.0))
            sub_scores["inactivity_risk"] = recency_normalized
            weights["inactivity_risk"] = 0.25

        # 3. Cancellation / Operational Friction Signal
        if cancellation_rate > 0:
            sub_scores["cancellation_risk"] = float(np_clip(cancellation_rate * 2.0, 0.0, 1.0))
            weights["cancellation_risk"] = 0.15

        # 4. Financial Exposure Signal
        if invoice_amount > 0 and exposure_amount > 0:
            exposure_ratio = float(np_clip(exposure_amount / invoice_amount, 0.0, 1.0))
            sub_scores["financial_exposure"] = exposure_ratio
            weights["financial_exposure"] = 0.20
        elif exposure_amount > 50_000:
            sub_scores["financial_exposure"] = 0.85
            weights["financial_exposure"] = 0.15

        # 5. NLP Sentiment / Dispute Signal
        if nlp_risk_prob is not None:
            sub_scores["nlp_communication_risk"] = float(np_clip(nlp_risk_prob, 0.0, 1.0))
            weights["nlp_communication_risk"] = 0.25

        # Normalize weights if components are missing
        total_weight = sum(weights.values())
        if total_weight > 0:
            raw_score = sum(sub_scores[k] * (weights[k] / total_weight) for k in weights)
            composite_score = int(round(raw_score * 100))
        else:
            composite_score = 15  # Baseline minimal risk

        composite_score = max(0, min(100, composite_score))

        # Risk Tiers
        if composite_score >= 80:
            tier = "CRITICAL"
        elif composite_score >= 60:
            tier = "HIGH"
        elif composite_score >= 35:
            tier = "MEDIUM"
        else:
            tier = "LOW"

        return {
            "composite_score": composite_score,
            "risk_tier": tier,
            "contributing_signals": sub_scores,
            "signal_weights": {k: round(v / (total_weight or 1.0), 3) for k, v in weights.items()},
            "methodology": "Multi-Signal Business Composite Index (Transparent Weighted Scoring)",
        }

    @staticmethod
    def generate_recommendations(
        composite_score: int,
        invoice_late_prob: Optional[float] = None,
        retail_inactivity_prob: Optional[float] = None,
        cancellation_rate: float = 0.0,
        recency_days: float = 0.0,
        monetary: float = 0.0,
        exposure_amount: float = 0.0,
        nlp_risk_prob: Optional[float] = None,
    ) -> List[Dict[str, str]]:
        """
        Produces explainable, deterministic recommendations tied to specific risk drivers.
        """
        recs: List[Dict[str, str]] = []

        # Payment risk triggers
        if invoice_late_prob is not None and invoice_late_prob >= 0.70:
            recs.append({
                "priority": "P1 - High Priority",
                "action": "Accelerate Receivables Follow-Up",
                "trigger": f"Late-payment probability is {invoice_late_prob:.1%}",
                "detail": "Dispatch early automated reminder 5 days prior to due date and notify assigned collections rep.",
            })
        elif invoice_late_prob is not None and invoice_late_prob >= 0.40:
            recs.append({
                "priority": "P3 - Standard Monitoring",
                "action": "Schedule Standard Payment Reminder",
                "trigger": f"Moderate late-payment probability ({invoice_late_prob:.1%})",
                "detail": "Verify invoice receipt and ensure customer payment terms are acknowledged.",
            })

        # Dormancy / Inactivity triggers
        if retail_inactivity_prob is not None and retail_inactivity_prob >= 0.70 and monetary >= 1000:
            recs.append({
                "priority": "P1 - High Priority",
                "action": "Executive Retention Intervention",
                "trigger": f"High value customer (${monetary:,.0f}) with {retail_inactivity_prob:.1%} dormancy probability",
                "detail": "Assign strategic account manager to deliver personalized discount campaign and review account health.",
            })
        elif recency_days >= 90:
            recs.append({
                "priority": "P2 - Targeted Outreach",
                "action": "Re-engagement Campaign",
                "trigger": f"Inactive for {int(recency_days)} days",
                "detail": "Enroll customer in automated win-back workflow with top catalog reorder recommendations.",
            })

        # Operational friction / cancellation triggers
        if cancellation_rate >= 0.15:
            recs.append({
                "priority": "P2 - Operational Review",
                "action": "Audit Order Fulfillment & Quality",
                "trigger": f"Elevated cancellation rate ({cancellation_rate:.1%})",
                "detail": "Review return reasons and inspect stock items frequently ordered by this account.",
            })

        # Financial exposure triggers
        if exposure_amount >= 50_000:
            recs.append({
                "priority": "P1 - Credit Risk",
                "action": "Review Credit Line & Collateral",
                "trigger": f"High financial exposure (${exposure_amount:,.0f})",
                "detail": "Hold temporary credit expansion until current receivables balance is reduced below policy threshold.",
            })

        # NLP communication triggers
        if nlp_risk_prob is not None and nlp_risk_prob >= 0.60:
            recs.append({
                "priority": "P1 - Communication Escalation",
                "action": "Senior Credit Controller Review",
                "trigger": f"Communication analysis indicates elevated dispute risk ({nlp_risk_prob:.1%})",
                "detail": "Customer correspondence exhibits dispute signals or liquidity strain. Escalate to dispute resolution team.",
            })

        if not recs:
            recs.append({
                "priority": "P4 - Routine",
                "action": "Maintain Standard Terms",
                "trigger": "All operational signals within normal boundaries",
                "detail": "Continue routine invoicing and standard quarterly account monitoring.",
            })

        return recs


def np_clip(val: float, low: float, high: float) -> float:
    """Helper to clip float value without external dependencies."""
    return max(low, min(high, float(val)))
