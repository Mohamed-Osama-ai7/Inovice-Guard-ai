import json
from pathlib import Path
from typing import Dict, Any, List

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

class ModelRegistry:
    """
    Enterprise Model Registry.
    Loads and validates metadata for deployed models.
    """
    
    @staticmethod
    def _load_metadata(filename: str) -> Dict[str, Any]:
        path = MODELS_DIR / filename
        if not path.exists():
            return {"error": f"Metadata file {filename} not found."}
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return {"error": f"Invalid JSON in {filename}."}
            
    @staticmethod
    def get_invoice_model_metadata() -> Dict[str, Any]:
        return ModelRegistry._load_metadata("classifier_metadata.json")
        
    @staticmethod
    def get_retail_repurchase_metadata() -> Dict[str, Any]:
        return ModelRegistry._load_metadata("retail_repurchase_model_metadata.json")
        
    @staticmethod
    def get_retail_revenue_metadata() -> Dict[str, Any]:
        return ModelRegistry._load_metadata("retail_future_revenue_model_metadata.json")
        
    @staticmethod
    def get_nlp_metadata() -> Dict[str, Any]:
        return ModelRegistry._load_metadata("nlp_payment_risk_metadata.json")

    @staticmethod
    def get_all_metrics() -> Dict[str, Any]:
        return {
            "invoice_payment_risk": ModelRegistry.get_invoice_model_metadata().get("metrics", {}),
            "retail_repurchase": ModelRegistry.get_retail_repurchase_metadata().get("metrics", {}),
            "retail_future_revenue": ModelRegistry.get_retail_revenue_metadata().get("metrics", {}),
            "nlp_payment_risk": ModelRegistry.get_nlp_metadata().get("metrics", {})
        }
