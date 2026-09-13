import pandas as pd
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.retail_train import (
    temporal_split,
    train_classification_model, evaluate_classification,
    train_regression_model, evaluate_regression
)
import joblib

DATA_PATH = ROOT / "data" / "processed" / "retail_customer_snapshots.csv"
FEATURES = [
    'recency_days', 'customer_age_days', 'frequency', 'monetary', 
    'avg_order_value', 'cancellation_rate', 'revenue_90d', 'orders_90d'
]

def format_markdown_table(data_dict, title):
    lines = [f"### {title}"]
    if not data_dict:
        return ""
    headers = ["Metric", "Train", "Validation", "Test", "OOT"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    
    # Get all unique metric keys
    keys = list(data_dict.get('Test', {}).keys())
    for k in keys:
        row = [k]
        for split in ["Train", "Validation", "Test", "OOT"]:
            val = data_dict.get(split, {}).get(k, "N/A")
            row.append(f"{val:.4f}" if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n\n"

def main():
    if not DATA_PATH.exists():
        print(f"Error: Processed data not found at {DATA_PATH}. Run src/retail_pipeline.py first.")
        sys.exit(1)
        
    print("Loading temporal dataset...")
    df = pd.read_csv(DATA_PATH)
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
    
    splits = temporal_split(df)
    
    X_train = splits['train'][FEATURES].fillna(0)
    X_val = splits['val'][FEATURES].fillna(0)
    X_test = splits['test'][FEATURES].fillna(0)
    X_oot = splits['oot'][FEATURES].fillna(0)
    
    report_content = "# Model Quality Report (Retail Intelligence)\n\n"
    report_content += "> Automatically generated during model training with temporal validation.\n\n"
    
    print("\n" + "="*50)
    print("TASK 1: REPURCHASE PREDICTION (Classification)")
    print("="*50)
    target_repurchase = 'repurchase_60d'
    y_train_c = splits['train'][target_repurchase]
    y_val_c = splits['val'][target_repurchase]
    y_test_c = splits['test'][target_repurchase]
    y_oot_c = splits['oot'][target_repurchase]
    
    model_c, name_c = train_classification_model(X_train, y_train_c, X_val, y_val_c, task_name="repurchase")
    
    c_metrics = {
        "Train": evaluate_classification(y_train_c, model_c.predict(X_train), model_c.predict_proba(X_train)[:, 1], name=name_c + "_Train"),
        "Validation": evaluate_classification(y_val_c, model_c.predict(X_val), model_c.predict_proba(X_val)[:, 1], name=name_c + "_Val"),
        "Test": evaluate_classification(y_test_c, model_c.predict(X_test), model_c.predict_proba(X_test)[:, 1], name=name_c + "_Test"),
        "OOT": evaluate_classification(y_oot_c, model_c.predict(X_oot), model_c.predict_proba(X_oot)[:, 1], name=name_c + "_OOT")
    }
    report_content += format_markdown_table(c_metrics, f"Repurchase Prediction Model: {name_c}")
    
    # Save Metadata for Repurchase Model
    c_metadata = {
        "model_name": name_c,
        "features": FEATURES,
        "metrics": c_metrics
    }
    (ROOT / "models" / "retail_repurchase_model_metadata.json").write_text(json.dumps(c_metadata, indent=2))
    
    print("\n" + "="*50)
    print("TASK 2: FUTURE REVENUE PREDICTION (Regression)")
    print("="*50)
    target_revenue = 'future_revenue_60d'
    y_train_r = splits['train'][target_revenue]
    y_val_r = splits['val'][target_revenue]
    y_test_r = splits['test'][target_revenue]
    y_oot_r = splits['oot'][target_revenue]
    
    model_r, name_r = train_regression_model(X_train, y_train_r, X_val, y_val_r, task_name="future_revenue")
    
    r_metrics = {
        "Train": evaluate_regression(y_train_r, model_r.predict(X_train), name=name_r + "_Train"),
        "Validation": evaluate_regression(y_val_r, model_r.predict(X_val), name=name_r + "_Val"),
        "Test": evaluate_regression(y_test_r, model_r.predict(X_test), name=name_r + "_Test"),
        "OOT": evaluate_regression(y_oot_r, model_r.predict(X_oot), name=name_r + "_OOT")
    }
    report_content += format_markdown_table(r_metrics, f"Future Revenue Prediction Model: {name_r}")
    
    # Save Metadata for Revenue Model
    r_metadata = {
        "model_name": name_r,
        "features": FEATURES,
        "metrics": r_metrics
    }
    (ROOT / "models" / "retail_future_revenue_model_metadata.json").write_text(json.dumps(r_metadata, indent=2))
    
    (ROOT / "docs" / "model_quality_report.md").write_text(report_content)
    
    # Save feature names for inference
    joblib.dump(FEATURES, ROOT / "models" / "retail_features.pkl")
    print("\nTraining complete and artifacts saved.")

if __name__ == "__main__":
    main()

