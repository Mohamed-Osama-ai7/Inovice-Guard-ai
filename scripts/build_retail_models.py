import pandas as pd
from pathlib import Path
import sys

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
    
    print("\n" + "="*50)
    print("TASK 1: REPURCHASE PREDICTION (Classification)")
    print("="*50)
    target_repurchase = 'repurchase_60d'
    y_train_c = splits['train'][target_repurchase]
    y_val_c = splits['val'][target_repurchase]
    y_test_c = splits['test'][target_repurchase]
    y_oot_c = splits['oot'][target_repurchase]
    
    model_c, name_c = train_classification_model(X_train, y_train_c, X_val, y_val_c, task_name="repurchase")
    
    print("\nEvaluating Repurchase Model (TEST):")
    evaluate_classification(y_test_c, model_c.predict(X_test), model_c.predict_proba(X_test)[:, 1], name=name_c)
    print("\nEvaluating Repurchase Model (OUT-OF-TIME):")
    evaluate_classification(y_oot_c, model_c.predict(X_oot), model_c.predict_proba(X_oot)[:, 1], name=name_c)
    
    print("\n" + "="*50)
    print("TASK 2: FUTURE REVENUE PREDICTION (Regression)")
    print("="*50)
    target_revenue = 'future_revenue_60d'
    y_train_r = splits['train'][target_revenue]
    y_val_r = splits['val'][target_revenue]
    y_test_r = splits['test'][target_revenue]
    y_oot_r = splits['oot'][target_revenue]
    
    model_r, name_r = train_regression_model(X_train, y_train_r, X_val, y_val_r, task_name="future_revenue")
    
    print("\nEvaluating Future Revenue Model (TEST):")
    evaluate_regression(y_test_r, model_r.predict(X_test), name=name_r)
    print("\nEvaluating Future Revenue Model (OUT-OF-TIME):")
    evaluate_regression(y_oot_r, model_r.predict(X_oot), name=name_r)
    
    # Save feature names for inference
    joblib.dump(FEATURES, ROOT / "models" / "retail_features.pkl")
    print("\nTraining complete and artifacts saved.")

if __name__ == "__main__":
    main()
