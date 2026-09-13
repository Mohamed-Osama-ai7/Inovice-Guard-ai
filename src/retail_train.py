import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
import xgboost as xgb
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

def temporal_split(df: pd.DataFrame, date_col: str = 'snapshot_date'):
    """
    Splits the dataset into Train, Validation, Test, and OOT based on chronological order.
    Ensures max(train) < min(val) < min(test) < min(oot).
    """
    df = df.sort_values(by=date_col).copy()
    dates = np.sort(df[date_col].unique())
    n = len(dates)
    
    if n < 4:
        raise ValueError("Not enough snapshot dates for a proper temporal split.")
        
    train_dates = dates[:int(n*0.5)]
    val_dates = dates[int(n*0.5):int(n*0.7)]
    test_dates = dates[int(n*0.7):int(n*0.85)]
    oot_dates = dates[int(n*0.85):]
    
    splits = {
        'train': df[df[date_col].isin(train_dates)],
        'val': df[df[date_col].isin(val_dates)],
        'test': df[df[date_col].isin(test_dates)],
        'oot': df[df[date_col].isin(oot_dates)]
    }
    
    print("Temporal Split Details:")
    for k, v in splits.items():
        if not v.empty:
            print(f"  {k.upper()}: {v[date_col].min().date()} to {v[date_col].max().date()} ({len(v)} samples)")
        
    return splits

def evaluate_classification(y_true, y_pred, y_prob, name="Model"):
    metrics = {
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'PR-AUC': average_precision_score(y_true, y_prob),
        'ROC-AUC': roc_auc_score(y_true, y_prob)
    }
    print(f"[{name}] " + ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
    return metrics

def evaluate_regression(y_true, y_pred, name="Model"):
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred)
    }
    print(f"[{name}] " + ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
    return metrics

def train_classification_model(X_train, y_train, X_val, y_val, task_name="repurchase"):
    print(f"--- Training models for {task_name} ---")
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'HistGradientBoosting': HistGradientBoostingClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
    }
    
    best_model = None
    best_score = -1
    best_name = ""
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        score = average_precision_score(y_val, y_prob) # Optimize for PR-AUC
        
        print(f"{name} PR-AUC on val: {score:.4f}")
        if score > best_score:
            best_score = score
            best_model = model
            best_name = name
            
    print(f"Best model for {task_name}: {best_name} with PR-AUC: {best_score:.4f}")
    
    # Save model
    joblib.dump(best_model, MODELS_DIR / f"retail_{task_name}_model.pkl")
    return best_model, best_name

def train_regression_model(X_train, y_train, X_val, y_val, task_name="future_revenue"):
    print(f"--- Training models for {task_name} ---")
    models = {
        'Ridge': Ridge(random_state=42),
        'RandomForest': RandomForestRegressor(random_state=42, n_jobs=-1),
        'HistGradientBoosting': HistGradientBoostingRegressor(random_state=42),
        'XGBoost': xgb.XGBRegressor(random_state=42, n_jobs=-1)
    }
    
    best_model = None
    best_score = float('inf') # Lower is better for MAE
    best_name = ""
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        score = mean_absolute_error(y_val, y_pred) # Optimize for MAE
        
        print(f"{name} MAE on val: {score:.4f}")
        if score < best_score:
            best_score = score
            best_model = model
            best_name = name
            
    print(f"Best model for {task_name}: {best_name} with MAE: {best_score:.4f}")
    
    # Save model
    joblib.dump(best_model, MODELS_DIR / f"retail_{task_name}_model.pkl")
    return best_model, best_name
