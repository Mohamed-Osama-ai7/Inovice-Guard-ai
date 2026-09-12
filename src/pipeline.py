from __future__ import annotations

import json, os, re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DEMO_DIR = ROOT / "data" / "demo"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(exist_ok=True); REPORT_DIR.mkdir(exist_ok=True)


def norm(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())

@dataclass
class Map:
    invoice_date: str
    due_date: str
    payment_date: str
    amount: Optional[str]
    customer: Optional[str]

CANDS = {
    "invoice_date": ["invoicedate","documentdate","postingdate","docdate","issuedate","billingdate"],
    "due_date": ["duedate","netduedate","paymentduedate"],
    "payment_date": ["paymentdate","actualpaymentdate","clearingdate","paiddate"],
    "amount": ["invoiceamount","amount","documentamount","netamount","invoicevalue"],
    "customer": ["customerid","customer","customername","buyer","client","customeraccount","customerno","customernumber"],
}

def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower()==".csv":
        for enc in ("utf-8","latin1","cp1252"):
            try: return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError: pass
        raise ValueError(f"Could not decode {path}")
    return pd.read_excel(path)

def candidate_files(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob('*') if p.suffix.lower() in {'.csv','.xlsx','.xls'}])

def choose_path(source: str = "auto") -> Path:
    override = os.getenv("INVOICE_DATA_FILE")
    if override: return Path(override)
    if source == "demo":
        p = DEMO_DIR / "demo_invoices.csv"
        if not p.exists(): raise FileNotFoundError("Run: python scripts/generate_demo.py")
        return p
    if source == "kaggle":
        raw_files = candidate_files(RAW_DIR)
        if not raw_files:
            raise FileNotFoundError("No raw Kaggle file. Run scripts/download_data.py")
        return sorted(raw_files, key=lambda p: (0 if any(k in p.name.lower() for k in ('invoice','payment','date','open')) else 1, len(p.name)))[0]
    raw_files = candidate_files(RAW_DIR)
    if raw_files:
        return sorted(raw_files, key=lambda p: (0 if any(k in p.name.lower() for k in ('invoice','payment','date','open')) else 1, len(p.name)))[0]
    demo_file = DEMO_DIR / "demo_invoices.csv"
    if demo_file.exists():
        return demo_file
    raise FileNotFoundError("No tabular data file found. Run python scripts/generate_demo.py or python scripts/download_data.py")

def find_col(cols: Iterable[str], cands: list[str], required: bool) -> Optional[str]:
    normalized={norm(c):c for c in cols}
    for c in cands:
        if c in normalized: return normalized[c]
    for nc, original in normalized.items():
        if any(c in nc for c in cands): return original
    if required: raise KeyError(f"Missing required column. Need one of {cands}. Available={list(cols)}")
    return None

def infer_map(df: pd.DataFrame) -> Map:
    return Map(*(find_col(df.columns, CANDS[k], k not in ('amount','customer')) for k in CANDS))

def clean_amount(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s): return pd.to_numeric(s, errors='coerce')
    return pd.to_numeric(s.astype(str).str.replace(r'[^0-9.\-]','',regex=True), errors='coerce')

def build_supervised(raw: pd.DataFrame, m: Map) -> pd.DataFrame:
    x=raw.copy().drop_duplicates()
    for c in [m.invoice_date,m.due_date,m.payment_date]: x[c]=pd.to_datetime(x[c], errors='coerce')
    x=x.dropna(subset=[m.invoice_date,m.due_date,m.payment_date]).sort_values(m.invoice_date).reset_index(drop=True)
    x['invoice_amount_clean']=clean_amount(x[m.amount]) if m.amount else np.nan
    x['delay_days']=(x[m.payment_date]-x[m.due_date]).dt.total_seconds()/86400
    x['late_payment']=(x['delay_days']>0).astype(int)
    x['delay_days_positive']=x['delay_days'].clip(lower=0)
    return x

def add_features(df: pd.DataFrame, m: Map) -> tuple[pd.DataFrame,list[str],list[str],list[str]]:
    x=df.copy().sort_values(m.invoice_date).reset_index(drop=True)
    # Safe temporal features.
    x['invoice_year']=x[m.invoice_date].dt.year
    x['invoice_month']=x[m.invoice_date].dt.month
    x['invoice_quarter']=x[m.invoice_date].dt.quarter
    x['invoice_dayofweek']=x[m.invoice_date].dt.dayofweek
    x['days_to_due']=(x[m.due_date]-x[m.invoice_date]).dt.total_seconds()/86400
    x['amount_log1p']=np.log1p(x['invoice_amount_clean'].clip(lower=0))
    if m.customer:
        key=x[m.customer].astype(str).fillna('UNKNOWN')
        x['customer_seen_before']=key.groupby(key).cumcount()
        prior_late=x.groupby(key,sort=False)['late_payment'].transform(lambda s:s.cumsum().shift(1)).fillna(0)
        prior_cnt=x['customer_seen_before'].astype(float)
        prior_sum=x.groupby(key,sort=False)['delay_days_positive'].transform(lambda s:s.cumsum().shift(1)).fillna(0)
        x['prior_late_count']=prior_late
        x['prior_late_ratio']=np.where(prior_cnt>0,prior_late/prior_cnt,0)
        x['prior_avg_delay']=np.where(prior_cnt>0,prior_sum/prior_cnt,0)
        x['customer_is_new']=(prior_cnt==0).astype(int)
    else:
        x['customer_seen_before']=0; x['prior_late_count']=0.; x['prior_late_ratio']=0.; x['prior_avg_delay']=0.; x['customer_is_new']=1

    numeric=[c for c in ['invoice_amount_clean','amount_log1p','days_to_due','invoice_year','invoice_month','invoice_quarter','invoice_dayofweek','customer_seen_before','customer_is_new','prior_late_count','prior_late_ratio','prior_avg_delay','outstanding_amount'] if c in x.columns]
    categorical=[c for c in ['industry','company_size','payment_method','customer_segment','customer_type','sales_org','currency'] if c in x.columns]
    # Never include payment_date/derived target or post-payment fields.
    leakage={'late_payment','delay_days','delay_days_positive',m.payment_date}
    numeric=[c for c in numeric if c not in leakage]
    categorical=[c for c in categorical if c not in leakage]
    return x,numeric,categorical,numeric+categorical

def make_preprocessor(num: list[str], cat: list[str]):
    return ColumnTransformer([
        ('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),
        ('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cat),
    ], remainder='drop')

def audit(df: pd.DataFrame, m: Map) -> dict:
    return {
        'rows':int(len(df)),'columns':int(df.shape[1]),'duplicate_rows':int(df.duplicated().sum()),
        'missing_by_column':df.isna().sum().sort_values(ascending=False).to_dict(),
        'date_parseable':{c:int(pd.to_datetime(df[c],errors='coerce').notna().sum()) for c in [m.invoice_date,m.due_date,m.payment_date]},
        'inferred_columns':m.__dict__,
    }
