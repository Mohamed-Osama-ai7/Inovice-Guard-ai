import os
from pathlib import Path
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
from datetime import timedelta

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def load_retail_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """Loads the UCI Online Retail II dataset, combining both sheets if using the xlsx."""
    if file_path is None:
        file_path = RAW_DIR / "online_retail_II.xlsx"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Raw data file not found at {file_path}. Please run scripts/download_retail_data.py first.")
    
    print(f"Loading data from {file_path}...")
    if file_path.suffix.lower() == ".xlsx":
        # The dataset contains two sheets: "Year 2009-2010" and "Year 2010-2011"
        xls = pd.ExcelFile(file_path)
        dfs = []
        for sheet_name in xls.sheet_names:
            df_sheet = pd.read_excel(xls, sheet_name=sheet_name)
            dfs.append(df_sheet)
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(file_path)
    
    return df

def clean_retail_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw retail data.
    - Normalizes column names
    - Drops missing CustomerIDs
    - Flags cancellations
    - Removes invalid records
    """
    df = df.copy()
    
    # Normalize columns
    df.columns = [str(c).strip().replace(" ", "_").lower() for c in df.columns]
    
    # Rename standard columns if they exist
    col_mapping = {
        'customer_id': 'customer_id',
        'customerid': 'customer_id',
        'invoice': 'invoice_no',
        'invoiceno': 'invoice_no',
        'stockcode': 'stock_code',
        'description': 'description',
        'quantity': 'quantity',
        'invoicedate': 'invoice_date',
        'invoice_date': 'invoice_date',
        'price': 'price',
        'country': 'country'
    }
    df = df.rename(columns=col_mapping)
    
    # Parse dates
    df['invoice_date'] = pd.to_datetime(df['invoice_date'])
    
    # Drop rows without CustomerID for customer intelligence
    df = df.dropna(subset=['customer_id'])
    
    # Convert Customer ID to string to match InvoiceGuard format
    df['customer_id'] = df['customer_id'].astype(int).astype(str)
    
    # Flag cancellations (InvoiceNo starts with 'C')
    df['is_cancellation'] = df['invoice_no'].astype(str).str.startswith('C')
    
    # Remove records with zero or negative price
    df = df[df['price'] > 0]
    
    # Calculate revenue per line
    df['revenue'] = df['quantity'] * df['price']
    
    return df

def generate_customer_snapshots(df: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    """
    Generates point-in-time customer features (RFM, etc.) strictly using data strictly prior to snapshot_date.
    """
    history = df[df['invoice_date'] < snapshot_date].copy()
    
    if history.empty:
        return pd.DataFrame()
        
    # Aggregate to invoice level first
    invoice_level = history.groupby(['customer_id', 'invoice_no', 'invoice_date', 'is_cancellation']).agg(
        invoice_revenue=('revenue', 'sum'),
        items_count=('quantity', 'sum'),
        unique_products=('stock_code', 'nunique')
    ).reset_index()
    
    # Aggregate to customer level
    customer_stats = invoice_level.groupby('customer_id').agg(
        first_purchase_date=('invoice_date', 'min'),
        last_purchase_date=('invoice_date', 'max'),
        total_orders=('invoice_no', 'nunique'),
        total_revenue=('invoice_revenue', 'sum'),
        total_cancellations=('is_cancellation', 'sum')
    ).reset_index()
    
    customer_stats['snapshot_date'] = snapshot_date
    
    # Calculate RFM
    customer_stats['recency_days'] = (snapshot_date - customer_stats['last_purchase_date']).dt.days
    customer_stats['customer_age_days'] = (snapshot_date - customer_stats['first_purchase_date']).dt.days
    customer_stats['frequency'] = customer_stats['total_orders']
    customer_stats['monetary'] = customer_stats['total_revenue']
    customer_stats['avg_order_value'] = np.where(customer_stats['frequency'] > 0, 
                                                customer_stats['monetary'] / customer_stats['frequency'], 0)
    customer_stats['cancellation_rate'] = np.where(customer_stats['frequency'] > 0,
                                                  customer_stats['total_cancellations'] / customer_stats['frequency'], 0)
                                                  
    # Short-term behavior (e.g., last 90 days)
    recent_90d = snapshot_date - pd.Timedelta(days=90)
    history_90d = history[history['invoice_date'] >= recent_90d]
    
    recent_stats = history_90d.groupby('customer_id').agg(
        revenue_90d=('revenue', 'sum'),
        orders_90d=('invoice_no', 'nunique')
    ).reset_index()
    
    customer_stats = pd.merge(customer_stats, recent_stats, on='customer_id', how='left')
    customer_stats['revenue_90d'] = customer_stats['revenue_90d'].fillna(0)
    customer_stats['orders_90d'] = customer_stats['orders_90d'].fillna(0)
    
    return customer_stats

def generate_targets(df: pd.DataFrame, snapshot_date: pd.Timestamp, forward_days: int = 60) -> pd.DataFrame:
    """
    Generates target labels for customers based on behavior after snapshot_date.
    e.g., Will they repurchase in the next 60 days? What will be their revenue?
    """
    future_start = snapshot_date
    future_end = snapshot_date + pd.Timedelta(days=forward_days)
    
    future_data = df[(df['invoice_date'] >= future_start) & (df['invoice_date'] < future_end)].copy()
    
    # Aggregate future behavior
    targets = future_data.groupby('customer_id').agg(
        future_revenue=('revenue', 'sum'),
        future_orders=('invoice_no', 'nunique')
    ).reset_index()
    
    targets = targets.rename(columns={
        'future_revenue': f'future_revenue_{forward_days}d',
        'future_orders': f'future_orders_{forward_days}d'
    })
    
    targets[f'repurchase_{forward_days}d'] = (targets[f'future_orders_{forward_days}d'] > 0).astype(int)
    
    return targets

def build_temporal_dataset(df: pd.DataFrame, snapshot_dates: List[pd.Timestamp], forward_days: int = 60) -> pd.DataFrame:
    """
    Builds a full dataset with features and targets across multiple snapshot dates.
    This guarantees no temporal leakage.
    """
    all_snapshots = []
    
    for snapshot in snapshot_dates:
        print(f"Building snapshot for {snapshot.date()}...")
        features = generate_customer_snapshots(df, snapshot)
        if features.empty:
            continue
            
        targets = generate_targets(df, snapshot, forward_days)
        
        # Merge features and targets (left join to keep customers who didn't repurchase)
        combined = pd.merge(features, targets, on='customer_id', how='left')
        
        # Fill NaN targets with 0 (no future activity)
        target_cols = [f'future_revenue_{forward_days}d', f'future_orders_{forward_days}d', f'repurchase_{forward_days}d']
        for col in target_cols:
            if col in combined.columns:
                combined[col] = combined[col].fillna(0)
                
        all_snapshots.append(combined)
        
    return pd.concat(all_snapshots, ignore_index=True)

if __name__ == "__main__":
    df = load_retail_data()
    clean_df = clean_retail_data(df)
    
    # Find dataset date range
    min_date = clean_df['invoice_date'].min()
    max_date = clean_df['invoice_date'].max()
    print(f"Dataset ranges from {min_date} to {max_date}")
    
    # Create multiple snapshot dates for training
    # For example, 1st of every month starting from month 6
    snapshot_dates = pd.date_range(start=min_date + pd.Timedelta(days=180), 
                                   end=max_date - pd.Timedelta(days=60), 
                                   freq='MS').tolist()
                                   
    final_dataset = build_temporal_dataset(clean_df, snapshot_dates, forward_days=60)
    
    out_path = PROCESSED_DIR / "retail_customer_snapshots.csv"
    final_dataset.to_csv(out_path, index=False)
    print(f"Saved processed temporal dataset to {out_path}")
