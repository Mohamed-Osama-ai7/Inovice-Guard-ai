import pytest
import pandas as pd
from src.data.schema_validator import InvoiceSchemaValidator
from src.data.data_profiler import DataProfiler

def test_validate_invoice_schema_success():
    df = pd.DataFrame({
        "invoice_date": ["2010-12-01"],
        "due_date": ["2011-01-01"],
        "invoice_amount": [1000.0],
        "customer": ["CUST1"]
    })
    
    mapping, missing = InvoiceSchemaValidator.detect_schema(df)
    assert len(missing) == 0
    assert mapping["invoice_date"] == "invoice_date"

def test_validate_invoice_schema_missing_cols():
    df = pd.DataFrame({
        "invoice_date": ["2010-12-01"],
        "customer": ["CUST1"]
    })
    
    mapping, missing = InvoiceSchemaValidator.detect_schema(df)
    assert len(missing) == 2
    assert "due_date" in missing

def test_data_profiler():
    df = pd.DataFrame({
        "invoice_amount": [2.55, -1.0, 5.0, 10.0, 2.55],
        "customer": ["A", "B", "C", "D", "A"]
    })
    mapping, _ = InvoiceSchemaValidator.detect_schema(df)
    
    profile = DataProfiler.profile_dataset(df, mapping)
    assert "total_rows" in profile
    assert profile["total_rows"] == 5
    assert "total_columns" in profile
    assert profile["total_columns"] == 2
