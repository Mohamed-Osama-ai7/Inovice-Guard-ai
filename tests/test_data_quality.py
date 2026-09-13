import pytest
import pandas as pd
from src.data.schema_validator import validate_retail_schema
from src.data.data_profiler import profile_dataframe

def test_validate_retail_schema_success():
    df = pd.DataFrame({
        "Invoice": ["536365"],
        "StockCode": ["85123A"],
        "Description": ["WHITE HANGING HEART T-LIGHT HOLDER"],
        "Quantity": [6],
        "InvoiceDate": ["2010-12-01 08:26:00"],
        "Price": [2.55],
        "Customer ID": [17850.0],
        "Country": ["United Kingdom"]
    })
    
    is_valid, errors = validate_retail_schema(df)
    assert is_valid is True
    assert len(errors) == 0

def test_validate_retail_schema_missing_cols():
    df = pd.DataFrame({
        "Invoice": ["536365"],
        "StockCode": ["85123A"]
    })
    
    is_valid, errors = validate_retail_schema(df)
    assert is_valid is False
    assert len(errors) > 0

def test_data_profiler():
    df = pd.DataFrame({
        "Price": [2.55, -1.0, 5.0, 10.0, 2.55], # Includes negative and duplicates
        "Quantity": [6, 0, -5, 10, 6]
    })
    
    profile = profile_dataframe(df)
    assert "rows" in profile
    assert profile["rows"] == 5
    assert profile["columns"] == 2
    assert "memory_usage_kb" in profile
    assert "numerical_stats" in profile
    assert "Price" in profile["numerical_stats"]
