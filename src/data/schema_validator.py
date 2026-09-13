import pandas as pd
from typing import List, Dict, Any, Tuple
import re

def normalize_column_name(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(col).lower())

class InvoiceSchemaValidator:
    REQUIRED_LOGICAL_COLUMNS = {
        "invoice_date": ["invoicedate", "documentdate", "postingdate", "docdate", "issuedate", "billingdate"],
        "due_date": ["duedate", "netduedate", "paymentduedate"],
        "invoice_amount": ["invoiceamount", "amount", "documentamount", "netamount", "invoicevalue"],
        "customer": ["customerid", "customer", "customername", "buyer", "client", "customeraccount", "customerno", "customernumber"]
    }
    
    OPTIONAL_LOGICAL_COLUMNS = {
        "payment_date": ["paymentdate", "actualpaymentdate", "clearingdate", "paiddate"],
        "outstanding_amount": ["outstandingamount", "openamount", "balance"],
        "industry": ["industry", "sector", "vertical"],
        "company_size": ["companysize", "size"],
        "payment_method": ["paymentmethod", "paymenttype"],
        "customer_segment": ["customersegment", "segment", "tier"]
    }

    @staticmethod
    def detect_schema(df: pd.DataFrame) -> Tuple[Dict[str, str], List[str]]:
        """
        Maps dataframe columns to required and optional logical columns.
        Returns a tuple of (mapping_dict, missing_required_fields).
        """
        normalized_cols = {normalize_column_name(c): c for c in df.columns}
        mapping = {}
        missing = []
        
        # Detect required
        for logical_name, candidates in InvoiceSchemaValidator.REQUIRED_LOGICAL_COLUMNS.items():
            found = False
            for cand in candidates:
                if cand in normalized_cols:
                    mapping[logical_name] = normalized_cols[cand]
                    found = True
                    break
            if not found:
                missing.append(logical_name)
                
        # Detect optional
        for logical_name, candidates in InvoiceSchemaValidator.OPTIONAL_LOGICAL_COLUMNS.items():
            for cand in candidates:
                if cand in normalized_cols:
                    mapping[logical_name] = normalized_cols[cand]
                    break
                    
        return mapping, missing
