import pandas as pd
from typing import Dict, Any

class DataProfiler:
    @staticmethod
    def profile_dataset(df: pd.DataFrame, mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        Generates an enterprise data quality report.
        """
        report = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "duplicate_rows": int(df.duplicated().sum()),
            "missingness": {},
            "data_quality_warnings": []
        }
        
        # Calculate missingness
        for logical_col, actual_col in mapping.items():
            missing_count = int(df[actual_col].isna().sum())
            report["missingness"][logical_col] = {
                "count": missing_count,
                "percentage": round((missing_count / len(df)) * 100, 2)
            }
            if missing_count > 0:
                report["data_quality_warnings"].append(f"{logical_col} is missing {missing_count} values.")
                
        # Value logic validation
        if "invoice_amount" in mapping:
            amt_col = mapping["invoice_amount"]
            numeric_amounts = pd.to_numeric(df[amt_col], errors='coerce')
            invalid_amounts = numeric_amounts[numeric_amounts <= 0]
            if not invalid_amounts.empty:
                report["data_quality_warnings"].append(f"Found {len(invalid_amounts)} records with zero or negative invoice amount.")
                
        if "invoice_date" in mapping and "due_date" in mapping:
            inv_col = mapping["invoice_date"]
            due_col = mapping["due_date"]
            inv_dates = pd.to_datetime(df[inv_col], errors='coerce')
            due_dates = pd.to_datetime(df[due_col], errors='coerce')
            invalid_dates = (due_dates < inv_dates).sum()
            if invalid_dates > 0:
                report["data_quality_warnings"].append(f"Found {invalid_dates} records where due date is before invoice date.")
                
        # Calculate global quality score
        max_score = 100
        penalty = 0
        
        penalty += (report["duplicate_rows"] / len(df)) * 20  # up to 20 points penalty for duplicates
        
        total_missing = sum(m["count"] for m in report["missingness"].values())
        total_cells = len(df) * len(mapping)
        if total_cells > 0:
            penalty += (total_missing / total_cells) * 40 # up to 40 points penalty for missing values
            
        penalty += min(len(report["data_quality_warnings"]) * 5, 40) # up to 40 points penalty for logic warnings
        
        report["data_quality_score"] = max(0, min(100, int(max_score - penalty)))
        
        return report
