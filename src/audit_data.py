from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import audit, choose_path, infer_map, read_table

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the invoice dataset and write data_audit.json")
    parser.add_argument("--source", choices=["auto", "kaggle", "demo"], default="auto")
    args = parser.parse_args()

    path = choose_path(args.source)
    raw = read_table(path)
    mapping = infer_map(raw)
    audit_out = audit(raw, mapping)

    output_path = REPORT_DIR / "data_audit.json"
    output_path.write_text(json.dumps(audit_out, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"source": str(path), "output": str(output_path), "rows": audit_out["rows"]}, indent=2))


if __name__ == "__main__":
    main()
