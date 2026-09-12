from __future__ import annotations

import traceback
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def main() -> None:
    print("InvoiceGuard AI - model verification diagnostic")
    print(f"Models directory: {MODELS}")
    print()

    artifacts = sorted(MODELS.glob("*"))
    if not artifacts:
        print("No files found under models/")
        return

    for path in artifacts:
        if path.suffix.lower() not in {".joblib", ".pkl"}:
            continue

        try:
            obj = joblib.load(path)
            print(f"PASS {path.name}")
            print(f"  loaded Python type: {type(obj).__module__}.{type(obj).__qualname__}")
        except Exception as exc:
            print(f"FAIL {path.name}")
            print(f"  exception: {type(exc).__name__}: {exc}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
