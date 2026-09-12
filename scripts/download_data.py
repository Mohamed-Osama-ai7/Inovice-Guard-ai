from pathlib import Path
import shutil
import zipfile

DATASET = "pradumn203/payment-date-prediction-for-invoices-dataset"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

try:
    import kagglehub
except ImportError as e:
    raise SystemExit("Install requirements first: pip install -r requirements.txt") from e

print(f"Downloading Kaggle dataset: {DATASET}")
path = Path(kagglehub.dataset_download(DATASET))
print("Downloaded to:", path)

if path.is_dir():
    for item in path.rglob('*'):
        if item.is_file():
            target = RAW / item.name
            shutil.copy2(item, target)
            print("Copied:", target)
elif path.suffix.lower() == ".zip":
    with zipfile.ZipFile(path) as z:
        z.extractall(RAW)
else:
    shutil.copy2(path, RAW / path.name)

print("Raw data directory:", RAW)
