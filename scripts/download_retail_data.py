import os
import urllib.request
from pathlib import Path

# URL for UCI Online Retail II dataset
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx"

def download_retail_data():
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    dest_path = raw_dir / "online_retail_II.xlsx"
    
    if dest_path.exists():
        print(f"Dataset already exists at: {dest_path}")
        return dest_path

    print(f"Downloading UCI Online Retail II dataset from {DATA_URL}...")
    try:
        urllib.request.urlretrieve(DATA_URL, dest_path)
        print(f"Successfully downloaded to: {dest_path}")
    except Exception as e:
        print(f"Error downloading the dataset: {e}")
        # Try kagglehub as fallback
        try:
            print("Attempting kagglehub fallback...")
            import kagglehub
            import shutil
            # The dataset might be under mathchi/online-retail-ii-data-set-from-ml-repository
            path = Path(kagglehub.dataset_download("mathchi/online-retail-ii-data-set-from-ml-repository"))
            for item in path.rglob('*.xlsx'):
                shutil.copy2(item, dest_path)
            print(f"Downloaded via Kaggle to: {dest_path}")
        except Exception as e2:
            print(f"Fallback failed: {e2}")

    return dest_path

if __name__ == "__main__":
    download_retail_data()
