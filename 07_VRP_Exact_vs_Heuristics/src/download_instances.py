import os
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# List of (set_name, instance_name, github_url)
# afourmy/pyVRP repo has Augerat A, B, P instances.
BASE_URL = "https://raw.githubusercontent.com/afourmy/pyVRP/master/data"

INSTANCES = [
    "P-n16-k8.vrp",
    "P-n19-k2.vrp",
    "P-n22-k8.vrp",
    "A-n32-k5.vrp",
    "A-n33-k5.vrp",
    "A-n44-k6.vrp",
    "A-n60-k9.vrp",
    "A-n80-k10.vrp",
]

def download_cvrplib():
    for inst_file in INSTANCES:
        url = f"{BASE_URL}/{inst_file}"
        out_path = RAW_DIR / inst_file
        
        if not out_path.exists():
            print(f"Downloading {inst_file}...")
            try:
                urllib.request.urlretrieve(url, out_path)
            except Exception as e:
                print(f"Failed to download {inst_file}: {e}")
        else:
            print(f"{inst_file} already exists.")

if __name__ == "__main__":
    download_cvrplib()
