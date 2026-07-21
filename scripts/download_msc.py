"""Download and extract the Multi-Session Chat (MSC) dataset.

Usage:
    python scripts/download_msc.py

Downloads msc_v0.1.tar.gz (~30 MB) from ParlAI's public server and extracts
it into data/msc/. Fully open-source; no API keys required.
"""

import os
import sys
import tarfile
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import DATA_DIR, MSC_DIR, MSC_URL


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    archive_path = os.path.join(DATA_DIR, "msc_v0.1.tar.gz")

    if not os.path.exists(archive_path):
        print(f"Downloading MSC dataset from {MSC_URL} ...")
        urllib.request.urlretrieve(MSC_URL, archive_path)
        print(f"Saved to {archive_path}")
    else:
        print("Archive already downloaded.")

    print("Extracting ...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(DATA_DIR)

    # The archive extracts to data/msc/
    if os.path.isdir(MSC_DIR):
        print(f"Extracted to {MSC_DIR}")
    else:
        print(f"Extraction complete. Check {DATA_DIR} for the msc/ folder.")

    print("Done. Now run: python run_pipeline.py")


if __name__ == "__main__":
    main()
