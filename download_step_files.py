import boto3
import os
import re
import time
import pandas as pd
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv, find_dotenv
from functools import partial
from pathlib import Path

env_file = find_dotenv()
load_dotenv(env_file)
# === CONFIGURATION ===
INPUT_CSV = "./data/item_data.csv"  # Update to your CSV path if needed
SAVE_DIR = r"E:/step_files"  # Directory to save downloaded files
FAILED_LOG_CSV = "./failed_downloads.csv"
MAX_WORKERS = 1  # Number of threads
FAIL_NUMBER = 0
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

failed_downloads = []
ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
SECRET_KEY = os.getenv('AWS_SECRET_KEY')
BUCKET_NAME_PARTS = os.getenv('AWS_BUCKET_NAME_PARTS_PROD')
AWS_BUCKET_NAME_ATTACHMENTS = os.getenv('AWS_BUCKET_NAME_ATTACHMENTS_PROD')
s3 = boto3.client(
    's3',
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY
)


# Download function with logging
def download_file(s3, row):
    try:
        s3_url = row["download_file_url"]
        file_url = s3_url.split("?")[0]
        bucket_name = file_url.split("/")[2].split(".")[0]
        file_key = "/".join(file_url.split("/")[3:])
        file_name = file_key.split("/")[-1]
        item_id = row["item_id"]
        s3.download_file(
            bucket_name,
            file_key,
            fr"E:/step_files_new/{item_id}_{file_name}"
        )
    except Exception as e:
        print(f"Download Failed: {row['download_file_url']} — {e}")
        failed_downloads.append(
            {'item_id': row["item_id"], 'url': row["download_file_url"]})
        print(f"Number of failed downloads: {FAIL_NUMBER}", end="\r")


def download_all_files(s3, data, num_rows=10):
    download_file_with_s3 = partial(download_file, s3)
    rows = [row for _, row in data.head(num_rows).iterrows()]

    # # Download files using ThreadPoolExecutor
    # print(f"Starting downloads with {MAX_WORKERS} threads...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(
            tqdm(
                executor.map(download_file_with_s3, rows),
                total=len(rows),
                desc="Downloading files",
                unit="file",
            )
        )
    # Save failed downloads
    if failed_downloads:
        failed_df = pd.DataFrame(failed_downloads)
        failed_df.to_csv(FAILED_LOG_CSV, index=False)
        print(
            f"\nLogged {len(failed_downloads)} failed downloads to {FAILED_LOG_CSV}")
    else:
        print("\nAll downloads completed successfully!")


def check_old_id():
    return [int(file.name.split("_")[0]) for file in Path(SAVE_DIR).glob("*.*")]


if __name__ == "__main__":
    data_na = pd.read_csv(INPUT_CSV)
    existing_ids = check_old_id()
    data_new = data_na[~data_na["item_id"].isin(existing_ids)]
    download_all_files(s3, data_new, len(data_new))
