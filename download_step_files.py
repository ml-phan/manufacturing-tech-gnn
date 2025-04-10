import boto3
import os
import re
import time
import pandas as pd

from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv, find_dotenv
from functools import partial
from pathlib import Path
env_file = find_dotenv()
load_dotenv(env_file)
# === CONFIGURATION ===
INPUT_CSV = "./data/item_data.csv"  # Update to your CSV path if needed
SAVE_DIR = "./data/step_files/"  # Directory to save downloaded files
FAILED_LOG_CSV = "./failed_downloads.csv"
MAX_WORKERS = 1  # Number of threads
FAIL_NUMBER = 0
# TIMEOUT = 10      # Optional: request timeout (in seconds)
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
# Load dataframe with item data
data_na = pd.read_csv(INPUT_CSV)
# downloads = data_na[['item_id', 'download_file_url']].dropna()

# Create file names
# def get_filename(row):
#     url = row['download_file_url']
#     if pd.isna(url):
#         return None
#
#     file_name = url.split('?')[0].split('/')[-1]
#     # match = re.search(r'\d+', row['item_id'])
#     # item_number = match.group() if match else "unknown"
#     return f"{row['item_id']}_{file_name}", url
#
# downloads['final_name_url'] = downloads.apply(get_filename, axis=1)
# downloads = downloads[downloads['final_name_url'].notnull()]
# downloads[['final_name', 'url']] = pd.DataFrame(downloads['final_name_url'].tolist(), index=downloads.index)

# Download function with logging
def download_file(s3, row):
    s3_url = row["download_file_url"]
    try:
        file_url = s3_url.split("?")[0]
        bucket_name = file_url.split("/")[2].split(".")[0]
        file_key = "/".join(file_url.split("/")[3:])
        file_name = file_key.split("/")[-1]
        item_id = row["item_id"]

        s3.download_file(
            bucket_name,
            file_key,
            f"./data/step_files/{item_id}_{file_name}"
        )
    except Exception as e:
        print(f"Download Failed: {row.url} — {e}")
    # After retries fail
    failed_downloads.append({'item_id': row.item_id, 'url': row.url, 'final_name': row.final_name})
    print(f"Number of failed downloads: {FAIL_NUMBER}", end="\r")

download_file_with_s3 = partial(download_file, s3)

# # Download files using ThreadPoolExecutor
# print(f"Starting downloads with {MAX_WORKERS} threads...")
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    executor.map(download_file_with_s3, [row for _, row in data_na.head(100).iterrows()])
#
# Save failed downloads
if failed_downloads:
    failed_df = pd.DataFrame(failed_downloads)
    failed_df.to_csv(FAILED_LOG_CSV, index=False)
    print(f"\nLogged {len(failed_downloads)} failed downloads to {FAILED_LOG_CSV}")
else:
    print("\nAll downloads completed successfully!")
