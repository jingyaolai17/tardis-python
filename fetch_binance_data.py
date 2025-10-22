import os
import requests
import gzip
import pandas as pd
from datetime import datetime, timedelta

EXCHANGE = "binance"           
DATA_TYPE = "incremental_book_L2"  
SYMBOL = "BTCUSDT"
FROM_DATE = "2024-10-21"         
TO_DATE = "2024-10-22"         
API_KEY = os.getenv("TARDIS_API_KEY", "API key")

SAVE_PATH = f"data/{EXCHANGE}/{DATA_TYPE}/"
os.makedirs(SAVE_PATH, exist_ok=True)

def download_tardis_csv(exchange, data_type, symbol, date, api_key):
    year, month, day = date.split("-")
    url = f"https://datasets.tardis.dev/v1/{exchange}/{data_type}/{year}/{month}/{day}/{symbol}.csv.gz"
    headers = {"Authorization": f"Bearer {api_key}"}

    print(f"Downloading {url} ...")
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        filename = os.path.join(SAVE_PATH, f"{date}_{symbol}.csv.gz")
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"Saved {filename}")
        return filename
    else:
        print(f"Failed {date}: {response.status_code} - {response.text}")
        return None

def main():
    start = datetime.fromisoformat(FROM_DATE)
    end = datetime.fromisoformat(TO_DATE)

    current = start
    downloaded_files = []

    while current < end:
        date_str = current.strftime("%Y-%m-%d")
        f = download_tardis_csv(EXCHANGE, DATA_TYPE, SYMBOL, date_str, API_KEY)
        if f:
            downloaded_files.append(f)
        current += timedelta(days=1)

    # Preview first file if available
    if downloaded_files:
        print("\n Preview of first downloaded file:")
        df = pd.read_csv(gzip.open(downloaded_files[0]), nrows=5)
        print(df.head())

if __name__ == "__main__":
    main()