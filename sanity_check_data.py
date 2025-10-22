import os
import gzip
import pandas as pd
import matplotlib.pyplot as plt

EXCHANGE = "binance"
SYMBOL = "BTCUSDT"
DATES = ["2024-10-21", "2024-10-22"]

# Data types per Tardis.dev documentation
DATA_TYPES = [
    "trades",                
    "incremental_book_L2",  
    "book_snapshot_25"    
]

BASE_PATH = "data" 

def load_csv(exchange, data_type, symbol, date):
    """Load a Tardis CSV file for a given exchange/data_type/date/symbol."""
    path = f"{BASE_PATH}/{exchange}/{data_type}/{date}_{symbol}.csv.gz"
    print(f"\n Loading: {path}")
    if not os.path.exists(path):
        print(f" File not found: {path}")
        return None

    df = pd.read_csv(gzip.open(path))
    print(f"{data_type.upper()} — shape: {df.shape}")
    print(df.head(3))
    return df

def sanity_checks(df, label):
    """Basic integrity and timestamp checks."""
    print(f"\n [{label}] basic checks:")
    print("Columns:", list(df.columns))
    print("Nulls per column:\n", df.isnull().sum())

    if "timestamp" in df.columns:
        # Convert from microseconds to datetime properly
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="us", errors="coerce")
        diffs = df["timestamp"].diff().dt.total_seconds().dropna()
        neg = (diffs < 0).sum()
        print(f"Timestamps strictly increasing? {'✅' if neg == 0 else f'❌ ({neg} reversals)'}")
        print(f"Time span: {df['timestamp'].min()} → {df['timestamp'].max()}")
        print(f"Avg delta: {diffs.mean():.6f} seconds")
    else:
        print("⚠️ No 'timestamp' column detected!")

    # Safe describe block — handles both datetime and non-datetime data
    try:
        print(df.describe(include='all', datetime_is_numeric=True))
    except TypeError:
        print(df.describe(include='all'))


def visualize_activity(df, label):
    """Plot the number of events per minute."""
    if df is None or "timestamp" not in df.columns:
        return
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df.set_index("timestamp", inplace=True)
    per_min = df.resample("1min").size()
    per_min.plot(title=f"{label} — events per minute", figsize=(8, 3))
    plt.xlabel("Time (UTC)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def main():
    print(f"Running data sanity check for {EXCHANGE} {SYMBOL}...")
    for date in DATES:
        print(f"\n=== {date} ===")
        for data_type in DATA_TYPES:
            df = load_csv(EXCHANGE, data_type, SYMBOL, date)
            if df is not None:
                sanity_checks(df, f"{data_type}-{date}")
                visualize_activity(df, f"{data_type}-{date}")
    print("\n Stage 2 completed: Data sanity check finished.")


if __name__ == "__main__":
    main()