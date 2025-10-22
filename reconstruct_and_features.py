# stage3_reconstruct_and_features.py
"""
Order-book reconstruction & microstructure features (OFI/QI/Microprice)

Inputs (under ./data/{exchange}/):
  - book_snapshot_25/YYYY-MM-DD_SYMBOL.csv.gz
  - incremental_book_L2/YYYY-MM-DD_SYMBOL.csv.gz
  - trade/YYYY-MM-DD_SYMBOL.csv.gz

Output:
  ./data/{exchange}/features/{YYYY-MM-DD}_{SYMBOL}_{freq}.parquet

Purpose:
  Rebuild the order book state over time from snapshot + incremental deltas,
  align trades, and compute features such as spread, microprice, queue imbalance,
  order-flow imbalance (OFI), and trade imbalance.
"""

from __future__ import annotations
import os, gzip
from datetime import timedelta
from typing import List, Tuple
import pandas as pd

from ob_core import BookLadder, extract_wide_updates, extract_long_update

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
EXCHANGE = "binance"
SYMBOL   = "BTCUSDT"
DATE     = "2024-10-21"   # change as needed
TOP_N    = 25             # number of levels to maintain in book
MICRO_K  = 5              # depth used for microprice weighting
FREQ     = "100ms"        # output feature sampling frequency
BASE     = "data"

# --------------------------------------------------
# PATH HELPERS
# --------------------------------------------------
def _p(kind: str, date=DATE) -> str:
    """
    Resolve dataset paths consistently with your folder layout.
    """
    kind_map = {
        "book_snapshot_25": "book_snapshot_25",
        "depth": "incremental_book_L2",
        "trade": "trade",
    }
    real_kind = kind_map.get(kind, kind)
    root = os.path.join(BASE, EXCHANGE, real_kind)
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, f"{date}_{SYMBOL}.csv.gz")

FEATURES_PATH = os.path.join(BASE, EXCHANGE, "features")
os.makedirs(FEATURES_PATH, exist_ok=True)


# --------------------------------------------------
# LOADERS
# --------------------------------------------------
def load_snapshot_wide() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """
    Load the first book snapshot (wide format: asks[i].*, bids[i].*).
    Returns (asks_df, bids_df, t0)
    """
    path = _p("book_snapshot_25")
    print(f"📘 Loading snapshot: {path}")
    df = pd.read_csv(gzip.open(path))
    df.columns = [c.lower() for c in df.columns]

    if "timestamp" not in df.columns:
        raise ValueError("Snapshot missing 'timestamp' column")

    row0 = df.iloc[0].to_dict()
    updates = extract_wide_updates(row0)

    asks, bids = [], []
    for u in updates:
        (asks if u.side == "ask" else bids).append((u.price, u.size))

    asks = pd.DataFrame(asks, columns=["price", "size"]).sort_values("price", ascending=True).reset_index(drop=True)
    bids = pd.DataFrame(bids, columns=["price", "size"]).sort_values("price", ascending=False).reset_index(drop=True)

    t0 = pd.to_datetime(df.loc[0, "timestamp"], unit="us", errors="coerce")
    return asks, bids, t0


def load_trades() -> pd.DataFrame:
    """
    Load executed trades: returns DataFrame(ts, side, price, amount)
    """
    path = _p("trades")
    print(f"💰 Loading trades: {path}")
    tdf = pd.read_csv(gzip.open(path))
    tdf.columns = [c.lower() for c in tdf.columns]
    required = {"timestamp", "side", "price", "amount"}
    if not required.issubset(tdf.columns):
        raise ValueError(f"Trade file missing columns: {tdf.columns.tolist()}")

    tdf["ts"] = pd.to_datetime(tdf["timestamp"], unit="us", errors="coerce")
    tdf["side"] = tdf["side"].astype(str).str.lower().map({"buy":"buy","sell":"sell"})
    return tdf[["ts", "side", "price", "amount"]].sort_values("ts").reset_index(drop=True)


def iterate_l2_deltas():
    """
    Generator over incremental L2 updates (wide or long format).
    Each yield: (ts, [Update, Update, ...])
    """
    path = _p("depth")
    print(f"📈 Loading incremental L2 deltas: {path}")
    for chunk in pd.read_csv(gzip.open(path), chunksize=50_000):
        chunk.columns = [c.lower() for c in chunk.columns]
        if "timestamp" not in chunk.columns:
            raise ValueError("L2 chunk missing timestamp column")

        chunk["ts"] = pd.to_datetime(chunk["timestamp"], unit="us", errors="coerce")
        chunk = chunk.sort_values("ts")

        for _, row in chunk.iterrows():
            d = row.to_dict()
            ups = extract_wide_updates(d)
            if not ups:
                ups = extract_long_update(d)
            if ups:
                yield row["ts"], ups


# --------------------------------------------------
# RECONSTRUCTION & FEATURES
# --------------------------------------------------
def build_feature_table() -> pd.DataFrame:
    # --- initialize from snapshot ---
    asks0, bids0, t0 = load_snapshot_wide()
    book = BookLadder(TOP_N)
    book.apply_updates([
        type("U", (), {"side": "ask", "price": p, "size": s}) for p, s in asks0.itertuples(index=False)
    ])
    book.apply_updates([
        type("U", (), {"side": "bid", "price": p, "size": s}) for p, s in bids0.itertuples(index=False)
    ])

    # --- trades preprocessing ---
    trades = load_trades()
    trades["bucket"] = trades["ts"].dt.floor(FREQ)
    trade_buckets = trades.groupby(["bucket", "side"])["amount"].sum().unstack(fill_value=0.0)
    for col in ("buy", "sell"):
        if col not in trade_buckets.columns:
            trade_buckets[col] = 0.0
    trade_buckets = trade_buckets.sort_index()

    # --- prepare depth iterator ---
    depth_iter = iterate_l2_deltas()
    try:
        ts_next, ups_next = next(depth_iter)
    except StopIteration:
        ts_next, ups_next = None, None

    # --- output grid ---
    end_ts = trades["ts"].max()
    grid = pd.date_range(t0.floor(FREQ), end_ts.ceil(FREQ), freq=FREQ)

    feat_rows = []
    prev_bb, prev_ba = book.best_bid(), book.best_ask()
    prev_sizes = (prev_bb[1], prev_ba[1])

    for t in grid:
        # apply all deltas up to current time
        while ts_next is not None and ts_next <= t:
            book.apply_updates(ups_next)
            try:
                ts_next, ups_next = next(depth_iter)
            except StopIteration:
                ts_next, ups_next = None, None
                break

        # current best levels
        bb = book.best_bid()
        ba = book.best_ask()
        spread, mid = book.spread_mid()
        micro = book.microprice(depth=MICRO_K)
        qi = book.queue_imbalance()

        # OFI (delta of top-queue sizes)
        ofi = None
        if all(x is not None for x in (*prev_sizes, bb[1], ba[1])):
            ofi = (bb[1] - prev_sizes[0]) - (ba[1] - prev_sizes[1])
            prev_sizes = (bb[1], ba[1])

        # trades in this bucket
        buys = float(trade_buckets.loc[t, "buy"]) if t in trade_buckets.index else 0.0
        sells = float(trade_buckets.loc[t, "sell"]) if t in trade_buckets.index else 0.0
        trade_imb = buys - sells

        feat_rows.append({
            "ts": t,
            "best_bid": bb[0],
            "best_ask": ba[0],
            "bid_size": bb[1],
            "ask_size": ba[1],
            "spread": spread,
            "mid": mid,
            "microprice": micro,
            "qi": qi,
            "ofi": ofi,
            "buy_vol": buys,
            "sell_vol": sells,
            "trade_imb": trade_imb,
        })

    feats = pd.DataFrame(feat_rows).set_index("ts")
    return feats


# --------------------------------------------------
# MAIN ENTRY
# --------------------------------------------------
def main():
    feats = build_feature_table()
    out_path = os.path.join(FEATURES_PATH, f"{DATE}_{SYMBOL}_{FREQ}.parquet")
    feats.to_parquet(out_path)
    print(f"✅ Features saved: {out_path}")
    print(feats.head(10))


if __name__ == "__main__":
    main()