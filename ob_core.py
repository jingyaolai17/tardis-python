from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Literal, Tuple, Dict
from sortedcontainers import SortedDict
import math

Side = Literal["bid", "ask"]

@dataclass
class Update:
    side: Side
    price: float
    size: float  # 0 => remove level on some venues


class BookLadder:
    """
    Minimal top-N order book with fast best bid/ask and volume lookups.
    Uses SortedDict:
      - asks: ascending price
      - bids: descending price via negative key
    """

    def __init__(self, top_n: int = 25):
        self.top_n = top_n
        self.asks = SortedDict()  # price -> size
        self.bids = SortedDict()  # -price -> size  (so .peekitem(0) is best bid)

    # ------------------ internal helpers ------------------

    def _insert(self, sd: SortedDict, key, val: float):
        if val is None or math.isnan(val):
            return
        if val <= 0:
            if key in sd:
                del sd[key]
        else:
            sd[key] = val

    def _trim(self):
        # keep only top-N levels on each side
        while len(self.asks) > self.top_n:
            self.asks.popitem(index=-1)   # drop worst ask (highest price)
        while len(self.bids) > self.top_n:
            self.bids.popitem(index=-1)   # drop worst bid (lowest price due to negative key)

    # ------------------ public API ------------------

    def apply_updates(self, updates: Iterable[Update]):
        for u in updates:
            if u.side == "ask":
                self._insert(self.asks, u.price, u.size)
            else:
                self._insert(self.bids, -u.price, u.size)
        self._trim()

    def best_ask(self) -> Tuple[float, float] | Tuple[None, None]:
        if not self.asks:
            return (None, None)
        p, s = self.asks.peekitem(0)
        return (p, s)

    def best_bid(self) -> Tuple[float, float] | Tuple[None, None]:
        if not self.bids:
            return (None, None)
        nkey, s = self.bids.peekitem(0)
        return (-nkey, s)

    def spread_mid(self) -> Tuple[float | None, float | None]:
        ba, sa = self.best_bid(), self.best_ask()
        if ba[0] is None or sa[0] is None:
            return (None, None)
        spread = sa[0] - ba[0]
        mid = (sa[0] + ba[0]) * 0.5
        return spread, mid

    def microprice(self, depth: int = 1) -> float | None:
        """
        Microprice = (α*best_ask + β*best_bid) / (α+β) with α,β taken from sizes.
        We extend α,β to top-k volumes (k = depth). If insufficient depth, we shrink.
        """
        if not self.asks or not self.bids:
            return None

        # Collect top-k buckets
        k = min(depth, len(self.asks), len(self.bids))
        asks = list(self.asks.items())[:k]             # [(price, size), ...] ascending
        bids = [(-p, s) for p, s in list(self.bids.items())[:k]]  # restore positive price

        # weights
        a_w = sum(max(s, 0.0) for _, s in asks)
        b_w = sum(max(s, 0.0) for _, s in bids)
        if a_w + b_w == 0:
            return None

        a_px = sum(p * max(s, 0.0) for p, s in asks)
        b_px = sum(p * max(s, 0.0) for p, s in bids)

        return (a_px + b_px) / (a_w + b_w)

    def queue_imbalance(self) -> float | None:
        """
        QI = bid_size_1 / (bid_size_1 + ask_size_1)
        """
        bb = self.best_bid()
        ba = self.best_ask()
        if bb[1] is None or ba[1] is None:
            return None
        tot = bb[1] + ba[1]
        if tot <= 0:
            return None
        return bb[1] / tot


# parsing helpers

def extract_wide_updates(row: dict) -> List[Update]:
    """
    Extract updates from a *wide* row:
      asks[0].price, asks[0].amount, ..., bids[0].price, bids[0].amount, ...
    Returns a list of Update(side, price, size).
    """
    ups: List[Update] = []

    for k, v in row.items():
        # we only care about .price keys; we'll pair with .amount by index
        if not isinstance(k, str):
            continue
        if k.startswith("asks[") and k.endswith("].price"):
            idx = k[k.find("[")+1:k.find("]")]
            size_key = f"asks[{idx}].amount"
            price = float(v) if v == v else None
            size  = float(row.get(size_key, float("nan")))
            if price is not None and size == size:  # not NaN
                ups.append(Update("ask", price, size))
        elif k.startswith("bids[") and k.endswith("].price"):
            idx = k[k.find("[")+1:k.find("]")]
            size_key = f"bids[{idx}].amount"
            price = float(v) if v == v else None
            size  = float(row.get(size_key, float("nan")))
            if price is not None and size == size:
                ups.append(Update("bid", price, size))
    return ups


def extract_long_update(row: dict) -> List[Update]:
    """
    Extract from "long" row format with columns:
      side, price, amount (or size)
    Returns [single Update] or [] if columns are missing.
    """
    low = {str(k).lower(): v for k, v in row.items()}

    side = low.get("side") or low.get("s")
    price = low.get("price") or low.get("p")
    size  = low.get("amount") or low.get("size") or low.get("q")

    if side is None or price is None or size is None:
        return []

    s = str(side).lower()
    if s in ("buy", "bid", "b", "1"):
        side = "bid"
    elif s in ("sell", "ask", "a", "2"):
        side = "ask"
    else:
        return []

    try:
        price = float(price)
        size  = float(size)
    except Exception:
        return []

    return [Update(side, price, size)]