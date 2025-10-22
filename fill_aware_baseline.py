"""
Fill-Aware OFI-Based Maker-Leaning Strategy
Goal:
  Extend the baseline OFI strategy with fill-aware logic:
  - Queue-aware placement
  - Maker/taker simulation
  - Cancels/partials/slippage adjustments
  - Re-run attribution and comparison

Outputs:
  - reports/ofi_pnl_plot.png
  - reports/ofi_fillaware_comparison.png
  - reports/attribution_fillaware.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FEATURE_PATH = "data/binance/features/2024-10-21_BTCUSDT_100ms.parquet"
REPORT_DIR = "Fill-Aware: reports"
FEE_BPS = 0.0001     # 1 basis point per side
BAR_FREQ = 100       # milliseconds, for annualization approximation
MAKER_REBATE = 0.00002   # 2 bps rebate
TAKER_FEE = 0.0004       # 4 bps taker cost
SLIPPAGE_TICK = 0.0001   # 1 tick ≈ 0.01%

# HELPER FUNCTIONS
def compute_metrics(df: pd.DataFrame, pnl_col: str = "after_cost") -> pd.DataFrame:
    """
    Extended attribution metrics including microstructure and performance diagnostics.
    """

    cum_pnl = df[pnl_col].cumsum()
    total_pnl = df[pnl_col].sum()
    mean_pnl = df[pnl_col].mean()
    std_pnl = df[pnl_col].std()
    sharpe = np.nan if std_pnl == 0 else mean_pnl / std_pnl * np.sqrt(24 * 60 * (1000 / BAR_FREQ))

    # Spread capture (bps)
    if "spread" in df.columns and df["spread"].sum() != 0:
        spread_capture_bps = (df["raw_pnl"].sum() / df["spread"].sum()) * 1e4  # bps
    else:
        spread_capture_bps = np.nan

    # Selection loss at various horizons (bps)
    # Forward midprice changes at 10/50/250 ms horizons
    df["fwd_10ms"] = df["mid"].shift(-1) - df["mid"]
    df["fwd_50ms"] = df["mid"].shift(-5) - df["mid"]
    df["fwd_250ms"] = df["mid"].shift(-25) - df["mid"]

    def selection_loss(col):
        return -np.nanmean(df["signal"] * df[col]) * 1e4  # in bps

    sel_loss_10 = selection_loss("fwd_10ms")
    sel_loss_50 = selection_loss("fwd_50ms")
    sel_loss_250 = selection_loss("fwd_250ms")

    # Hedge slippage / carry (bps)
    # Proxy: standard deviation of microprice drift relative to mid
    if "microprice" in df.columns:
        hedge_slip = np.nanstd(df["microprice"].diff() / df["mid"]) * 1e4
    else:
        hedge_slip = np.nan

    # Fill rate (% of quotes)
    if "fill_prob" in df.columns:
        fill_rate = df["fill_prob"].mean() * 100
    else:
        fill_rate = np.nan

    # Inventory breaches
    # Simulate cumulative position and count threshold exceedances
    df["pos"] = df["signal"].cumsum()
    inventory_breaches = (df["pos"].abs() > 10).sum()  # arbitrary threshold = 10 units

    # --- Worst day PnL / Max Drawdown ---
    roll_max = cum_pnl.cummax()
    drawdown = cum_pnl - roll_max
    max_dd = drawdown.min()
    worst_day_pnl = df.groupby(df.index.floor("1D"))[pnl_col].sum().min()

    hit_ratio = (df[pnl_col] > 0).mean()

    return pd.DataFrame({
        "Metric": [
            "After-cost Sharpe",
            "Spread Capture (bps)",
            "Selection Loss 10ms (bps)",
            "Selection Loss 50ms (bps)",
            "Selection Loss 250ms (bps)",
            "Hedge Slippage / Carry (bps)",
            "Fill Rate (% of quotes)",
            "Inventory Breaches (#)",
            "Worst Day PnL",
            "Max Drawdown",
            "Hit Ratio"
        ],
        "Value": [
            sharpe,
            spread_capture_bps,
            sel_loss_10,
            sel_loss_50,
            sel_loss_250,
            hedge_slip,
            fill_rate,
            inventory_breaches,
            worst_day_pnl,
            max_dd,
            hit_ratio
        ]
    })


def plot_cum_pnl(df: pd.DataFrame, output_path: str):
    """Plot cumulative ideal after-cost PnL."""
    plt.figure(figsize=(10, 5))
    df["after_cost"].cumsum().plot()
    plt.title("Baseline: Cumulative After-Cost PnL (OFI Signal)")
    plt.xlabel("Time")
    plt.ylabel("PnL (quote currency)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_fillaware_comparison(df: pd.DataFrame, output_path: str):
    """Plot fill-aware vs ideal PnL curves."""
    plt.figure(figsize=(10, 5))
    df["after_cost"].cumsum().plot(label="After-Cost (Ideal)")
    df["realized_pnl"].cumsum().plot(label="Fill-Aware (Realized)")
    plt.title("Cumulative PnL Comparison (OFI Fill-Aware Execution)")
    plt.xlabel("Time")
    plt.ylabel("PnL (quote currency)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

# Main pipeline
def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    print(f"Loading features from: {FEATURE_PATH}")

    df = pd.read_parquet(FEATURE_PATH)

    # Baseline signal logic
    df["signal"] = np.sign(df["ofi"]).fillna(0)
    df["raw_pnl"] = df["signal"].shift() * df["mid"].diff()
    df["spread_cost"] = df["spread"] / 2 if "spread" in df.columns else 0
    df["after_cost"] = df["raw_pnl"] - df["spread_cost"] - FEE_BPS * df["mid"]

    # Fill-aware execution model
    # Expected markout from OFI × next mid move
    df["expected_markout"] = df["ofi"] * df["mid"].diff().shift(-1)

    # Placement: join maker only if expected alpha ≥ half-spread + fees
    half_spread = df["spread"] / 2
    df["join_maker"] = df["expected_markout"] >= (half_spread + FEE_BPS * df["mid"])

    # Queue model: fill odds depend on queue imbalance (QI)
    df["fill_prob"] = np.where(
        df["join_maker"],
        (1 - df["qi"].abs()).clip(lower=0, upper=1),
        1.0
    )

    # Cancels on adverse microprice drift
    micro_drift = df["microprice"].diff().fillna(0)
    df["cancel"] = np.sign(micro_drift) != np.sign(df["signal"])
    df.loc[df["cancel"], "fill_prob"] *= 0.2

    # Maker/taker fees
    df["trade_fee"] = np.where(df["join_maker"], -MAKER_REBATE, TAKER_FEE) * df["mid"]

    # Slippage penalty on partial fills
    df["slippage_cost"] = (1 - df["fill_prob"]) * SLIPPAGE_TICK * df["mid"]

    # Realized PnL
    df["realized_pnl"] = (
        df["after_cost"] * df["fill_prob"]
        - df["trade_fee"]
        - df["slippage_cost"]
    )

    # Attribution & output
    attrib_after = compute_metrics(df, "after_cost")
    attrib_realized = compute_metrics(df, "realized_pnl")

    attrib_after["Scenario"] = "After-Cost (Ideal)"
    attrib_realized["Scenario"] = "Fill-Aware (Realized)"
    attrib = pd.concat([attrib_after, attrib_realized])

    print("\n=== Updated Attribution Table (Fill-Aware) ===")
    print(attrib.to_string(index=False))

    attrib_path = os.path.join(REPORT_DIR, "attribution_fillaware.csv")
    attrib.to_csv(attrib_path, index=False)
    print(f" Attribution table saved: {attrib_path}")

    # Save plots
    plot_cum_pnl(df, os.path.join(REPORT_DIR, "ofi_pnl_plot.png"))
    plot_fillaware_comparison(df, os.path.join(REPORT_DIR, "ofi_fillaware_comparison.png"))
    print(" Plots saved in reports/")

    print("\nDone — fill-aware baseline completed.")


if __name__ == "__main__":
    main()