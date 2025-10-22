"""
Toxicity / Flow Filter Improvement
Goal:
  Add a flow-toxicity control to the Fill-Aware OFI strategy.
  The filter disables or scales signals when order-flow imbalance
  and short-term volatility jointly indicate toxic market conditions.

Inputs:
  - data/binance/features/{DATE}_BTCUSDT_100ms.parquet
Outputs:
  Flow-Control: reports/
    • ofi_flowfilter_comparison.png
    • attribution_flowfilter.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FEATURE_PATH = "data/binance/features/2024-10-21_BTCUSDT_100ms.parquet"
REPORT_DIR = "Flow-Control: reports"
FEE_BPS = 0.0001
BAR_FREQ = 100
MAKER_REBATE = 0.00002
TAKER_FEE = 0.0004
SLIPPAGE_TICK = 0.0001

# Metric engine (re-uses Fill-Aware structure)
def compute_metrics(df: pd.DataFrame, pnl_col: str = "realized_pnl") -> pd.DataFrame:
    cum_pnl = df[pnl_col].cumsum()
    total_pnl = df[pnl_col].sum()
    mean_pnl = df[pnl_col].mean()
    std_pnl = df[pnl_col].std()
    sharpe = np.nan if std_pnl == 0 else mean_pnl / std_pnl * np.sqrt(24 * 60 * (1000 / BAR_FREQ))

    if "spread" in df.columns and df["spread"].sum() != 0:
        spread_capture_bps = (df["raw_pnl"].sum() / df["spread"].sum()) * 1e4
    else:
        spread_capture_bps = np.nan

    # Forward selection losses
    df["fwd_10ms"] = df["mid"].shift(-1) - df["mid"]
    df["fwd_50ms"] = df["mid"].shift(-5) - df["mid"]
    df["fwd_250ms"] = df["mid"].shift(-25) - df["mid"]
    def sel_loss(col): return -np.nanmean(df["signal"] * df[col]) * 1e4
    sel10, sel50, sel250 = sel_loss("fwd_10ms"), sel_loss("fwd_50ms"), sel_loss("fwd_250ms")

    hedge_slip = np.nanstd(df["microprice"].diff() / df["mid"]) * 1e4 if "microprice" in df.columns else np.nan
    fill_rate = df["fill_prob"].mean() * 100 if "fill_prob" in df.columns else np.nan
    df["pos"] = df["signal"].cumsum()
    breaches = (df["pos"].abs() > 10).sum()

    roll_max = cum_pnl.cummax()
    drawdown = cum_pnl - roll_max
    max_dd = drawdown.min()
    worst_day = df.groupby(df.index.floor("1D"))[pnl_col].sum().min()
    hit = (df[pnl_col] > 0).mean()

    return pd.DataFrame({
        "Metric": [
            "After-cost Sharpe","Spread Capture (bps)",
            "Selection Loss 10ms (bps)","Selection Loss 50ms (bps)","Selection Loss 250ms (bps)",
            "Hedge Slippage / Carry (bps)","Fill Rate (% of quotes)",
            "Inventory Breaches (#)","Worst Day PnL","Max Drawdown","Hit Ratio"
        ],
        "Value": [sharpe,spread_capture_bps,sel10,sel50,sel250,
                  hedge_slip,fill_rate,breaches,worst_day,max_dd,hit]
    })

# PnL plotters
def plot_comparison(df_base, df_ctrl, out_path):
    plt.figure(figsize=(10,5))
    df_base["realized_pnl"].cumsum().plot(label="Fill-Aware (Baseline)")
    df_ctrl["realized_pnl"].cumsum().plot(label="Flow-Filtered (Improved)")
    plt.title("Cumulative PnL Comparison — Flow-Toxicity Filter")
    plt.xlabel("Time"); plt.ylabel("PnL (quote currency)")
    plt.legend(); plt.grid(True,linestyle="--",alpha=0.6)
    plt.tight_layout(); plt.savefig(out_path,dpi=300); plt.close()

# Main pipeline
def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    print(f"Loading features from: {FEATURE_PATH}")
    df = pd.read_parquet(FEATURE_PATH)

    # Fill-Aware Baseline
    df["signal"] = np.sign(df["ofi"]).fillna(0)
    df["raw_pnl"] = df["signal"].shift() * df["mid"].diff()
    df["spread_cost"] = df["spread"]/2 if "spread" in df.columns else 0
    df["after_cost"] = df["raw_pnl"] - df["spread_cost"] - FEE_BPS*df["mid"]

    df["expected_markout"] = df["ofi"] * df["mid"].diff().shift(-1)
    half_spread = df["spread"]/2
    df["join_maker"] = df["expected_markout"] >= (half_spread + FEE_BPS*df["mid"])
    df["fill_prob"] = np.where(df["join_maker"], (1-df["qi"].abs()).clip(0,1), 1.0)

    micro_drift = df["microprice"].diff().fillna(0)
    df["cancel"] = np.sign(micro_drift)!=np.sign(df["signal"])
    df.loc[df["cancel"],"fill_prob"]*=0.2
    df["trade_fee"] = np.where(df["join_maker"], -MAKER_REBATE, TAKER_FEE)*df["mid"]
    df["slippage_cost"] = (1-df["fill_prob"])*SLIPPAGE_TICK*df["mid"]
    df["realized_pnl"] = (df["after_cost"]*df["fill_prob"]
                          - df["trade_fee"] - df["slippage_cost"])

    base_metrics = compute_metrics(df, "realized_pnl")

    # Flow / Toxicity Filter
    print("Applying flow-toxicity filter...")

    df2 = df.copy()
    df2["vol"] = df2["mid"].pct_change().rolling(100).std()
    df2["flow_intensity"] = df2["ofi"].abs().rolling(200).mean()

    # Toxic if flow burst + high realized vol
    df2["toxic"] = (df2["ofi"].abs() > 2*df2["flow_intensity"]) & (df2["vol"] > df2["vol"].median())

    # Scale down signals instead of cutting entirely (less turnover shock)
    df2["signal"] = np.where(df2["toxic"], df2["signal"]*0.3, df2["signal"])

    # Re-compute realized PnL with same microstructure logic
    df2["raw_pnl"] = df2["signal"].shift()*df2["mid"].diff()
    df2["after_cost"] = df2["raw_pnl"] - df2["spread_cost"] - FEE_BPS*df2["mid"]
    df2["expected_markout"] = df2["ofi"]*df2["mid"].diff().shift(-1)
    df2["join_maker"] = df2["expected_markout"] >= (half_spread + FEE_BPS*df2["mid"])
    df2["fill_prob"] = np.where(df2["join_maker"], (1-df2["qi"].abs()).clip(0,1),1.0)
    df2.loc[df2["cancel"],"fill_prob"]*=0.2
    df2["trade_fee"] = np.where(df2["join_maker"], -MAKER_REBATE, TAKER_FEE)*df2["mid"]
    df2["slippage_cost"] = (1-df2["fill_prob"])*SLIPPAGE_TICK*df2["mid"]
    df2["realized_pnl"] = (df2["after_cost"]*df2["fill_prob"]
                           - df2["trade_fee"] - df2["slippage_cost"])

    filter_metrics = compute_metrics(df2, "realized_pnl")

    # Combine Attribution
    base_metrics["Scenario"] = "Fill-Aware (Baseline)"
    filter_metrics["Scenario"] = "Flow-Filtered (Improved)"
    attrib = pd.concat([base_metrics, filter_metrics])
    attrib_path = os.path.join(REPORT_DIR, "attribution_flowfilter.csv")
    attrib.to_csv(attrib_path, index=False)
    print(f"Attribution table saved: {attrib_path}")

    # Plot Comparison
    plot_path = os.path.join(REPORT_DIR, "ofi_flowfilter_comparison.png")
    plot_comparison(df, df2, plot_path)
    print(f"Comparison plot saved: {plot_path}")

    print("\n=== Flow-Filter Attribution ===")
    print(attrib.to_string(index=False))
    print("\nDone — surgical improvement complete.")


if __name__ == "__main__":
    main()