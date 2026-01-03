import sys
from pathlib import Path
import json
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
from vendor.GeneralBacktest.backtest import GeneralBacktest

from src.agent.sance_public_agent import SanCePublicAgent, SanCePublicConfig


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def make_equal_weight_benchmark(universe: pd.DataFrame, prices: pd.DataFrame, start: str, end: str):
    """
    Buy & Hold equal-weight benchmark:
    - Build once at the first trading day within [start, end]
    - Hold weights constant
    """
    codes = universe["code"].astype(str).unique().tolist()
    if len(codes) == 0:
        return pd.DataFrame(columns=["date", "code", "weight"])

    px = prices.copy()
    px["date"] = pd.to_datetime(px["date"])
    start_dt, end_dt = pd.to_datetime(start), pd.to_datetime(end)
    in_range = px[(px["date"] >= start_dt) & (px["date"] <= end_dt)]
    first_day = in_range["date"].min()
    if pd.isna(first_day):
        first_day = start_dt

    w = 1.0 / len(codes)
    return pd.DataFrame({"date": [first_day] * len(codes), "code": codes, "weight": [w] * len(codes)})


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/sance_public.yaml")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    outdir = ROOT / args.outdir
    safe_mkdir(outdir)

    # --- Agent config mapping (robust to missing keys) ---
    agent_cfg = SanCePublicConfig(
        start=str(cfg["start"]),
        end=str(cfg["end"]),

        rebalance_mode=str(cfg.get("rebalance_mode", "week_end")),
        rebalance_freq=int(cfg.get("rebalance_freq", 5)),

        topn=int(cfg.get("topn", cfg.get("topN", 10))),
        sticky_buffer=int(cfg.get("sticky_buffer", 5)),
        min_weight=float(cfg.get("min_weight", 0.01)),
        no_trade_band=float(cfg.get("no_trade_band", 0.02)),

        w_story_gap=float(cfg.get("w_story_gap", 1.0)),
        w_policy_contra=float(cfg.get("w_policy_contra", 1.0)),
        w_sector_sent=float(cfg.get("w_sector_sent", 0.6)),
        w_news_tda=float(cfg.get("w_news_tda", 0.4)),
        w_liquidity=float(cfg.get("w_liquidity", 0.3)),
        w_retail_overheat=float(cfg.get("w_retail_overheat", -0.5)),

        demo_data_dir=str(cfg.get("demo_data_dir", "demo_data")),
        factors_subdir=str(cfg.get("factors_subdir", "factors")),
        audit_dir=str(outdir / "audit"),
    )

    agent = SanCePublicAgent(agent_cfg)

    # --- Load data ---
    universe = agent.load_universe()
    prices = agent.load_prices()

    # --- Generate weights ---
    weights_df = agent.generate_weights(universe, prices)
    weights_path = outdir / "weights.csv"
    weights_df.to_csv(weights_path, index=False)

    # --- Benchmark: equal-weight buy&hold ---
    bench_weights = make_equal_weight_benchmark(universe, prices, agent_cfg.start, agent_cfg.end)
    bench_weights_path = outdir / "benchmark_equal_weight.csv"
    bench_weights.to_csv(bench_weights_path, index=False)

    # --- Backtest ---
    bt = GeneralBacktest(start_date=agent_cfg.start, end_date=agent_cfg.end)

    res = bt.run_backtest(
        weights_data=weights_df,
        price_data=prices,
        buy_price=str(cfg.get("buy_price", "open")),
        sell_price=str(cfg.get("sell_price", "close")),
        adj_factor_col=str(cfg.get("adj_factor_col", "adj_factor")),
        close_price_col=str(cfg.get("close_price_col", "close")),
        date_col=str(cfg.get("date_col", "date")),
        asset_col=str(cfg.get("asset_col", "code")),
        weight_col=str(cfg.get("weight_col", "weight")),
        rebalance_threshold=float(cfg.get("rebalance_threshold", 0.005)),
        transaction_cost=list(cfg.get("transaction_cost", [0.001, 0.001])),
        initial_capital=float(cfg.get("initial_capital", 1.0)),
        slippage=float(cfg.get("slippage", 0.0)),
        benchmark_weights=bench_weights,
        benchmark_name=str(cfg.get("benchmark_name", "EqualWeight")),
    )

    # --- Save outputs (DataFrame/Series -> csv, others -> json-safe) ---
    json_path = outdir / "backtest_result.json"
    serializable = {}

    for k, v in res.items():
        if isinstance(v, pd.DataFrame):
            v.to_csv(outdir / f"{k}.csv", index=False)
            serializable[k] = f"{k}.csv"
        elif isinstance(v, pd.Series):
            v.to_frame(name=k).reset_index().to_csv(outdir / f"{k}.csv", index=False)
            serializable[k] = f"{k}.csv"
        else:
            try:
                json.dumps(v)
                serializable[k] = v
            except TypeError:
                serializable[k] = str(v)

    json.dump(serializable, open(json_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("✅ Done.")
    print("Weights saved to:", weights_path)
    print("Benchmark weights saved to:", bench_weights_path)
    print("Backtest index saved to:", json_path)
    print("Outputs dir:", outdir)
    if (outdir / "audit" / "audit.jsonl").exists():
        print("Audit trail:", outdir / "audit" / "audit.jsonl")


if __name__ == "__main__":
    main()
