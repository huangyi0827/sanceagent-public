import sys
from pathlib import Path
import json
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
from vendor.GeneralBacktest.backtest import GeneralBacktest

from src.data.csv_provider import CsvDataProvider
from src.agent.sance_public_agent import SanCePublicAgent, SanCePublicConfig


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def make_equal_weight_benchmark(universe, prices, start, end):
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
    bench = pd.DataFrame({"date": [first_day] * len(codes), "code": codes, "weight": [w] * len(codes)})
    return bench


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sance_public.yaml")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))

    outdir = ROOT / args.outdir
    safe_mkdir(outdir)
    safe_mkdir(outdir / "audit")

    provider = CsvDataProvider(root=Path(cfg.get("data_root", "demo_data")))
    universe = provider.load_universe()
    prices = provider.load_prices()

    # Commit-4 default: load all 6 factors (provider handles schema differences)
    factor_names = cfg.get(
        "factor_names",
        [
            "sector_sentiment",
            "retail_sentiment",
            "narrative_price_gap",
            "policy_sent_gap",
            "macro_liquidity",
            "news_tda",
        ],
    )
    factors = provider.load_factors(list(factor_names))

    agent_cfg = SanCePublicConfig(
        start=cfg["start"],
        end=cfg["end"],
        rebalance_mode=str(cfg.get("rebalance_mode", "week_end")),
        include_first_rebalance=bool(cfg.get("include_first_rebalance", True)),
        score_weights=dict(cfg.get("score_weights", {})) or None,
        use_price_mom=bool(cfg.get("use_price_mom", True)),
        mom_lookback=int(cfg.get("mom_lookback", 20)),
        mom_weight=float(cfg.get("mom_weight", 0.2)),
        base_topk=int(cfg.get("topk", cfg.get("base_topk", 10))),
        base_softmax_temp=float(cfg.get("softmax_temp", cfg.get("base_softmax_temp", 0.8))),
        min_weight=float(cfg.get("min_weight", 0.01)),
        no_trade_band=float(cfg.get("no_trade_band", 0.02)),
        sticky_buffer=int(cfg.get("sticky_buffer", 2)),
        audit_dir=str(outdir / "audit"),
    )

    agent = SanCePublicAgent(agent_cfg)
    weights_df = agent.generate_weights(universe, prices, factors)

    weights_path = outdir / "weights.csv"
    weights_df.to_csv(weights_path, index=False)

    # Benchmark: equal-weight buy&hold
    bench_weights = make_equal_weight_benchmark(universe, prices, cfg["start"], cfg["end"])
    bench_weights.to_csv(outdir / "benchmark_equal_weight.csv", index=False)

    # Backtest
    bt = GeneralBacktest(start_date=cfg["start"], end_date=cfg["end"])
    res = bt.run_backtest(
        weights_data=weights_df,
        price_data=prices,
        buy_price=cfg.get("buy_price", "open"),
        sell_price=cfg.get("sell_price", "close"),
        adj_factor_col=cfg.get("adj_factor_col", "adj_factor"),
        close_price_col=cfg.get("close_price_col", "close"),
        date_col=cfg.get("date_col", "date"),
        asset_col=cfg.get("asset_col", "code"),
        weight_col=cfg.get("weight_col", "weight"),
        rebalance_threshold=float(cfg.get("rebalance_threshold", 0.005)),
        transaction_cost=list(cfg.get("transaction_cost", [0.001, 0.001])),
        initial_capital=float(cfg.get("initial_capital", 1.0)),
        slippage=float(cfg.get("slippage", 0.0)),
        benchmark_weights=bench_weights,
        benchmark_name=str(cfg.get("benchmark_name", "EqualWeight")),
    )

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

    json.dump(serializable, open(json_path, "w"), ensure_ascii=False, indent=2)

    print("✅ Done.")
    print("Weights saved to:", weights_path)
    print("Backtest result index saved to:", json_path)
    print("Audit log:", outdir / "audit" / "audit.jsonl")
    print("Outputs dir:", outdir)


if __name__ == "__main__":
    main()
