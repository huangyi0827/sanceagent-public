# scripts/generate_demo_market_data.py
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FACTORS_DIR = ROOT / "demo_data" / "factors"
OUT_DIR = ROOT / "demo_data"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_csv(fp: Path) -> pd.DataFrame:
    return pd.read_csv(fp)


def _find_col(cols, candidates):
    low = {c.lower(): c for c in cols}
    for k in candidates:
        if k in low:
            return low[k]
    return None


def _pick_top_codes_from_factor_files(factors_dir: Path, topk: int) -> list[str]:
    """
    Pick top-K ETF codes by data coverage (#rows) from factor files that have (date, code).
    This is robust: some factor files may not have code column; we skip those.
    """
    code_counts = None

    for fp in sorted(factors_dir.glob("*.csv")):
        df = _read_csv(fp)
        dcol = _find_col(df.columns, ["date", "trade_date", "dt"])
        ccol = _find_col(df.columns, ["code", "ticker", "symbol"])
        if dcol is None or ccol is None:
            continue

        # normalize
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
        df = df.dropna(subset=[dcol])
        df[ccol] = df[ccol].astype(str)

        # count coverage
        vc = df[ccol].value_counts()
        if code_counts is None:
            code_counts = vc
        else:
            code_counts = code_counts.add(vc, fill_value=0)

    if code_counts is None or len(code_counts) == 0:
        raise ValueError(
            f"No factor CSV under {factors_dir} contains both date and code columns. "
            f"Cannot infer ETF universe."
        )

    top_codes = code_counts.sort_values(ascending=False).head(topk).index.astype(str).tolist()
    return top_codes


def _infer_date_range_last_year(factors_dir: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Infer end date as max(date) across all factor files that have a date column.
    start date = end date - 365 days
    """
    max_dt = None
    for fp in sorted(factors_dir.glob("*.csv")):
        df = _read_csv(fp)
        dcol = _find_col(df.columns, ["date", "trade_date", "dt"])
        if dcol is None:
            continue
        dt = pd.to_datetime(df[dcol], errors="coerce")
        dt = dt.dropna()
        if dt.empty:
            continue
        cur = dt.max()
        if max_dt is None or cur > max_dt:
            max_dt = cur

    if max_dt is None:
        raise ValueError(f"Cannot infer end date: no usable date column found in {factors_dir}")

    end = pd.Timestamp(max_dt).normalize()
    start = (end - pd.Timedelta(days=365)).normalize()
    return start, end


def _make_business_dates(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    # Simplify: use business days as proxy. Good enough for a public demo.
    # (If you want true CN trading calendar later, swap this with an exchange calendar.)
    return pd.bdate_range(start=start, end=end, freq="C")


def _simulate_prices(codes: list[str], dates: pd.DatetimeIndex, seed: int = 42) -> pd.DataFrame:
    """
    Generate small reproducible OHLC demo prices:
    - Each code has its own drift/vol depending on prefix-ish heuristics
    - Prices are positive and smooth-ish
    - adj_factor fixed to 1.0 for demo (can be extended)
    """
    rng = np.random.default_rng(seed)

    def params(code: str) -> tuple[float, float]:
        # Heuristic regimes by code prefix (feel free to adjust)
        # bond-like ETFs often start with 511/512 etc; equity with 510/159 etc.
        if code.startswith(("511", "512")):
            mu, sigma = 0.00005, 0.0025
        elif code.startswith(("518", "513")):
            mu, sigma = 0.00015, 0.010
        else:
            mu, sigma = 0.00010, 0.008
        return mu, sigma

    rows = []
    for code in codes:
        mu, sigma = params(code)
        n = len(dates)
        rets = rng.normal(loc=mu, scale=sigma, size=n)
        # start price depends on code group for variety
        p0 = float(rng.uniform(0.8, 1.2)) * (100 if not code.startswith(("511", "512")) else 1.0)
        close = p0 * np.exp(np.cumsum(rets))

        # open is previous close with small noise
        open_ = np.concatenate([[close[0]], close[:-1]]) * (1 + rng.normal(0, sigma / 4, size=n))

        # high/low around open/close
        hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, sigma / 3, size=n)))
        lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, sigma / 3, size=n)))

        df = pd.DataFrame(
            {
                "date": dates,
                "code": code,
                "open": open_,
                "high": hi,
                "low": lo,
                "close": close,
                "adj_factor": 1.0,
            }
        )
        rows.append(df)

    out = pd.concat(rows, ignore_index=True)
    # keep stable ordering
    out["date"] = pd.to_datetime(out["date"])
    out["code"] = out["code"].astype(str)
    out = out.sort_values(["date", "code"]).reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=30, help="how many ETF codes to keep")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not FACTORS_DIR.exists():
        raise FileNotFoundError(f"Missing factors dir: {FACTORS_DIR}")

    _ensure_dir(OUT_DIR)

    # 1) pick universe codes (TopK by coverage)
    codes = _pick_top_codes_from_factor_files(FACTORS_DIR, topk=int(args.topk))

    # 2) infer last-year date range from factors
    start, end = _infer_date_range_last_year(FACTORS_DIR)
    dates = _make_business_dates(start, end)

    # 3) write universe.csv
    universe = pd.DataFrame(
        {
            "code": codes,
            "asset_class": ["EQUITY"] * len(codes),  # placeholder, can refine later
        }
    )
    universe_path = OUT_DIR / "universe.csv"
    universe.to_csv(universe_path, index=False)

    # 4) write prices.csv (small!)
    prices = _simulate_prices(codes=codes, dates=dates, seed=int(args.seed))
    prices_path = OUT_DIR / "prices.csv"
    prices.to_csv(prices_path, index=False)

    print("✅ Demo market data generated")
    print("Universe:", universe_path, "| n_codes=", len(codes))
    print("Prices:", prices_path, "| rows=", len(prices))
    print("Date range:", str(start.date()), "->", str(end.date()))


if __name__ == "__main__":
    main()
