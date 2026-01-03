import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FACTORS_DIR = ROOT / "demo_data" / "factors"
OUT_UNIVERSE = ROOT / "demo_data" / "universe.csv"
OUT_PRICES = ROOT / "demo_data" / "prices.csv"


def _norm_cols(df: pd.DataFrame):
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date") or cols.get("trade_date") or cols.get("dt")
    code_col = cols.get("code") or cols.get("ticker") or cols.get("symbol")
    return date_col, code_col


def _read_dates_and_optional_codes(fp: Path) -> tuple[pd.Series, pd.Series | None]:
    df = pd.read_csv(fp)
    date_col, code_col = _norm_cols(df)

    if date_col is None:
        raise ValueError(f"{fp.name} must have a date column; got: {list(df.columns)}")

    dates = pd.to_datetime(df[date_col])

    if code_col is None:
        return dates, None

    codes = df[code_col].astype(str)
    return dates, codes


def main(seed: int = 7) -> None:
    rng = np.random.default_rng(seed)

    files = sorted(glob.glob(str(FACTORS_DIR / "*.csv")))
    if not files:
        raise FileNotFoundError(f"No factor CSV found under {FACTORS_DIR}")

    all_dates = []
    all_codes = []

    for f in files:
        fp = Path(f)
        dates, codes = _read_dates_and_optional_codes(fp)
        all_dates.append(dates)
        if codes is not None:
            all_codes.append(codes)

    dates = pd.concat(all_dates, ignore_index=True)
    start = dates.min()
    end = dates.max()
    if pd.isna(start) or pd.isna(end):
        raise ValueError("Cannot infer date range from factor CSVs")

    if not all_codes:
        raise ValueError(
            "Cannot infer universe: no factor CSV contains a code/ticker/symbol column."
        )

    codes = sorted(pd.concat(all_codes, ignore_index=True).dropna().astype(str).unique().tolist())

    # Use business days as trading calendar
    cal = pd.bdate_range(start, end)

    OUT_UNIVERSE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"code": codes}).to_csv(OUT_UNIVERSE, index=False)

    rows = []
    for code in codes:
        n = len(cal)
        base = (abs(hash(code)) % 1000) / 1000.0

        drift = 0.00010 + 0.00015 * base
        vol = 0.008 + 0.010 * (1 - base)

        rets = rng.normal(loc=drift, scale=vol, size=n)
        close = 1.0 * np.exp(np.cumsum(rets))

        open_ = np.empty_like(close)
        open_[0] = close[0] * (1 + rng.normal(0, vol / 4))
        open_[1:] = close[:-1] * (1 + rng.normal(0, vol / 4, size=n - 1))

        rows.append(
            pd.DataFrame(
                {
                    "date": cal,
                    "code": code,
                    "open": open_,
                    "close": close,
                    "adj_factor": 1.0,
                }
            )
        )

    prices = pd.concat(rows, ignore_index=True)
    prices.to_csv(OUT_PRICES, index=False)

    print("✅ Wrote:", OUT_UNIVERSE)
    print("✅ Wrote:", OUT_PRICES)
    print("codes:", len(codes), "| dates:", len(cal))


if __name__ == "__main__":
    main()
