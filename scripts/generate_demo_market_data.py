import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FACTORS_DIR = ROOT / "demo_data" / "factors"
OUT_UNIVERSE = ROOT / "demo_data" / "universe.csv"
OUT_PRICES = ROOT / "demo_data" / "prices.csv"


def _read_factor_csv(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp)
    cols = {c.lower(): c for c in df.columns}

    date_col = cols.get("date") or cols.get("trade_date") or cols.get("dt")
    code_col = cols.get("code") or cols.get("ticker") or cols.get("symbol")

    if date_col is None or code_col is None:
        raise ValueError(f"{fp.name} must have date/code columns; got: {list(df.columns)}")

    out = df[[date_col, code_col]].copy()
    out.columns = ["date", "code"]
    out["date"] = pd.to_datetime(out["date"])
    out["code"] = out["code"].astype(str)
    return out


def main(seed: int = 7) -> None:
    rng = np.random.default_rng(seed)

    files = sorted(glob.glob(str(FACTORS_DIR / "*.csv")))
    if not files:
        raise FileNotFoundError(f"No factor CSV found under {FACTORS_DIR}")

    all_dc = [_read_factor_csv(Path(f)) for f in files]
    dc = pd.concat(all_dc, ignore_index=True).drop_duplicates()

    codes = sorted(dc["code"].unique().tolist())
    start = dc["date"].min()
    end = dc["date"].max()
    if pd.isna(start) or pd.isna(end):
        raise ValueError("Cannot infer date range from factor CSVs")

    dates = pd.bdate_range(start, end)

    OUT_UNIVERSE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"code": codes}).to_csv(OUT_UNIVERSE, index=False)

    rows = []
    for code in codes:
        n = len(dates)
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
                    "date": dates,
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
    print("codes:", len(codes), "| dates:", len(dates))


if __name__ == "__main__":
    main()
