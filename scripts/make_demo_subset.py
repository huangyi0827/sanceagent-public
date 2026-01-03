from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

BACKUP = ROOT / "demo_data" / "_local_backup"  # place big csv here
FACTORS = ROOT / "demo_data" / "factors"       # output demo csv here

# ---- demo controls ----
START = "2024-01-01"
END = "2025-12-31"
MAX_CODES = 20  # keep top-N codes by coverage

DATE_COL_CAND = ["date", "trade_date", "dt"]

def _find_date_col(cols):
    low = {c.lower(): c for c in cols}
    for k in DATE_COL_CAND:
        if k in low:
            return low[k]
    return None

def _find_code_col(cols):
    low = {c.lower(): c for c in cols}
    for k in ["code", "ticker", "symbol"]:
        if k in low:
            return low[k]
    # sometimes code is last col named "code" already, handled above
    return None

def _coerce_date(s):
    return pd.to_datetime(s, errors="coerce")

def subset_file(fp: Path, out_fp: Path, keep_codes: list[str] | None):
    df = pd.read_csv(fp)
    dcol = _find_date_col(df.columns)
    if dcol is None:
        raise ValueError(f"{fp.name}: cannot find date column in {list(df.columns)}")
    df[dcol] = _coerce_date(df[dcol])
    df = df.dropna(subset=[dcol])

    df = df[(df[dcol] >= pd.to_datetime(START)) & (df[dcol] <= pd.to_datetime(END))].copy()

    ccol = _find_code_col(df.columns)
    if ccol is not None and keep_codes is not None:
        df[ccol] = df[ccol].astype(str)
        df = df[df[ccol].isin(keep_codes)].copy()

    out_fp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_fp, index=False)
    return len(df), list(df.columns)

def main():
    FACTORS.mkdir(parents=True, exist_ok=True)

    # Pick universe codes from one ETF-level file if possible
    # Prefer sector sentiment file (has date+code)
    candidates = [
        "etf_sector_sentiment_factor.csv",
        "etf_policy_vs_sent_gap_factor.csv",
        "etf_narrative_vs_price_gap_factor.csv",
    ]
    src = None
    for name in candidates:
        if (BACKUP / name).exists():
            src = BACKUP / name
            break

    keep_codes = None
    if src is not None:
        df = pd.read_csv(src)
        dcol = _find_date_col(df.columns)
        ccol = _find_code_col(df.columns)
        if dcol and ccol:
            df[dcol] = _coerce_date(df[dcol])
            df = df.dropna(subset=[dcol])
            df = df[(df[dcol] >= pd.to_datetime(START)) & (df[dcol] <= pd.to_datetime(END))].copy()
            df[ccol] = df[ccol].astype(str)
            top = df[ccol].value_counts().head(MAX_CODES).index.tolist()
            keep_codes = top

    files = sorted([p for p in BACKUP.glob("*.csv")])
    if not files:
        raise FileNotFoundError(f"No CSV found under {BACKUP}. Put big factor CSVs there first.")

    for fp in files:
        out_fp = FACTORS / fp.name
        n, cols = subset_file(fp, out_fp, keep_codes)
        print(f"✅ {fp.name} -> {out_fp} | rows={n} | cols={cols}")

    print("DONE. Demo factors written to demo_data/factors/")
    if keep_codes is not None:
        print("Keep codes:", keep_codes)

if __name__ == "__main__":
    main()
