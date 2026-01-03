from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


# Strategy semantic name -> file + schema
FACTOR_SPECS: Dict[str, Dict[str, Any]] = {
    # ETF-level (date+code) factors
    "sector_sentiment": {
        "file": "etf_sector_sentiment_factor.csv",
        "date_col": "date",
        "code_col": "code",
        "value_col": "sina_sector_sentiment",
    },

    # Date-only (macro / market-wide) factors: broadcast to all codes
    "news_tda": {
        "file": "news_tda_factor.csv",
        "date_col": "date",
        "code_col": None,
        "value_col": "news_tda_score",
    },

    # You can fill these when you confirm headers; below are sensible defaults:
    "macro_liquidity": {
        "file": "macro_liquidity_factor.csv",
        "date_col": "date",
        "code_col": None,
        "value_col": "liq_20d",   # your earlier example had liq_raw, liq_20d
    },
    "narrative_price_gap": {
        "file": "etf_narrative_vs_price_gap_factor.csv",
        "date_col": "date",
        "code_col": "code",
        "value_col": "value",
    },
    "policy_sent_gap": {
        "file": "etf_policy_vs_sent_gap_factor.csv",
        "date_col": "date",
        "code_col": "code",
        "value_col": "value",
    },
    "retail_sentiment": {
        "file": "retail_sentiment_factor.csv",
        "date_col": "date",
        "code_col": "code",
        "value_col": "value",
    },
}

GLOBAL_CODE = "__ALL__"  # sentinel for date-only factors


@dataclass
class CsvDataProvider:
    root: Path = Path("demo_data")

    def load_universe(self) -> pd.DataFrame:
        p = self.root / "universe.csv"
        df = pd.read_csv(p)
        if "code" not in df.columns:
            raise ValueError("universe.csv must contain column: code")
        df["code"] = df["code"].astype(str)
        return df

    def load_prices(self) -> pd.DataFrame:
        p = self.root / "prices.csv"
        df = pd.read_csv(p)
        need = {"date", "code", "open", "close"}
        if not need.issubset(df.columns):
            raise ValueError(f"prices.csv must contain columns: {sorted(list(need))}")
        df["date"] = pd.to_datetime(df["date"])
        df["code"] = df["code"].astype(str)
        if "adj_factor" not in df.columns:
            df["adj_factor"] = 1.0
        return df.sort_values(["date", "code"]).reset_index(drop=True)

    def load_factor(self, name: str) -> pd.DataFrame:
        if name not in FACTOR_SPECS:
            raise KeyError(f"Unknown factor name: {name}. Add it to FACTOR_SPECS in csv_provider.py")

        spec = FACTOR_SPECS[name]
        fp = self.root / "factors" / spec["file"]
        df = pd.read_csv(fp)

        date_col = spec["date_col"]
        code_col: Optional[str] = spec.get("code_col")
        value_col = spec["value_col"]

        if date_col not in df.columns:
            raise ValueError(f"{fp.name} missing date_col '{date_col}'. got={list(df.columns)}")
        if value_col not in df.columns:
            raise ValueError(f"{fp.name} missing value_col '{value_col}'. got={list(df.columns)}")

        out = pd.DataFrame()
        out["date"] = pd.to_datetime(df[date_col])

        if code_col is None:
            out["code"] = GLOBAL_CODE
        else:
            if code_col not in df.columns:
                raise ValueError(f"{fp.name} missing code_col '{code_col}'. got={list(df.columns)}")
            out["code"] = df[code_col].astype(str)

        out["value"] = pd.to_numeric(df[value_col], errors="coerce")
        out = out.dropna(subset=["date"]).sort_values(["date", "code"]).reset_index(drop=True)
        return out

    def load_factors(self, names: list[str]) -> Dict[str, pd.DataFrame]:
        return {n: self.load_factor(n) for n in names}
