from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd


def _read_csv(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp)
    # Normalize column names
    df.columns = [str(c).strip() for c in df.columns]

    # Handle duplicated "date" columns from some exports (e.g., retail_sentiment_factor.csv)
    date_cols = [i for i, c in enumerate(df.columns) if c == "date"]
    if len(date_cols) >= 2:
        # Keep the first date as 'date', rename others to date_2, date_3...
        for j, idx in enumerate(date_cols[1:], start=2):
            df.rename(columns={"date": "date"} if idx == date_cols[0] else {}, inplace=True)
            df.columns = [
                (f"date_{j}" if (k == idx) else col) for k, col in enumerate(df.columns)
            ]

    if "date" not in df.columns:
        raise ValueError(f"{fp.name} must contain a 'date' column; got: {list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"])
    return df


def _ensure_code_str(df: pd.DataFrame) -> pd.DataFrame:
    if "code" in df.columns:
        df["code"] = df["code"].astype(str)
    return df


@dataclass(frozen=True)
class FactorPaths:
    root: Path  # demo_data/factors
    etf_narrative: str = "etf_narrative_vs_price_gap_factor.csv"
    etf_policy_gap: str = "etf_policy_vs_sent_gap_factor.csv"
    etf_sector_sent: str = "etf_sector_sentiment_factor.csv"
    macro_liquidity: str = "macro_liquidity_factor.csv"
    news_tda: str = "news_tda_factor.csv"
    retail_sent: str = "retail_sentiment_factor.csv"


class FactorPanelLoader:
    """
    Build a unified feature panel for the agent:
    - Panel-level factors: merge on (date, code)
    - Macro-level factors: merge on date then broadcast to all codes
    - Apply shift(1) to avoid leakage (use <= D-1)
    """

    def __init__(self, factor_paths: FactorPaths):
        self.p = factor_paths

    def load_panel(
        self,
        universe: pd.DataFrame,
        start: Optional[str] = None,
        end: Optional[str] = None,
        shift_days: int = 1,
        keep_codes: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        uni = universe.copy()
        if "code" not in uni.columns:
            raise ValueError("universe must contain 'code' column")
        uni["code"] = uni["code"].astype(str)

        if keep_codes is not None:
            keep_codes = [str(x) for x in keep_codes]
            uni = uni[uni["code"].isin(keep_codes)].copy()

        # --- panel-level ---
        nar = _ensure_code_str(_read_csv(self.p.root / self.p.etf_narrative))
        pol = _ensure_code_str(_read_csv(self.p.root / self.p.etf_policy_gap))
        sec = _ensure_code_str(_read_csv(self.p.root / self.p.etf_sector_sent))

        # pick stable, interpretable columns (avoid leaking private semantics)
        nar_cols = [c for c in ["gap_score", "gap_pct", "story_z", "price_mom_z"] if c in nar.columns]
        pol_cols = [c for c in ["sent_minus_policy_gap", "contrarian_score", "policy_z", "sent_z"] if c in pol.columns]
        sec_cols = [c for c in ["sina_sector_sentiment_raw", "sina_sector_sentiment", "sina_sector_pos_ratio_20d"] if c in sec.columns]

        nar = nar[["date", "code"] + nar_cols].copy()
        pol = pol[["date", "code"] + pol_cols].copy()
        sec = sec[["date", "code"] + sec_cols].copy()

        panel = nar.merge(pol, on=["date", "code"], how="outer").merge(sec, on=["date", "code"], how="outer")

        # restrict to universe codes
        panel = panel.merge(uni[["code"]].drop_duplicates(), on="code", how="inner")

        # --- macro-level (date only) ---
        macro = _read_csv(self.p.root / self.p.macro_liquidity)
        news = _read_csv(self.p.root / self.p.news_tda)
        retail = _read_csv(self.p.root / self.p.retail_sent)

        # pick columns
        macro_cols = [c for c in ["liq_20d", "liq_raw"] if c in macro.columns]
        news_cols = [c for c in ["news_tda_score"] if c in news.columns] or [c for c in news.columns if c != "date"]
        retail_cols = [c for c in ["retail_overheat_z", "retail_heat_raw", "sina_score", "east_score"] if c in retail.columns]

        macro = macro[["date"] + macro_cols].copy()
        news = news[["date"] + news_cols].copy()
        retail = retail[["date"] + retail_cols].copy()

        macro_all = macro.merge(news, on="date", how="outer").merge(retail, on="date", how="outer")

        # broadcast macro to all codes by merging on date
        panel = panel.merge(macro_all, on="date", how="left")

        # date filter
        if start is not None:
            panel = panel[panel["date"] >= pd.to_datetime(start)]
        if end is not None:
            panel = panel[panel["date"] <= pd.to_datetime(end)]

        # leakage control: shift numeric features by code for panel-level, and by date for macro-level
        num_cols = [c for c in panel.columns if c not in ("date", "code")]

        if shift_days and shift_days > 0:
            panel = panel.sort_values(["code", "date"]).reset_index(drop=True)
            # shift all numeric columns within each code (works for both panel+macro after merge)
            panel[num_cols] = panel.groupby("code")[num_cols].shift(shift_days)

        return panel
