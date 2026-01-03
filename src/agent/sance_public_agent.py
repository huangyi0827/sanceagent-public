from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import json

import numpy as np
import pandas as pd


# -----------------------------
# Config
# -----------------------------
@dataclass
class SanCePublicConfig:
    # backtest / demo window
    start: str
    end: str

    # rebalance (SanCe: weekly by default)
    rebalance_mode: str = "week_end"   # "week_end" | "every_n_days"
    rebalance_freq: int = 5            # used when rebalance_mode="every_n_days"

    # portfolio construction
    topn: int = 10
    sticky_buffer: int = 5             # keep previous holdings if still within topn+buffer
    min_weight: float = 0.01
    no_trade_band: float = 0.02        # if abs(w_new-w_old) < band -> keep old

    # factor weights (public surrogate; tune for demo aesthetics)
    w_story_gap: float = 1.0
    w_policy_contra: float = 1.0
    w_sector_sent: float = 0.6
    w_news_tda: float = 0.4
    w_liquidity: float = 0.3
    w_retail_overheat: float = -0.5    # overheat => risk-off, so negative

    # IO
    demo_data_dir: str = "demo_data"
    factors_subdir: str = "factors"
    audit_dir: Optional[str] = None    # if set, write audit.jsonl


# -----------------------------
# Small utils (safe for public)
# -----------------------------
def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _as_str_code(x) -> str:
    # keep leading zeros if present; for ints, pad to 6 digits
    if pd.isna(x):
        return ""
    if isinstance(x, (int, np.integer)):
        return f"{int(x):06d}"
    s = str(x).strip()
    if s.isdigit() and len(s) < 6:
        s = s.zfill(6)
    return s


def _zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    m = s.mean(skipna=True)
    sd = s.std(skipna=True, ddof=0)
    if sd is None or float(sd) == 0.0 or np.isnan(sd):
        return s * 0.0
    return (s - m) / sd


def _week_end_trading_dates(dates: pd.DatetimeIndex) -> List[pd.Timestamp]:
    df = pd.DataFrame({"date": pd.to_datetime(dates)})
    iso = df["date"].dt.isocalendar()
    df["year"] = iso.year.astype(int)
    df["week"] = iso.week.astype(int)
    week_last = df.groupby(["year", "week"])["date"].max().sort_values()
    return list(pd.to_datetime(week_last.values))


def _prev_trading_day(trading_dates: pd.DatetimeIndex, asof: pd.Timestamp) -> Optional[pd.Timestamp]:
    td = pd.to_datetime(trading_dates).sort_values()
    asof = pd.to_datetime(asof)
    idx = td.searchsorted(asof, side="left") - 1
    if idx < 0:
        return None
    return td[int(idx)]


def _read_csv(fp: Path) -> pd.DataFrame:
    return pd.read_csv(fp)


def _normalize_date(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    if date_col not in df.columns:
        raise ValueError(f"missing '{date_col}' column in {list(df.columns)}")
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    return df


def _fix_duplicate_date_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Some exported CSVs accidentally contain two "date" headers.
    cols = list(df.columns)
    if cols.count("date") <= 1:
        return df
    new_cols = []
    seen = 0
    for c in cols:
        if c == "date":
            seen += 1
            new_cols.append("date" if seen == 1 else f"date_{seen}")
        else:
            new_cols.append(c)
    df = df.copy()
    df.columns = new_cols
    return df


# -----------------------------
# Agent
# -----------------------------
class SanCePublicAgent:
    """
    Public SanCeagent (de-sensitized):
    - consumes 6 factor tables that are already NLP-processed (signals/panels)
    - applies D-1 shift (no look-ahead)
    - weekly rebalance (week-end trading day)
    - sticky TopN + no-trade band
    - emits weights DataFrame: [date, code, weight]
    """

    def __init__(self, cfg: SanCePublicConfig):
        self.cfg = cfg
        self._audit_fp: Optional[Path] = None
        if cfg.audit_dir:
            ad = Path(cfg.audit_dir)
            _safe_mkdir(ad)
            self._audit_fp = ad / "audit.jsonl"

    # ---------- load ----------
    def _paths(self) -> Dict[str, Path]:
        base = Path(self.cfg.demo_data_dir)
        factors = base / self.cfg.factors_subdir
        return {
            "universe": base / "universe.csv",
            "prices": base / "prices.csv",
            "story_gap": factors / "etf_narrative_vs_price_gap_factor.csv",
            "policy_gap": factors / "etf_policy_vs_sent_gap_factor.csv",
            "sector_sent": factors / "etf_sector_sentiment_factor.csv",
            "macro_liq": factors / "macro_liquidity_factor.csv",
            "news_tda": factors / "news_tda_factor.csv",
            "retail_sent": factors / "retail_sentiment_factor.csv",
        }

    def load_universe(self) -> pd.DataFrame:
        fp = self._paths()["universe"]
        if not fp.exists():
            raise FileNotFoundError(f"Missing universe.csv at {fp}. Generate demo market data first.")
        df = _read_csv(fp)
        if "code" not in df.columns:
            raise ValueError(f"universe.csv must contain 'code' column; got {list(df.columns)}")
        df = df.copy()
        df["code"] = df["code"].apply(_as_str_code)
        return df

    def load_prices(self) -> pd.DataFrame:
        fp = self._paths()["prices"]
        if not fp.exists():
            raise FileNotFoundError(f"Missing prices.csv at {fp}. Generate demo market data first.")
        df = _read_csv(fp)
        need = {"date", "code", "open", "close"}
        miss = need - set(df.columns)
        if miss:
            raise ValueError(f"prices.csv missing columns: {sorted(miss)}; got {list(df.columns)}")
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["code"] = df["code"].apply(_as_str_code)
        if "adj_factor" not in df.columns:
            df["adj_factor"] = 1.0
        return df

    def load_factors(self) -> Dict[str, pd.DataFrame]:
        ps = self._paths()

        story = _normalize_date(_read_csv(ps["story_gap"]))
        if "code" not in story.columns:
            raise ValueError(f"{ps['story_gap'].name} must contain 'code' column")
        story = story.copy()
        story["code"] = story["code"].apply(_as_str_code)
        if "gap_score" not in story.columns and "gap_pct" not in story.columns:
            raise ValueError(f"{ps['story_gap'].name} needs gap_score or gap_pct")

        policy = _normalize_date(_read_csv(ps["policy_gap"]))
        if "code" not in policy.columns:
            raise ValueError(f"{ps['policy_gap'].name} must contain 'code' column")
        policy = policy.copy()
        policy["code"] = policy["code"].apply(_as_str_code)
        if "contrarian_score" not in policy.columns and "sent_minus_policy_gap" not in policy.columns:
            raise ValueError(f"{ps['policy_gap'].name} needs contrarian_score or sent_minus_policy_gap")

        sec = _normalize_date(_read_csv(ps["sector_sent"]))
        if "code" not in sec.columns:
            raise ValueError(f"{ps['sector_sent'].name} must contain 'code' column")
        sec = sec.copy()
        sec["code"] = sec["code"].apply(_as_str_code)
        if "sina_sector_sentiment" not in sec.columns and "sina_sector_sentiment_raw" not in sec.columns:
            raise ValueError(f"{ps['sector_sent'].name} needs sina_sector_sentiment or sina_sector_sentiment_raw")

        liq = _fix_duplicate_date_cols(_read_csv(ps["macro_liq"]))
        liq = _normalize_date(liq, "date")
        if "liq_20d" not in liq.columns and "liq_raw" not in liq.columns:
            raise ValueError(f"{ps['macro_liq'].name} needs liq_20d or liq_raw")

        news = _fix_duplicate_date_cols(_read_csv(ps["news_tda"]))
        news = _normalize_date(news, "date")
        if "news_tda_score" not in news.columns:
            non_date = [c for c in news.columns if c != "date"]
            if len(non_date) == 1:
                news = news.rename(columns={non_date[0]: "news_tda_score"})
            else:
                raise ValueError(f"{ps['news_tda'].name} needs news_tda_score; got {list(news.columns)}")

        retail = _fix_duplicate_date_cols(_read_csv(ps["retail_sent"]))
        date_col = "date" if "date" in retail.columns else ("date_2" if "date_2" in retail.columns else None)
        if not date_col:
            raise ValueError(f"{ps['retail_sent'].name} needs date column; got {list(retail.columns)}")
        retail = _normalize_date(retail, date_col).rename(columns={date_col: "date"})
        if "retail_overheat_z" not in retail.columns:
            cand = [c for c in retail.columns if "overheat" in c or "heat" in c]
            if cand:
                retail = retail.rename(columns={cand[0]: "retail_overheat_z"})
            else:
                retail["retail_overheat_z"] = 0.0

        return {
            "story": story,
            "policy": policy,
            "sector": sec,
            "liq": liq,
            "news": news,
            "retail": retail,
        }

    # ---------- core scoring ----------
    def _build_panel(self, universe: pd.DataFrame, prices: pd.DataFrame, factors: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        codes = universe["code"].astype(str).unique()
        cal = pd.DatetimeIndex(pd.to_datetime(prices["date"].unique())).sort_values()

        base = factors["story"][["date", "code"]].drop_duplicates()
        base = base.merge(
            factors["policy"][["date", "code"]].drop_duplicates(),
            on=["date", "code"],
            how="outer",
        ).merge(
            factors["sector"][["date", "code"]].drop_duplicates(),
            on=["date", "code"],
            how="outer",
        ).drop_duplicates()

        base = base[base["code"].isin(codes)].copy()

        story_val = "gap_score" if "gap_score" in factors["story"].columns else "gap_pct"
        base = base.merge(
            factors["story"][["date", "code", story_val, "sector_name"]].rename(columns={story_val: "story_gap"}),
            on=["date", "code"],
            how="left",
        )

        policy_val = "contrarian_score" if "contrarian_score" in factors["policy"].columns else "sent_minus_policy_gap"
        base = base.merge(
            factors["policy"][["date", "code", policy_val]].rename(columns={policy_val: "policy_contra"}),
            on=["date", "code"],
            how="left",
        )

        sec_val = "sina_sector_sentiment" if "sina_sector_sentiment" in factors["sector"].columns else "sina_sector_sentiment_raw"
        base = base.merge(
            factors["sector"][["date", "code", sec_val]].rename(columns={sec_val: "sector_sent"}),
            on=["date", "code"],
            how="left",
        )

        liq_val = "liq_20d" if "liq_20d" in factors["liq"].columns else "liq_raw"
        liq = factors["liq"][["date", liq_val]].rename(columns={liq_val: "macro_liq"})
        news = factors["news"][["date", "news_tda_score"]].rename(columns={"news_tda_score": "news_tda"})
        retail = factors["retail"][["date", "retail_overheat_z"]].rename(columns={"retail_overheat_z": "retail_overheat"})

        base = base.merge(liq, on="date", how="left").merge(news, on="date", how="left").merge(retail, on="date", how="left")

        start_dt, end_dt = pd.to_datetime(self.cfg.start), pd.to_datetime(self.cfg.end)
        base = base[(base["date"] >= start_dt) & (base["date"] <= end_dt)].copy()

        # D-1 shift: asof_date uses previous trading day's factor value
        base["asof_date"] = base["date"]
        prev_map = {d: _prev_trading_day(cal, d) for d in sorted(pd.to_datetime(base["asof_date"].unique()))}
        base["date"] = base["asof_date"].map(prev_map)
        base = base.dropna(subset=["date"]).copy()
        base = base.rename(columns={"asof_date": "date_asof"})

        base = base.sort_values(["date_asof", "code", "date"]).drop_duplicates(["date_asof", "code"], keep="last")
        base = base.drop(columns=["date"]).rename(columns={"date_asof": "date"})

        for c in ["story_gap", "policy_contra", "sector_sent", "news_tda", "macro_liq", "retail_overheat"]:
            if c not in base.columns:
                base[c] = 0.0
            base[c] = pd.to_numeric(base[c], errors="coerce").fillna(0.0)

        return base

    def _score_cross_section(self, panel: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        df = panel.copy()

        for c in ["story_gap", "policy_contra", "sector_sent"]:
            df[c + "_z"] = df.groupby("date")[c].transform(_zscore)

        for c in ["news_tda", "macro_liq", "retail_overheat"]:
            df[c + "_z"] = _zscore(df[c])

        df["score"] = (
            cfg.w_story_gap * df["story_gap_z"]
            + cfg.w_policy_contra * df["policy_contra_z"]
            + cfg.w_sector_sent * df["sector_sent_z"]
            + cfg.w_news_tda * df["news_tda_z"]
            + cfg.w_liquidity * df["macro_liq_z"]
            + cfg.w_retail_overheat * df["retail_overheat_z"]
        )
        return df

    def _rebalance_dates(self, trading_dates: pd.DatetimeIndex) -> List[pd.Timestamp]:
        cfg = self.cfg
        td = pd.DatetimeIndex(pd.to_datetime(trading_dates)).sort_values()
        start_dt, end_dt = pd.to_datetime(cfg.start), pd.to_datetime(cfg.end)
        td = td[(td >= start_dt) & (td <= end_dt)]
        if cfg.rebalance_mode == "week_end":
            return _week_end_trading_dates(td)
        return list(td[:: max(1, int(cfg.rebalance_freq))])

    def _apply_sticky_topn(self, ranked: List[str], prev_holdings: List[str], topn: int, buffer_n: int) -> List[str]:
        if not ranked:
            return []
        keep_zone = ranked[: max(topn + buffer_n, topn)]
        kept = [c for c in prev_holdings if c in keep_zone]
        out = kept[:]
        for c in ranked:
            if c not in out:
                out.append(c)
            if len(out) >= topn:
                break
        return out[:topn]

    def _weights_from_selection(self, selected: List[str]) -> Dict[str, float]:
        if not selected:
            return {}
        w = 1.0 / len(selected)
        return {c: w for c in selected}

    def _apply_no_trade_band(self, w_new: Dict[str, float], w_old: Dict[str, float]) -> Dict[str, float]:
        band = float(self.cfg.no_trade_band)
        if band <= 0:
            return w_new
        all_codes = set(w_new) | set(w_old)
        out = {}
        for c in all_codes:
            wn = w_new.get(c, 0.0)
            wo = w_old.get(c, 0.0)
            out[c] = wo if abs(wn - wo) < band else wn
        s = sum(out.values())
        if s > 0:
            out = {k: v / s for k, v in out.items() if v > 0}
        return out

    def _emit_audit(self, payload: Dict) -> None:
        if not self._audit_fp:
            return
        with self._audit_fp.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # ---------- public API ----------
    def generate_weights(self, universe: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        factors = self.load_factors()
        panel = self._build_panel(universe, prices, factors)
        scored = self._score_cross_section(panel)

        trading_dates = pd.DatetimeIndex(pd.to_datetime(prices["date"].unique())).sort_values()
        rebal_dates = self._rebalance_dates(trading_dates)

        out_rows = []
        prev_holdings: List[str] = []
        prev_w: Dict[str, float] = {}

        for asof in rebal_dates:
            asof = pd.to_datetime(asof)

            snap_dates = pd.to_datetime(scored["date"].unique())
            snap_dates = snap_dates[snap_dates <= asof]
            if len(snap_dates) == 0:
                continue
            d = pd.to_datetime(max(snap_dates))

            snap = scored[scored["date"] == d].copy().sort_values("score", ascending=False)
            ranked = snap["code"].astype(str).tolist()

            selected = self._apply_sticky_topn(ranked, prev_holdings, self.cfg.topn, self.cfg.sticky_buffer)
            w_raw = self._weights_from_selection(selected)
            w_final = self._apply_no_trade_band(w_raw, prev_w)

            mw = float(self.cfg.min_weight)
            if mw > 0:
                w_final = {k: v for k, v in w_final.items() if v >= mw}
                s = sum(w_final.values())
                if s > 0:
                    w_final = {k: v / s for k, v in w_final.items()}

            for code, w in w_final.items():
                out_rows.append({"date": asof, "code": code, "weight": float(w)})

            self._emit_audit(
                {
                    "asof_date": str(asof.date()),
                    "feature_date": str(d.date()),
                    "params_used": {
                        "rebalance_mode": self.cfg.rebalance_mode,
                        "topn": self.cfg.topn,
                        "sticky_buffer": self.cfg.sticky_buffer,
                        "no_trade_band": self.cfg.no_trade_band,
                        "weights": {
                            "story_gap": self.cfg.w_story_gap,
                            "policy_contra": self.cfg.w_policy_contra,
                            "sector_sent": self.cfg.w_sector_sent,
                            "news_tda": self.cfg.w_news_tda,
                            "macro_liq": self.cfg.w_liquidity,
                            "retail_overheat": self.cfg.w_retail_overheat,
                        },
                    },
                    "selected": selected,
                    "final_weights": {k: float(v) for k, v in sorted(w_final.items(), key=lambda kv: -kv[1])},
                    "note": "Signals are pre-processed (NLP factors) and shifted to <=D-1 to prevent look-ahead.",
                }
            )

            prev_holdings = list(w_final.keys())
            prev_w = w_final

        wdf = pd.DataFrame(out_rows)
        if wdf.empty:
            return pd.DataFrame(columns=["date", "code", "weight"])
        wdf["date"] = pd.to_datetime(wdf["date"])
        wdf["code"] = wdf["code"].apply(_as_str_code)
        return wdf.sort_values(["date", "code"]).reset_index(drop=True)
