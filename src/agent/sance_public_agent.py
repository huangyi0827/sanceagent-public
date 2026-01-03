from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.csv_provider import GLOBAL_CODE
from src.router.public_rule_router import RouterParams, route_params


@dataclass
class SanCePublicConfig:
    start: str
    end: str

    # rebalance
    rebalance_mode: str = "week_end"  # week-end trading day (Fri/holiday-adjusted)
    include_first_rebalance: bool = True

    # scoring weights (ETF-level signals)
    score_weights: Optional[Dict[str, float]] = None  # keys: sector_sentiment, narrative_price_gap, policy_sent_gap
    use_price_mom: bool = True
    mom_lookback: int = 20
    mom_weight: float = 0.2

    # router base knobs
    base_topk: int = 10
    base_softmax_temp: float = 0.8
    min_weight: float = 0.01
    no_trade_band: float = 0.02
    sticky_buffer: int = 2

    # audit
    audit_dir: str = "outputs/audit"


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _softmax(x: np.ndarray, temp: float) -> np.ndarray:
    if len(x) == 0:
        return x
    t = max(float(temp), 1e-6)
    z = x / t
    z = z - np.nanmax(z)
    e = np.exp(z)
    s = np.nansum(e)
    return e / s if s > 0 else np.ones_like(e) / len(e)


def _zscore(s: pd.Series) -> pd.Series:
    x = s.astype(float)
    mu = np.nanmean(x.values)
    sig = np.nanstd(x.values)
    if not np.isfinite(mu) or not np.isfinite(sig) or sig < 1e-12:
        return pd.Series(0.0, index=s.index)
    return (x - mu) / (sig + 1e-12)


def _week_end_trading_dates(trading_dates: pd.DatetimeIndex) -> List[pd.Timestamp]:
    td = pd.DatetimeIndex(sorted(trading_dates.unique()))
    df = pd.DataFrame({"date": td})
    iso = df["date"].dt.isocalendar()
    df["iso_y"] = iso["year"].astype(int)
    df["iso_w"] = iso["week"].astype(int)
    week_end = df.groupby(["iso_y", "iso_w"], as_index=False)["date"].max()["date"]
    return list(pd.to_datetime(week_end).sort_values())


def _prev_trading_day(trading_dates: pd.DatetimeIndex, d: pd.Timestamp) -> pd.Timestamp:
    td = pd.DatetimeIndex(sorted(trading_dates.unique()))
    pos = td.searchsorted(d)
    if pos <= 0:
        return td[0]
    # if d is a trading day, use previous for anti-lookahead
    prev = td[pos - 1]
    if prev < d:
        return prev
    return td[max(pos - 2, 0)]


def _asof_value_for_codes(factor_df: pd.DataFrame, asof: pd.Timestamp, codes: List[str]) -> pd.Series:
    """
    factor_df normalized by CsvDataProvider: columns date, code, value
    - if code == GLOBAL_CODE: date-only factor => return scalar broadcast series
    - else: ETF-level factor => last value per code with date<=asof
    """
    df = factor_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["code"].astype(str)

    df = df[df["date"] <= asof]
    if df.empty:
        return pd.Series(index=pd.Index(codes, name="code"), dtype=float)

    if (df["code"] == GLOBAL_CODE).any():
        # date-only
        g = df[df["code"] == GLOBAL_CODE].sort_values("date")
        v = g["value"].iloc[-1]
        return pd.Series([v] * len(codes), index=pd.Index(codes, name="code"), dtype=float)

    # ETF-level
    df = df[df["code"].isin(codes)].sort_values(["code", "date"])
    if df.empty:
        return pd.Series(index=pd.Index(codes, name="code"), dtype=float)
    last = df.groupby("code")["value"].last()
    return last.reindex(codes)


def _price_momentum(prices: pd.DataFrame, asof: pd.Timestamp, codes: List[str], lookback: int) -> pd.Series:
    df = prices[prices["code"].isin(codes)].copy()
    df = df[df["date"] <= asof].sort_values(["code", "date"])
    if df.empty:
        return pd.Series(index=pd.Index(codes, name="code"), dtype=float)

    if "adj_factor" in df.columns:
        df["adj_close"] = pd.to_numeric(df["close"], errors="coerce") * pd.to_numeric(df["adj_factor"], errors="coerce")
    else:
        df["adj_close"] = pd.to_numeric(df["close"], errors="coerce")

    out = {}
    lb = int(lookback)
    for c, g in df.groupby("code"):
        g = g.dropna(subset=["adj_close"])
        if len(g) < lb + 1:
            out[c] = np.nan
            continue
        a = g["adj_close"].iloc[-1]
        b = g["adj_close"].iloc[-(lb + 1)]
        out[c] = float(a / b - 1.0) if b and b > 0 else np.nan

    return pd.Series(out).reindex(codes)


class SanCePublicAgent:
    def __init__(self, cfg: SanCePublicConfig):
        self.cfg = cfg
        self.audit_path = Path(cfg.audit_dir) / "audit.jsonl"
        _safe_mkdir(self.audit_path.parent)

    def _write_audit(self, rec: dict) -> None:
        import json
        with self.audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _router(self, asof: pd.Timestamp, signals: Dict[str, pd.Series]) -> Tuple[RouterParams, dict]:
        base = RouterParams(
            topk=int(self.cfg.base_topk),
            softmax_temp=float(self.cfg.base_softmax_temp),
            min_weight=float(self.cfg.min_weight),
            no_trade_band=float(self.cfg.no_trade_band),
            sticky_buffer=int(self.cfg.sticky_buffer),
        )

        # date-only proxies (broadcast series -> take the first value)
        macro_liq = signals.get("macro_liquidity")
        retail = signals.get("retail_sentiment")
        news = signals.get("news_tda")

        params = route_params(
            macro_liquidity=(float(macro_liq.iloc[0]) if macro_liq is not None and len(macro_liq) else None),
            retail_sentiment=(float(retail.iloc[0]) if retail is not None and len(retail) else None),
            news_tda=(float(news.iloc[0]) if news is not None and len(news) else None),
            base=base,
        )

        note = {
            "macro_liquidity": (float(macro_liq.iloc[0]) if macro_liq is not None and len(macro_liq) else None),
            "retail_sentiment": (float(retail.iloc[0]) if retail is not None and len(retail) else None),
            "news_tda": (float(news.iloc[0]) if news is not None and len(news) else None),
        }
        return params, note

    def _score(self, signals: Dict[str, pd.Series], mom: Optional[pd.Series]) -> pd.Series:
        # default weights: SanCe-style (gap + sentiment)
        w = dict(self.cfg.score_weights or {})
        if not w:
            w = {"sector_sentiment": 0.5, "narrative_price_gap": 0.3, "policy_sent_gap": 0.2}

        score = None
        for k, wk in w.items():
            s = signals.get(k)
            if s is None:
                continue
            z = _zscore(s)
            score = z * float(wk) if score is None else score + z * float(wk)

        if score is None:
            # fallback all zeros
            any_series = next(iter(signals.values()))
            score = pd.Series(0.0, index=any_series.index)

        if mom is not None and self.cfg.use_price_mom and self.cfg.mom_weight != 0:
            score = score + _zscore(mom) * float(self.cfg.mom_weight)

        return score

    def _apply_sticky(self, ranked_codes: List[str], prev_holdings: List[str], sticky_buffer: int) -> List[str]:
        """
        Keep some previous holdings to reduce churn:
        - take ranked list first
        - if prev holdings exist, allow up to sticky_buffer names from prev holdings
          that are within top (topk + sticky_buffer) of current ranking
        """
        if sticky_buffer <= 0 or not prev_holdings:
            return ranked_codes

        allow = set(ranked_codes[: len(ranked_codes)])
        prev_keep = [c for c in prev_holdings if c in allow][:sticky_buffer]
        out = []
        seen = set()
        for c in ranked_codes:
            if c not in seen:
                out.append(c)
                seen.add(c)
        for c in prev_keep:
            if c not in seen:
                out.insert(0, c)  # prioritize keeping
                seen.add(c)
        return out

    def _no_trade_band(self, prev: Optional[pd.Series], target: pd.Series, band: float) -> pd.Series:
        if prev is None or prev.empty or band <= 0:
            return target
        # align
        prev = prev.reindex(target.index).fillna(0.0)
        keep = (target - prev).abs() < float(band)
        adj = target.copy()
        adj[keep] = prev[keep]
        s = float(adj.sum())
        if s > 0:
            adj = adj / s
        return adj

    def generate_weights(self, universe: pd.DataFrame, prices: pd.DataFrame, factors: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        start_dt = pd.to_datetime(self.cfg.start)
        end_dt = pd.to_datetime(self.cfg.end)

        codes = universe["code"].astype(str).unique().tolist()

        px = prices.copy()
        px["date"] = pd.to_datetime(px["date"])
        px["code"] = px["code"].astype(str)

        trading_dates = pd.DatetimeIndex(px[(px["date"] >= start_dt) & (px["date"] <= end_dt)]["date"].unique()).sort_values()
        if len(trading_dates) == 0:
            raise ValueError("No trading dates in prices.csv within start/end.")

        if self.cfg.rebalance_mode != "week_end":
            raise ValueError("Public SanCeagent supports rebalance_mode='week_end' (week end trading day).")

        rebal_dates = [d for d in _week_end_trading_dates(trading_dates) if start_dt <= d <= end_dt]
        if self.cfg.include_first_rebalance and (trading_dates[0] not in rebal_dates):
            rebal_dates = [trading_dates[0]] + rebal_dates

        prev_w: Optional[pd.Series] = None
        prev_holdings: List[str] = []

        rows = []
        for d in rebal_dates:
            d = pd.Timestamp(d)
            signal_asof = _prev_trading_day(trading_dates, d)

            # 1) compute signals as-of D-1
            sigs: Dict[str, pd.Series] = {}
            coverage = {}
            for name, fdf in factors.items():
                s = _asof_value_for_codes(fdf, signal_asof, codes)
                sigs[name] = s
                coverage[name] = float(s.isna().mean())

            # 2) router params (uses date-only proxies)
            params, regime = self._router(signal_asof, sigs)

            # 3) momentum (optional)
            mom = None
            if self.cfg.use_price_mom and self.cfg.mom_lookback > 0:
                mom = _price_momentum(px, signal_asof, codes, self.cfg.mom_lookback)

            # 4) score -> rank
            score = self._score(sigs, mom)
            ranked = score.sort_values(ascending=False)
            ranked_codes = [c for c in ranked.index.tolist() if pd.notna(ranked.loc[c])]

            if not ranked_codes:
                ranked_codes = codes[:]  # fallback

            # 5) sticky buffer (reduce churn)
            pool = ranked_codes[: max(params.topk + params.sticky_buffer, params.topk)]
            pool = self._apply_sticky(pool, prev_holdings, params.sticky_buffer)

            picked = pool[: params.topk]
            picked_scores = ranked.loc[picked].astype(float).to_numpy()
            w_raw = _softmax(picked_scores, params.softmax_temp)

            w_ser = pd.Series(w_raw, index=pd.Index([str(x) for x in picked], name="code"))

            # min_weight cleanup
            w_ser[w_ser < params.min_weight] = 0.0
            ssum = float(w_ser.sum())
            if ssum > 0:
                w_ser = w_ser / ssum
            else:
                w_ser[:] = 1.0 / len(w_ser)

            # no-trade band vs previous weights
            w_ser = self._no_trade_band(prev_w, w_ser, params.no_trade_band)

            # output rows
            for code, wgt in w_ser.items():
                rows.append({"date": d.strftime("%Y-%m-%d"), "code": str(code), "weight": float(wgt)})

            # audit
            audit = {
                "asof_date": d.strftime("%Y-%m-%d"),
                "signal_asof_date": signal_asof.strftime("%Y-%m-%d"),
                "params_used": params.to_dict(),
                "regime_inputs": regime,
                "score_weights": dict(self.cfg.score_weights or {"sector_sentiment": 0.5, "narrative_price_gap": 0.3, "policy_sent_gap": 0.2}),
                "picked": [str(x) for x in picked],
                "picked_scores_head": {str(k): float(v) for k, v in ranked.loc[picked].head(min(5, len(picked))).items()},
                "final_weights": {str(k): float(v) for k, v in w_ser.items()},
                "coverage_nan_ratio": coverage,
                "note": "public surrogate: signals <= D-1; weekly rebalance uses week-end trading day (Fri/holiday-adjusted); router is rule-based (LLM stubbed).",
            }
            self._write_audit(audit)

            # update state
            prev_w = w_ser.copy()
            prev_holdings = w_ser.index.tolist()

        weights_df = pd.DataFrame(rows)
        return weights_df.sort_values(["date", "code"]).reset_index(drop=True)
