from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class WeightConstraints:
    min_weight: float = 0.0          # prune tiny weights
    cap_lo: float = 0.0              # lower bound per asset
    cap_hi: float = 1.0              # upper bound per asset


def normalize_weights(w: pd.Series) -> pd.Series:
    w = w.fillna(0.0)
    s = float(w.sum())
    if s <= 0:
        return w * 0.0
    return w / s


def apply_caps_and_prune(w: pd.Series, c: WeightConstraints) -> pd.Series:
    w = w.fillna(0.0).clip(lower=c.cap_lo, upper=c.cap_hi)
    if c.min_weight and c.min_weight > 0:
        w = w.where(w >= c.min_weight, 0.0)
    return normalize_weights(w)


def no_trade_band(prev: Optional[pd.Series], target: pd.Series, band: float) -> pd.Series:
    """
    If |target - prev| < band, keep prev to reduce churn.
    """
    if prev is None or prev.empty:
        return target
    prev = prev.reindex(target.index).fillna(0.0)
    keep = (target - prev).abs() < float(band)
    out = target.copy()
    out[keep] = prev[keep]
    return normalize_weights(out)


def sticky_topn(prev_holdings: Optional[pd.Index], scores: pd.Series, topn: int, buffer: int = 0) -> pd.Index:
    """
    Sticky TopN: keep previous top names if they are within topN+buffer by score.
    """
    scores = scores.dropna().sort_values(ascending=False)
    if scores.empty:
        return pd.Index([])
    cand = scores.index[: max(topn + buffer, topn)]
    if prev_holdings is None or len(prev_holdings) == 0:
        return pd.Index(scores.index[:topn])
    prev_in = [x for x in prev_holdings if x in cand]
    new_need = max(0, topn - len(prev_in))
    new_pick = [x for x in scores.index if x not in prev_in][:new_need]
    return pd.Index(prev_in + new_pick)
