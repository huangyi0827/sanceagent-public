from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class RouterParams:
    topk: int = 10
    softmax_temp: float = 0.8
    min_weight: float = 0.01
    no_trade_band: float = 0.02
    sticky_buffer: int = 2

    risk_scale: float = 1.0  # kept for audit / future; public demo stays fully invested

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if v != v:  # NaN
            return None
        return v
    except Exception:
        return None


def route_params(
    *,
    macro_liquidity: Any = None,
    retail_sentiment: Any = None,
    news_tda: Any = None,
    base: Optional[RouterParams] = None,
) -> RouterParams:
    """
    Public rule router (no LLM, no private thresholds).
    Uses date-only regime proxies:
      - macro_liquidity (liq_20d): lower -> more conservative
      - retail_sentiment (retail_overheat_z): higher -> more conservative
      - news_tda (news_tda_score): lower -> slightly conservative

    Output controls:
      topk / softmax_temp / min_weight / no_trade_band / sticky_buffer
    """
    p = base or RouterParams()

    liq = _safe_float(macro_liquidity)
    retail = _safe_float(retail_sentiment)
    news = _safe_float(news_tda)

    # --- simple, explainable regimes (public demo) ---
    # We avoid magic "private" cutoffs by using relative / generic thresholds.
    risk_off = False
    risk_note = []

    if liq is not None and liq < 0:
        risk_off = True
        risk_note.append("liq_20d<0")

    if retail is not None and retail > 1.0:
        risk_off = True
        risk_note.append("retail_overheat_z>1")

    if news is not None and news < 0:
        risk_note.append("news_tda_score<0")

    # --- dispatch ---
    if risk_off:
        # conservative: fewer names, smoother weights, stronger no-trade band
        p.topk = max(6, int(p.topk * 0.7))
        p.softmax_temp = max(0.9, p.softmax_temp + 0.2)
        p.no_trade_band = min(0.04, max(p.no_trade_band, 0.025))
        p.risk_scale = 0.8
    else:
        # neutral/risk-on
        p.topk = int(p.topk)
        p.softmax_temp = float(p.softmax_temp)
        p.no_trade_band = float(p.no_trade_band)
        p.risk_scale = 1.0

    # if news is weak, slightly smooth weights
    if news is not None and news < 0 and not risk_off:
        p.softmax_temp = max(p.softmax_temp, 0.9)

    # keep bounds sane
    p.topk = int(min(max(p.topk, 3), 50))
    p.softmax_temp = float(min(max(p.softmax_temp, 0.3), 2.0))
    p.min_weight = float(min(max(p.min_weight, 0.0), 0.2))
    p.no_trade_band = float(min(max(p.no_trade_band, 0.0), 0.1))
    p.sticky_buffer = int(min(max(p.sticky_buffer, 0), 10))

    return p
