from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any


def _clamp_int(x: int, lo: int, hi: int) -> int:
    return int(min(hi, max(lo, int(x))))


def _clamp_float(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, float(x))))


@dataclass(frozen=True)
class RouterParams:
    """
    Router output knobs (public, rule-based).
    Consumed by SanCePublicAgent:
      - topk: holding count (concentration)
      - softmax_temp: within-portfolio softness (higher => flatter weights)
      - min_weight: prune tiny weights
      - no_trade_band: turnover suppression band
      - sticky_buffer: keep some previous holdings
    """
    topk: int = 10
    softmax_temp: float = 0.8
    min_weight: float = 0.01
    no_trade_band: float = 0.02
    sticky_buffer: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topk": int(self.topk),
            "softmax_temp": float(self.softmax_temp),
            "min_weight": float(self.min_weight),
            "no_trade_band": float(self.no_trade_band),
            "sticky_buffer": int(self.sticky_buffer),
        }


def route_params(
    *,
    macro_liquidity: Optional[float],
    retail_sentiment: Optional[float],
    news_tda: Optional[float],
    base: RouterParams,
) -> RouterParams:
    """
    Public rule router (LLM stubbed):
    Inputs are DATE-ONLY proxies (should be <= D-1 to avoid lookahead).
      - macro_liquidity: liquidity proxy (higher => looser liquidity)
      - retail_sentiment: retail overheat proxy (higher => crowded/overheat risk)
      - news_tda: news topology complexity (higher => stress / narrative tearing)

    Returns RouterParams that adjust:
      - concentration (topk)
      - diversification (softmax_temp)
      - turnover controls (no_trade_band, sticky_buffer)
    """

    # normalize missing to neutral
    ml = float(macro_liquidity) if macro_liquidity is not None else 0.0
    rs = float(retail_sentiment) if retail_sentiment is not None else 0.0
    nt = float(news_tda) if news_tda is not None else 0.0

    # regime thresholds (public + explainable)
    LIQ_TIGHT = -0.8
    LIQ_LOOSE = 0.8

    RETAIL_OVERHEAT = 1.5
    RETAIL_PANIC = -1.5

    NEWS_STRESS = 1.5

    risk_off_score = 0
    risk_on_score = 0

    if ml <= LIQ_TIGHT:
        risk_off_score += 1
    elif ml >= LIQ_LOOSE:
        risk_on_score += 1

    if rs >= RETAIL_OVERHEAT:
        risk_off_score += 1
    elif rs <= RETAIL_PANIC:
        risk_on_score += 1

    if nt >= NEWS_STRESS:
        risk_off_score += 1

    # start from base
    topk = base.topk
    temp = base.softmax_temp
    min_w = base.min_weight
    band = base.no_trade_band
    sticky = base.sticky_buffer

    # risk-off: fewer names + flatter weights + trade less
    if risk_off_score >= 2:
        topk = _clamp_int(topk - 3, 5, 20)
        temp = _clamp_float(temp + 0.25, 0.5, 2.0)
        band = _clamp_float(band + 0.01, 0.0, 0.10)
        sticky = _clamp_int(sticky + 1, 0, 8)
        min_w = _clamp_float(min_w + 0.005, 0.0, 0.10)

    # mild risk-off
    elif risk_off_score == 1:
        topk = _clamp_int(topk - 1, 5, 20)
        temp = _clamp_float(temp + 0.10, 0.5, 2.0)
        band = _clamp_float(band + 0.005, 0.0, 0.10)
        sticky = _clamp_int(sticky + 0, 0, 8)

    # risk-on
    elif risk_on_score >= 2:
        topk = _clamp_int(topk + 2, 5, 25)
        temp = _clamp_float(temp - 0.10, 0.4, 2.0)
        band = _clamp_float(band - 0.005, 0.0, 0.10)
        sticky = _clamp_int(sticky - 1, 0, 8)
        min_w = _clamp_float(min_w, 0.0, 0.10)

    return RouterParams(
        topk=int(topk),
        softmax_temp=float(temp),
        min_weight=float(min_w),
        no_trade_band=float(band),
        sticky_buffer=int(sticky),
    )
