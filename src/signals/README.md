# Signals (public demo)

This folder **does NOT** rebuild NLP factors.

The CSV files under `demo_data/factors/` are **already processed outputs** from the private pipeline
(e.g., NLP parsing / sentiment extraction / topic scoring / normalization). The public repo only
loads them to reproduce the end-to-end allocation pipeline.

## Factor CSVs (inputs)

All CSVs use `date` as the time key (YYYY-MM-DD). Some are panel-level (`date, code`), some are macro-level (`date` only):

### Panel-level (date + code)
- `etf_sector_sentiment_factor.csv`
- `etf_policy_vs_sent_gap_factor.csv`
- `etf_narrative_vs_price_gap_factor.csv`

### Macro-level (date only, broadcast to all codes in universe)
- `macro_liquidity_factor.csv`
- `news_tda_factor.csv`
- `retail_sentiment_factor.csv` (may contain duplicated `date` columns from export; loader will normalize)

## Leakage policy
By default, the loader applies **shift(1)** (use <= D-1 information) when building features for date D.
