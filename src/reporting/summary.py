from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def dump_result_index(outdir: Path, res: Dict[str, Any]) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    idx: Dict[str, Any] = {}
    for k, v in res.items():
        if isinstance(v, pd.DataFrame):
            fp = outdir / f"{k}.csv"
            v.to_csv(fp, index=False)
            idx[k] = fp.name
        elif isinstance(v, pd.Series):
            fp = outdir / f"{k}.csv"
            v.to_frame(name=k).reset_index().to_csv(fp, index=False)
            idx[k] = fp.name
        else:
            try:
                json.dumps(v)
                idx[k] = v
            except TypeError:
                idx[k] = str(v)

    fp = outdir / "backtest_result.json"
    fp.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")
    return fp
