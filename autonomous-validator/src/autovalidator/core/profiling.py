from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd


InferredType = Literal["string", "integer", "number", "boolean", "date", "datetime", "unknown"]


def infer_type(s: pd.Series) -> InferredType:
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_integer_dtype(s):
        return "integer"
    if pd.api.types.is_float_dtype(s):
        return "number"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    # Try parse dates if strings look like dates
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        sample = s.dropna().astype(str).head(50)
        if sample.empty:
            return "unknown"
        # heuristic: if most parse as date
        parsed = pd.to_datetime(sample, errors="coerce", utc=False)
        ok = parsed.notna().mean()
        if ok > 0.9:
            return "date"
        return "string"
    return "unknown"


def pattern_fingerprint(x: str) -> str:
    out = []
    for ch in x:
        if ch.isalpha():
            out.append("A")
        elif ch.isdigit():
            out.append("9")
        elif ch.isspace():
            out.append(" ")
        else:
            out.append(ch)
    # compress long
    fp = "".join(out)
    return fp[:64]


@dataclass
class ColumnProfile:
    name: str
    inferred_type: InferredType
    null_rate: float
    unique_rate: float
    example_values_masked: list[str]
    top_values_masked: list[str]
    length_min: Optional[int]
    length_max: Optional[int]
    pattern_samples: list[str]
    numeric_stats: Optional[Dict[str, float]]
    date_stats: Optional[Dict[str, str]]


def profile_dataframe(df_masked: pd.DataFrame, dataset_name: str, row_count_estimate: int) -> Dict[str, Any]:
    cols = []
    n = len(df_masked)
    for col in df_masked.columns:
        s = df_masked[col]
        it = infer_type(s)
        null_rate = float(s.isna().mean()) if n else 0.0
        nunique = s.nunique(dropna=True)
        unique_rate = float(nunique / max(1, (n - int(s.isna().sum())))) if n else 0.0

        # examples (masked already)
        examples = [str(v) for v in s.dropna().astype(str).head(3).tolist()]

        # top values (capped)
        top_vals = []
        try:
            vc = s.dropna().astype(str).value_counts().head(5)
            top_vals = vc.index.tolist()
        except Exception:
            top_vals = []

        length_min = length_max = None
        pattern_samples = []
        if it == "string":
            ss = s.dropna().astype(str)
            if not ss.empty:
                lens = ss.map(len)
                length_min = int(lens.min())
                length_max = int(lens.max())
                pattern_samples = [pattern_fingerprint(x) for x in ss.head(10).tolist()]

        numeric_stats = None
        if it in ("integer", "number"):
            sn = pd.to_numeric(s, errors="coerce")
            sn = sn.dropna()
            if not sn.empty:
                q = sn.quantile([0.05, 0.5, 0.95]).to_dict()
                numeric_stats = {
                    "min": float(sn.min()),
                    "max": float(sn.max()),
                    "mean": float(sn.mean()),
                    "std": float(sn.std(ddof=1)) if len(sn) > 1 else 0.0,
                    "p05": float(q.get(0.05, np.nan)),
                    "p50": float(q.get(0.5, np.nan)),
                    "p95": float(q.get(0.95, np.nan)),
                }

        date_stats = None
        if it in ("date", "datetime"):
            sd = pd.to_datetime(s, errors="coerce")
            sd = sd.dropna()
            if not sd.empty:
                date_stats = {
                    "min": str(sd.min()),
                    "max": str(sd.max()),
                }

        cols.append(
            {
                "name": str(col),
                "inferred_type": it,
                "example_values_masked": examples,
                "null_rate": null_rate,
                "unique_rate": unique_rate,
                "top_values_masked": top_vals,
                "length_min": length_min,
                "length_max": length_max,
                "pattern_samples": pattern_samples,
                "numeric_stats": numeric_stats,
                "date_stats": date_stats,
            }
        )

    return {
        "dataset_name": dataset_name,
        "row_count_estimate": int(row_count_estimate),
        "columns": cols,
        "notes": ["profile computed on masked sample"],
    }
