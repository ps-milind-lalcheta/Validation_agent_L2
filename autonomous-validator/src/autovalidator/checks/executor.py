from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from autovalidator.llm.schemas import CandidateCheck, EvidenceResult


def _cap_examples(rows: pd.DataFrame, max_examples: int = 5) -> list[dict[str, Any]]:
    if rows.empty:
        return []
    return rows.head(max_examples).to_dict(orient="records")


def execute_check(df: pd.DataFrame, check: CandidateCheck) -> EvidenceResult:
    ctype = check.check_type
    params = check.params
    notes: list[str] = []
    evaluated = 0
    violations = pd.Series(dtype=bool)

    def safe_col(name: str) -> pd.Series:
        if name not in df.columns:
            notes.append(f"Missing column: {name}")
            return pd.Series([np.nan] * len(df))
        return df[name]
    
    def resolve_col(p: dict) -> str:
        if "column" in p: return p["column"]
        if "columns" in p:
            val = p["columns"]
            if isinstance(val, list) and val: return val[0]
            if isinstance(val, str): return val
        return "unknown_column"

    if ctype == "not_null":
        col = resolve_col(params)
        s = safe_col(col)
        evaluated = len(s)
        violations = s.isna()
    elif ctype == "unique":
        cols = params.get("columns")
        if cols is None and "column" in params:
            cols = [params["column"]]

        if not isinstance(cols, list) or not cols:
            notes.append("Invalid columns param")
            evaluated = len(df)
            violations = pd.Series([False]*len(df))
        else:
            for col in cols:
                if col not in df.columns:
                    notes.append(f"Missing column: {col}")
            evaluated = len(df)
            dup = df.duplicated(subset=cols, keep=False)
            violations = dup
    elif ctype == "value_range":
        col = resolve_col(params)
        min_v = params.get("min")
        max_v = params.get("max")
        s = pd.to_numeric(safe_col(col), errors="coerce")
        evaluated = int(s.notna().sum())
        violations = pd.Series([False]*len(df))
        mask = s.notna()
        if min_v is not None:
            violations = violations | (mask & (s < float(min_v)))
        if max_v is not None:
            violations = violations | (mask & (s > float(max_v)))
    elif ctype == "length_range":
        col = resolve_col(params)
        min_len = params.get("min_len")
        max_len = params.get("max_len")
        s = safe_col(col).astype(str)
        mask = ~safe_col(col).isna()
        lens = s.map(len)
        evaluated = int(mask.sum())
        violations = pd.Series([False]*len(df))
        if min_len is not None:
            violations = violations | (mask & (lens < int(min_len)))
        if max_len is not None:
            violations = violations | (mask & (lens > int(max_len)))
    elif ctype == "regex_match":
        col = resolve_col(params)
        pattern = params.get("pattern")
        if not pattern:
            notes.append("Missing pattern for regex_match")
            pattern = ".*" # Dummy pattern
        sraw = safe_col(col)
        mask = ~sraw.isna()
        s = sraw.astype(str)
        evaluated = int(mask.sum())
        try:
            rx = re.compile(pattern)
            ok = s.apply(lambda x: bool(rx.search(x)))
            violations = mask & (~ok)
        except re.error:
            notes.append("Invalid regex pattern")
            violations = pd.Series([False]*len(df))
    elif ctype == "outlier_iqr":
        col = resolve_col(params)
        s = pd.to_numeric(safe_col(col), errors="coerce").dropna()
        evaluated = len(s)
        if evaluated < 8:
            notes.append("Too few numeric points for IQR outlier detection")
            violations = pd.Series([False]*len(df))
        else:
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
            sfull = pd.to_numeric(safe_col(col), errors="coerce")
            mask = sfull.notna()
            violations = mask & ((sfull < lo) | (sfull > hi))
            notes.append(f"IQR bounds: [{lo:.4g}, {hi:.4g}]")
    elif ctype == "date_not_future":
        col = resolve_col(params)
        now = datetime.now(timezone.utc)
        s = pd.to_datetime(safe_col(col), errors="coerce", utc=True)
        mask = s.notna()
        evaluated = int(mask.sum())
        violations = mask & (s > now)
    elif ctype == "category_newness":
        col = resolve_col(params)
        allowed = params.get("allowed_values", [])
        sraw = safe_col(col)
        mask = ~sraw.isna()
        s = sraw.astype(str)
        evaluated = int(mask.sum())
        if not allowed:
            # fallback: treat top N in sample as baseline; cannot measure newness
            notes.append("No baseline allowed_values provided; newness check is weak in MVP")
            violations = pd.Series([False]*len(df))
        else:
            violations = mask & (~s.isin(set(allowed)))
    elif ctype == "cross_field_dependency":
        if_col = resolve_col({"column": params.get("if_present", "unknown")})
        then_col = resolve_col({"column": params.get("then_required", "unknown")})
        s_if = safe_col(if_col)
        s_then = safe_col(then_col)
        mask = ~s_if.isna()
        evaluated = int(mask.sum())
        violations = mask & s_then.isna()
    else:
        notes.append(f"Unsupported check_type: {ctype}")
        evaluated = len(df)
        violations = pd.Series([False]*len(df))

    violation_count = int(violations.sum()) if len(df) else 0
    violation_rate = float(violation_count / max(1, evaluated)) if evaluated else 0.0

    examples = _cap_examples(df[violations], max_examples=5) if len(df) else []
    return EvidenceResult(
        check_id=check.check_id,
        check_type=check.check_type,
        total_evaluated=int(evaluated),
        violation_count=violation_count,
        violation_rate=violation_rate,
        examples_masked=examples,
        notes=notes,
    )


def execute_checks(df: pd.DataFrame, checks: list[CandidateCheck]) -> list[EvidenceResult]:
    results: list[EvidenceResult] = []
    for chk in checks:
        results.append(execute_check(df, chk))
    return results
