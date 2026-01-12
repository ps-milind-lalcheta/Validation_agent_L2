from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import pandas as pd


EMAIL_RE = re.compile(r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")
# Very loose; used for masking only (not for asserting it's an address)
ADDRESS_HINT_RE = re.compile(r"(?i)\b(street|st\.|road|rd\.|ave|avenue|lane|ln\.|boulevard|blvd|sector|block|city|zip|postcode|pincode)\b")

SENSITIVE_COL_HINTS = (
    "name", "email", "phone", "mobile", "address", "ssn", "pan", "aadhaar", "passport",
    "patient", "medical", "report", "diagnosis", "dob", "birth", "mrn",
)


def _mask_string_keep_shape(s: str, keep: int = 2) -> str:
    s = str(s)
    if len(s) <= keep * 2:
        return "*" * len(s)
    return s[:keep] + ("*" * (len(s) - keep * 2)) + s[-keep:]


def _mask_email(s: str) -> str:
    # Keep domain shape but mask localpart
    m = EMAIL_RE.search(s)
    if not m:
        return _mask_string_keep_shape(s)
    email = m.group(0)
    local, domain = email.split("@", 1)
    return s.replace(email, _mask_string_keep_shape(local, 1) + "@" + domain)


def _mask_phone(s: str) -> str:
    m = PHONE_RE.search(s)
    if not m:
        return _mask_string_keep_shape(s)
    ph = m.group(0)
    digits = re.sub(r"\D", "", ph)
    masked_digits = digits[:2] + ("*" * max(0, len(digits) - 4)) + digits[-2:]
    return s.replace(ph, masked_digits)


def _mask_value(v: Any) -> Any:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        # preserve sign + rough magnitude; avoid leaking exact
        sign = "-" if float(v) < 0 else ""
        av = abs(float(v))
        if av == 0:
            return 0
        # bucket magnitude
        import math
        exp = int(math.floor(math.log10(av))) if av > 0 else 0
        return f"{sign}~1e{exp}"
    s = str(v)
    if EMAIL_RE.search(s):
        return _mask_email(s)
    if PHONE_RE.search(s):
        return _mask_phone(s)
    if ADDRESS_HINT_RE.search(s):
        return _mask_string_keep_shape(s, 1)
    # Default mask lightly (preserve a bit for pattern)
    return _mask_string_keep_shape(s, 2)


@dataclass
class RedactionResult:
    redacted_df: pd.DataFrame
    sensitive_column_flags: Dict[str, bool]
    sensitive_score: float  # 0..1


def redact_dataframe(df: pd.DataFrame, max_examples_per_col: int = 200) -> RedactionResult:
    """Best-effort redaction for sending small samples to LLM.

    - Masks values that look like email/phone/address.
    - If column name hints sensitivity (name, patient, etc.), masks all its non-null values.
    """
    redacted = df.copy()
    flags: Dict[str, bool] = {}
    sensitive_hits = 0
    total_cols = max(1, len(df.columns))

    for col in df.columns:
        col_l = str(col).lower()
        sensitive_col = any(h in col_l for h in SENSITIVE_COL_HINTS)
        flags[col] = sensitive_col
        if sensitive_col:
            sensitive_hits += 1

        series = df[col]
        # limit work for huge sample frames
        # apply masking row-wise on a limited slice then extend via vectorized operations
        # (MVP: simple apply)
        def mask_cell(x):
            if pd.isna(x):
                return None
            if sensitive_col:
                return _mask_value(x)
            # If not sensitive col, only mask when value itself looks sensitive
            s = str(x)
            if EMAIL_RE.search(s) or PHONE_RE.search(s) or ADDRESS_HINT_RE.search(s):
                return _mask_value(x)
            return x

        redacted[col] = series.apply(mask_cell)

    # sensitive_score: fraction of columns flagged sensitive
    sensitive_score = min(1.0, sensitive_hits / total_cols)
    return RedactionResult(redacted_df=redacted, sensitive_column_flags=flags, sensitive_score=sensitive_score)
