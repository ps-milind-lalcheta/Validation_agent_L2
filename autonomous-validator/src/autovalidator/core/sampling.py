from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import pandas as pd


@dataclass
class LoadOptions:
    file_type: Optional[Literal["csv", "parquet"]] = None
    encoding: str = "utf-8"
    csv_sep: str = ","
    csv_na_values: Optional[list[str]] = None


def infer_file_type(dataset_ref: str) -> Literal["csv", "parquet"]:
    p = Path(dataset_ref)
    suffix = p.suffix.lower().lstrip(".")
    if suffix in ("csv",):
        return "csv"
    if suffix in ("parquet", "pq"):
        return "parquet"
    raise ValueError(f"Unsupported dataset type: {suffix}. Only CSV/Parquet supported in MVP.")


def load_dataframe(dataset_ref: str, options: Optional[LoadOptions] = None) -> pd.DataFrame:
    options = options or LoadOptions()
    ftype = options.file_type or infer_file_type(dataset_ref)

    if ftype == "csv":
        return pd.read_csv(
            dataset_ref,
            sep=options.csv_sep,
            encoding=options.encoding,
            na_values=options.csv_na_values,
            low_memory=False,
        )
    if ftype == "parquet":
        return pd.read_parquet(dataset_ref)
    raise ValueError(f"Unsupported file_type: {ftype}")


def sample_dataframe(df: pd.DataFrame, n: int = 500, seed: int = 42) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed).copy()


def load_and_sample(dataset_ref: str, n: int = 500, seed: int = 42, options: Optional[LoadOptions] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = load_dataframe(dataset_ref, options=options)
    sample = sample_dataframe(df, n=n, seed=seed)
    return df, sample
