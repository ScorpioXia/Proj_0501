"""Shared, side-effect-free input helpers for the current experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def normalise_patient_ids(values: pd.Series) -> pd.Series:
    """Canonicalize patient IDs without using names or pinyin."""
    result = values.astype("string").str.strip()
    result = result.mask(result.eq(""))
    numeric = pd.to_numeric(result, errors="coerce")
    integer_mask = numeric.notna() & np.isfinite(numeric) & np.equal(numeric % 1, 0)
    result = result.copy()
    result.loc[integer_mask] = numeric.loc[integer_mask].astype("Int64").astype("string")
    return result


def read_csv_compatible(path: Path, **kwargs) -> pd.DataFrame:
    """Read project CSV files using the two encodings present in the dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Required CSV does not exist: {path}")
    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is None:  # pragma: no cover
        raise RuntimeError(f"Unable to read CSV: {path}")
    raise last_error
