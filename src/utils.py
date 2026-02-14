# src/utils.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd


def punchlist_rate(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """
    Returns punchlisting rate (mean of punchlisted) by the given grouping columns.
    Output columns:
      - group_cols...
      - n (count of rows)
      - punchlist_rate (0..1)
    """
    if "punchlisted" not in df.columns:
        raise ValueError("Expected column 'punchlisted' in dataframe.")

    out = (
        df.groupby(group_cols, dropna=False)["punchlisted"]
        .agg(n="size", punchlist_rate="mean")
        .reset_index()
        .sort_values(group_cols)
        .reset_index(drop=True)
    )
    return out


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_table_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def save_plot(fig, path: Path) -> None:
    """
    Save a matplotlib figure to disk.
    """
    ensure_dir(path.parent)
    fig.savefig(path, bbox_inches="tight", dpi=200)
