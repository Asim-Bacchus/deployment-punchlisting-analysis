from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def punchlist_rate(df: pd.DataFrame, groupby: list[str]) -> pd.DataFrame:
    return (
        df.groupby(groupby)["punchlisted"]
        .agg(punchlist_rate="mean", n="count")
        .reset_index()
    )


def save_plot(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)


def save_table_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)