# src/analyze.py
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from utils import punchlist_rate, save_plot, save_table_csv

DATA_PATH = Path("data/raw/tickets.csv")
FIG_DIR = Path("outputs/figures")
TAB_DIR = Path("outputs/tables")


def add_anchor_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - anchor_status: both_present / one_present / none_present
      - anchor_combo: 6-state combo for plotting
    """
    df = df.copy()
    spatial_missing = df["spatial_anchor"].eq("missing")
    human_absent = df["human_anchor"].eq("absent")

    # Anchor status (3 states)
    df["anchor_status"] = "one_present"
    df.loc[~spatial_missing & ~human_absent, "anchor_status"] = "both_present"
    df.loc[spatial_missing & human_absent, "anchor_status"] = "none_present"

    # Anchor combo (fine-grained, 6 states)
    df["anchor_combo"] = df["spatial_anchor"].astype(str) + " + " + df["human_anchor"].astype(str)
    return df


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Run: python src\\generate_data.py"
        )

    df = pd.read_csv(DATA_PATH)

    # Add derived labels
    df = add_anchor_status(df)

    # 1) Table: punchlist rate by task_applicability (all rows)
    t1 = punchlist_rate(df, ["task_applicability"])
    save_table_csv(t1, TAB_DIR / "punchlist_rate_by_task_applicability.csv")

    # Focus analysis on preventable punchlisting: valid targets only
    df_valid = df[df["task_applicability"] == "valid_target"].copy()

    # 2) Table: punchlist rate by (spatial_anchor, human_anchor) for valid targets
    t2 = punchlist_rate(df_valid, ["spatial_anchor", "human_anchor"])
    save_table_csv(t2, TAB_DIR / "punchlist_rate_valid_by_spatial_and_human.csv")

    # 3) Table: punchlist rate by dept_label_quality and anchor_status (valid targets)
    if "dept_label_quality" in df_valid.columns:
        t3 = punchlist_rate(df_valid, ["dept_label_quality", "anchor_status"])
        save_table_csv(t3, TAB_DIR / "punchlist_rate_valid_by_ambiguity_and_anchor_status.csv")
    else:
        t3 = None

    # -------------------------
    # Figures (minimal, decision-focused)
    # -------------------------

    # Fig 1: punchlist rate by anchor combo (valid targets)
    fig1_df = (
        df_valid.groupby("anchor_combo")["punchlisted"]
        .mean()
        .sort_values(ascending=False)
        .reset_index(name="punchlist_rate")
    )
    fig1, ax1 = plt.subplots()
    ax1.bar(fig1_df["anchor_combo"], fig1_df["punchlist_rate"])
    ax1.set_title("Punchlisting rate by anchor combo (valid targets)")
    ax1.set_xlabel("Anchor combo (spatial + human)")
    ax1.set_ylabel("Punchlisting rate")
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis="x", rotation=35, labelsize=9)
    save_plot(fig1, FIG_DIR / "punchlist_rate_by_anchor_combo_valid.png")
    plt.close(fig1)

    # Fig 2: punchlist rate by spatial anchor (valid targets)
    fig2_df = (
        df_valid.groupby("spatial_anchor")["punchlisted"]
        .mean()
        .reindex(["specific", "vague", "missing"])
        .reset_index(name="punchlist_rate")
    )
    fig2, ax2 = plt.subplots()
    ax2.bar(fig2_df["spatial_anchor"], fig2_df["punchlist_rate"])
    ax2.set_title("Punchlisting rate by spatial anchor (valid targets)")
    ax2.set_xlabel("Spatial anchor")
    ax2.set_ylabel("Punchlisting rate")
    ax2.set_ylim(0, 1)
    save_plot(fig2, FIG_DIR / "punchlist_rate_by_spatial_anchor_valid.png")
    plt.close(fig2)

    # Fig 3 (optional but strong): ambiguity effect when anchors are weak
    if "dept_label_quality" in df_valid.columns:
        weak = df_valid[df_valid["anchor_status"].isin(["none_present", "one_present"])].copy()
        fig3_df = (
            weak.groupby(["dept_label_quality", "anchor_status"])["punchlisted"]
            .mean()
            .reset_index(name="punchlist_rate")
        )

        # Simple grouped bars: plot each anchor_status as its own bars per dept quality
        pivot = fig3_df.pivot(index="dept_label_quality", columns="anchor_status", values="punchlist_rate").fillna(0)
        # Ensure stable column order
        col_order = [c for c in ["none_present", "one_present"] if c in pivot.columns]
        pivot = pivot[col_order]

        fig3, ax3 = plt.subplots()
        x = range(len(pivot.index))
        width = 0.35 if len(col_order) == 2 else 0.6

        for j, col in enumerate(col_order):
            offsets = [i + (j - (len(col_order)-1)/2) * width for i in x]
            ax3.bar(offsets, pivot[col].values, width=width, label=col)

        ax3.set_title("Ambiguity increases punchlisting when anchors are weak (valid targets)")
        ax3.set_xlabel("Department label quality")
        ax3.set_ylabel("Punchlisting rate")
        ax3.set_ylim(0, 1)
        ax3.set_xticks(list(x))
        ax3.set_xticklabels(list(pivot.index))
        ax3.legend(title="Anchor status")
        save_plot(fig3, FIG_DIR / "punchlist_rate_ambiguity_effect_weak_anchors.png")
        plt.close(fig3)

    # -------------------------
    # Headline findings to terminal
    # -------------------------
    overall_valid = df_valid["punchlisted"].mean()

    both_missing_rate = df_valid[
        (df_valid["spatial_anchor"] == "missing") & (df_valid["human_anchor"] == "absent")
    ]["punchlisted"].mean()

    at_least_one_anchor_rate = df_valid[
        ~((df_valid["spatial_anchor"] == "missing") & (df_valid["human_anchor"] == "absent"))
    ]["punchlisted"].mean()

    print("Headline findings (valid targets):")
    print(f"- Overall punchlisting rate: {overall_valid:.3f}")
    print(f"- Both anchors missing punchlisting rate: {both_missing_rate:.3f}")
    print(f"- At least one anchor present punchlisting rate: {at_least_one_anchor_rate:.3f}")

    if t3 is not None:
        # crude ambiguity delta in weak-anchor subset
        weak = df_valid[df_valid["anchor_status"].isin(["none_present", "one_present"])].copy()
        amb = weak[weak["dept_label_quality"] == "ambiguous"]["punchlisted"].mean()
        unamb = weak[weak["dept_label_quality"] == "unambiguous"]["punchlisted"].mean()
        print(f"- Weak-anchor ambiguity lift (ambiguous minus unambiguous): {(amb - unamb):.3f}")

    print(f"\nSaved tables to: {TAB_DIR}")
    print(f"Saved figures to: {FIG_DIR}")


if __name__ == "__main__":
    main()
