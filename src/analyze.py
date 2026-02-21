from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from utils import punchlist_rate, save_plot, save_table_csv

DATA_PATH = Path("data/raw/tickets.csv")
FIG_DIR = Path("outputs/figures")
TAB_DIR = Path("outputs/tables")
SUMMARY_PATH = Path("outputs/summary.json")


# ── Helpers ───────────────────────────────────────────────────────────────────

def add_anchor_status(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    spatial_missing = df["spatial_anchor"].eq("missing")
    human_absent = df["human_anchor"].eq("absent")

    df["anchor_status"] = "one_present"
    df.loc[~spatial_missing & ~human_absent, "anchor_status"] = "both_present"
    df.loc[spatial_missing & human_absent, "anchor_status"] = "none_present"

    df["anchor_combo"] = (
        df["spatial_anchor"].astype(str) + " + " + df["human_anchor"].astype(str)
    )
    return df


def encode_for_regression(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode features for logistic regression.
    Spatial anchor is treated as categorical (C()) with 'specific' as reference.
    Human anchor and dept quality are binary.
    No redundant interaction terms — the categorical encoding for spatial_anchor
    combined with human_absent gives us a clean, non-overlapping feature set.
    """
    df = df.copy()
    # Set reference levels explicitly for interpretability
    df["spatial_anchor"] = pd.Categorical(
        df["spatial_anchor"], categories=["specific", "vague", "missing"]
    )
    df["human_absent"] = (df["human_anchor"] == "absent").astype(int)
    df["dept_ambiguous"] = (df["dept_label_quality"] == "ambiguous").astype(int)
    return df


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_anchor_status_thesis(df_valid: pd.DataFrame) -> None:
    """
    THE thesis chart: 3 bars showing punchlisting rate by anchor status.
    This is the visual that makes the minimum viable anchor argument.
    """
    order = ["both_present", "one_present", "none_present"]
    labels = ["Both anchors\npresent", "One anchor\npresent", "No anchors\npresent"]
    colors = ["#4CAF50", "#FF9800", "#F44336"]

    rates = (
        df_valid.groupby("anchor_status")["punchlisted"]
        .mean()
        .reindex(order)
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, rates.values, color=colors, width=0.5, edgecolor="white")

    # Annotate bar tops
    for bar, rate in zip(bars, rates.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{rate:.0%}",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )

    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_ylabel("Punchlisting rate", fontsize=11)
    ax.set_title(
        "Punchlisting rate by anchor availability\n(valid targets only)",
        fontsize=13,
        fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(df_valid["punchlisted"].mean(), color="gray", linestyle="--", linewidth=1)
    ax.text(
        2.4,
        df_valid["punchlisted"].mean() + 0.015,
        f"Overall avg: {df_valid['punchlisted'].mean():.0%}",
        color="gray",
        fontsize=9,
    )

    save_plot(fig, FIG_DIR / "thesis_anchor_status_punchlist_rate.png")
    plt.close(fig)


def fig_anchor_combo(df_valid: pd.DataFrame) -> None:
    """Detailed 6-combo breakdown — useful for appendix/deep dive."""
    fig1_df = (
        df_valid.groupby("anchor_combo")["punchlisted"]
        .mean()
        .sort_values(ascending=False)
        .reset_index(name="punchlist_rate")
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(fig1_df["anchor_combo"], fig1_df["punchlist_rate"], color="#5C85D6")
    ax.set_title("Punchlisting rate by anchor combination (valid targets)")
    ax.set_xlabel("Anchor combo (spatial + human)")
    ax.set_ylabel("Punchlisting rate")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    save_plot(fig, FIG_DIR / "punchlist_rate_by_anchor_combo_valid.png")
    plt.close(fig)


def fig_ambiguity_weak_anchors(df_valid: pd.DataFrame) -> None:
    """Ambiguity effect when anchors are weak."""
    weak = df_valid[df_valid["anchor_status"].isin(["none_present", "one_present"])].copy()
    fig3_df = (
        weak.groupby(["dept_label_quality", "anchor_status"])["punchlisted"]
        .mean()
        .reset_index(name="punchlist_rate")
    )
    pivot = (
        fig3_df.pivot(index="dept_label_quality", columns="anchor_status", values="punchlist_rate")
        .fillna(0)
    )
    col_order = [c for c in ["none_present", "one_present"] if c in pivot.columns]
    pivot = pivot[col_order]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(pivot.index))
    width = 0.35
    colors = ["#F44336", "#FF9800"]

    for j, col in enumerate(col_order):
        offsets = x + (j - (len(col_order) - 1) / 2) * width
        ax.bar(offsets, pivot[col].values, width=width, label=col, color=colors[j])

    ax.set_title("Ambiguity increases punchlisting when anchors are weak\n(valid targets)")
    ax.set_xlabel("Department label quality")
    ax.set_ylabel("Punchlisting rate")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.legend(title="Anchor status")
    ax.spines[["top", "right"]].set_visible(False)
    save_plot(fig, FIG_DIR / "punchlist_rate_ambiguity_effect_weak_anchors.png")
    plt.close(fig)


# ── Logistic Regression ───────────────────────────────────────────────────────

def run_logistic_regression(df_valid: pd.DataFrame) -> dict:
    """
    Logistic regression predicting punchlisted on valid targets.
    Uses statsmodels for odds ratios and confidence intervals.
    Returns a dict of key findings.
    """
    df_model = encode_for_regression(df_valid)

    # C(spatial_anchor) uses 'specific' as reference. Coefficients for 'vague' and
    # 'missing' are directly interpretable: odds relative to having a specific location.
    formula = (
        "punchlisted ~ C(spatial_anchor) + human_absent + dept_ambiguous "
        "+ execution_constraint"
    )

    model = smf.logit(formula, data=df_model).fit(disp=0)

    # Odds ratios and 95% CIs
    odds_ratios = np.exp(model.params)
    ci = np.exp(model.conf_int())
    ci.columns = ["OR_lower", "OR_upper"]

    results_df = pd.concat([odds_ratios.rename("odds_ratio"), ci], axis=1).reset_index()
    results_df.columns = ["feature", "odds_ratio", "OR_lower", "OR_upper"]
    results_df["p_value"] = model.pvalues.values

    save_table_csv(results_df, TAB_DIR / "logistic_regression_odds_ratios.csv")

    # Key interpretable numbers for resume/README
    missing_or = odds_ratios.get("C(spatial_anchor)[T.missing]", float("nan"))
    vague_or = odds_ratios.get("C(spatial_anchor)[T.vague]", float("nan"))
    human_absent_or = odds_ratios.get("human_absent", float("nan"))
    exec_or = odds_ratios.get("execution_constraint", float("nan"))

    print("\n── Logistic Regression (valid targets) ─────────────────────────")
    print(model.summary2())
    print(f"\nOdds ratios (relative to 'specific' spatial anchor + anchor present):")
    print(f"  spatial_anchor=missing:      {missing_or:.2f}x")
    print(f"  spatial_anchor=vague:        {vague_or:.2f}x")
    print(f"  human_anchor=absent:         {human_absent_or:.2f}x")
    print(f"  execution_constraint:        {exec_or:.2f}x")

    return {
        "spatial_missing_OR": round(float(missing_or), 2),
        "spatial_vague_OR": round(float(vague_or), 2),
        "human_absent_OR": round(float(human_absent_or), 2),
        "execution_constraint_OR": round(float(exec_or), 2),
    }


# ── Counterfactual ────────────────────────────────────────────────────────────

def compute_counterfactual(df_valid: pd.DataFrame) -> dict:
    """
    Minimum Anchor Rule counterfactual:
    If all valid-target tickets had at least one anchor,
    how many punchlisted tickets would we avoid per 2,000 deployments?
    """
    n_valid = len(df_valid)
    current_rate = df_valid["punchlisted"].mean()

    compliant = df_valid[df_valid["anchor_status"] != "none_present"]
    compliant_rate = compliant["punchlisted"].mean()

    non_compliant_n = len(df_valid[df_valid["anchor_status"] == "none_present"])
    non_compliant_rate = df_valid[df_valid["anchor_status"] == "none_present"]["punchlisted"].mean()

    # Conservative counterfactual: assume non-compliant tickets would drop to the
    # compliant rate if the Minimum Anchor Rule were enforced upstream.
    # compliant_rate * n_valid = compliant_rate * len(compliant) + compliant_rate * non_compliant_n
    # — simplified here for clarity.
    current_punchlisted = int(round(current_rate * n_valid))
    counterfactual_punchlisted = int(round(compliant_rate * n_valid))
    tickets_avoided = current_punchlisted - counterfactual_punchlisted

    # Scale to 2,000 valid deployments for the business case
    scale_factor = 2000 / n_valid
    avoided_per_2k = int(round(tickets_avoided * scale_factor))

    print("\n── Counterfactual: Minimum Anchor Rule ──────────────────────────")
    print(f"Current punchlisting rate (valid targets):     {current_rate:.3f}")
    print(f"Compliant ticket punchlisting rate:            {compliant_rate:.3f}")
    print(f"Non-compliant ticket punchlisting rate:        {non_compliant_rate:.3f}")
    print(f"Tickets avoided in this sample ({n_valid}):    {tickets_avoided}")
    print(f"Estimated revisits avoided per 2,000 valid deployments: {avoided_per_2k}")
    print(f"Relative reduction: {(current_rate - compliant_rate) / current_rate:.1%}")

    return {
        "current_punchlist_rate_valid": round(current_rate, 3),
        "compliant_punchlist_rate": round(compliant_rate, 3),
        "non_compliant_punchlist_rate": round(non_compliant_rate, 3),
        "tickets_avoided_per_2000": avoided_per_2k,
        "relative_reduction_pct": round(
            (current_rate - compliant_rate) / current_rate * 100, 1
        ),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run: python src/generate_data.py")

    df = pd.read_csv(DATA_PATH)
    df = add_anchor_status(df)

    # ── Tables ────────────────────────────────────────────────────────────────
    t1 = punchlist_rate(df, ["task_applicability"])
    save_table_csv(t1, TAB_DIR / "punchlist_rate_by_task_applicability.csv")

    df_valid = df[df["task_applicability"] == "valid_target"].copy()

    t2 = punchlist_rate(df_valid, ["spatial_anchor", "human_anchor"])
    save_table_csv(t2, TAB_DIR / "punchlist_rate_valid_by_spatial_and_human.csv")

    t3 = punchlist_rate(df_valid, ["dept_label_quality", "anchor_status"])
    save_table_csv(t3, TAB_DIR / "punchlist_rate_valid_by_ambiguity_and_anchor_status.csv")

    t4 = punchlist_rate(df_valid, ["anchor_status"])
    save_table_csv(t4, TAB_DIR / "punchlist_rate_valid_by_anchor_status.csv")

    # ── Figures ───────────────────────────────────────────────────────────────
    fig_anchor_status_thesis(df_valid)   # THE thesis chart
    fig_anchor_combo(df_valid)
    fig_ambiguity_weak_anchors(df_valid)

    # ── Logistic Regression ───────────────────────────────────────────────────
    lr_findings = run_logistic_regression(df_valid)

    # ── Counterfactual ────────────────────────────────────────────────────────
    cf_findings = compute_counterfactual(df_valid)

    # ── Headline terminal output ──────────────────────────────────────────────
    both_missing_rate = df_valid[
        (df_valid["spatial_anchor"] == "missing") & (df_valid["human_anchor"] == "absent")
    ]["punchlisted"].mean()

    at_least_one_rate = df_valid[
        df_valid["anchor_status"] != "none_present"
    ]["punchlisted"].mean()

    weak = df_valid[df_valid["anchor_status"].isin(["none_present", "one_present"])]
    amb_lift = (
        weak[weak["dept_label_quality"] == "ambiguous"]["punchlisted"].mean()
        - weak[weak["dept_label_quality"] == "unambiguous"]["punchlisted"].mean()
    )

    print("\n── Headline Findings (valid targets) ────────────────────────────")
    print(f"Overall punchlisting rate:              {df_valid['punchlisted'].mean():.3f}")
    print(f"Both anchors missing rate:              {both_missing_rate:.3f}")
    print(f"At least one anchor present rate:       {at_least_one_rate:.3f}")
    print(f"Ambiguity lift (weak-anchor subset):    {amb_lift:.3f}")

    # ── Save summary JSON ─────────────────────────────────────────────────────
    summary = {
        "headline": {
            "overall_valid_punchlist_rate": round(df_valid["punchlisted"].mean(), 3),
            "both_anchors_missing_rate": round(both_missing_rate, 3),
            "at_least_one_anchor_rate": round(at_least_one_rate, 3),
            "ambiguity_lift_weak_anchor_subset": round(amb_lift, 3),
        },
        "logistic_regression": lr_findings,
        "counterfactual": cf_findings,
    }

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved tables to: {TAB_DIR}")
    print(f"Saved figures to: {FIG_DIR}")
    print(f"Saved summary to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()