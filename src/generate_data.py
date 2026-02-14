# src/generate_data.py
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    SEED,
    N_ROWS,
    P_INVALID_TARGET,
    P_HUMAN_PRESENT,
    P_SPATIAL_SPECIFIC,
    P_SPATIAL_VAGUE,
    P_DEPT_AMBIGUOUS,
)

OUT_PATH = Path("data/raw/tickets.csv")


def logistic(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def main() -> None:
    rng = np.random.default_rng(SEED)

    # Task applicability
    task_applicability = np.where(
        rng.random(N_ROWS) < P_INVALID_TARGET, "invalid_target", "valid_target"
    )

    # Dept label quality (captures “Neurology has multiple locations”)
    dept_label_quality = np.where(
        rng.random(N_ROWS) < P_DEPT_AMBIGUOUS, "ambiguous", "unambiguous"
    )

    # Human anchor
    human_anchor = np.where(rng.random(N_ROWS) < P_HUMAN_PRESENT, "present", "absent")

    # Spatial anchor, with ambiguity nudging toward vagueness
    spatial_draw = rng.random(N_ROWS)
    spatial_anchor = np.full(N_ROWS, "missing", dtype=object)

    # Base spatial distribution
    specific_cut = P_SPATIAL_SPECIFIC
    vague_cut = P_SPATIAL_SPECIFIC + P_SPATIAL_VAGUE
    spatial_anchor[spatial_draw < specific_cut] = "specific"
    spatial_anchor[(spatial_draw >= specific_cut) & (spatial_draw < vague_cut)] = "vague"

    # Ambiguous departments degrade specificity a bit
    amb_mask = dept_label_quality == "ambiguous"
    degrade = (spatial_anchor == "specific") & amb_mask & (rng.random(N_ROWS) < 0.25)
    spatial_anchor[degrade] = "vague"

    # Punchlisting probability model
    # We intentionally make preventable punchlisting spike when both anchors are missing on valid targets.
    punchlisted = np.zeros(N_ROWS, dtype=int)
    revisit_required = np.zeros(N_ROWS, dtype=int)

    for i in range(N_ROWS):
        score = -2.0  # baseline: low punchlisting

        # Invalid targets are often punchlisted after verification
        if task_applicability[i] == "invalid_target":
            score += 2.0

        # Anchor effects for valid targets
        if task_applicability[i] == "valid_target":
            if spatial_anchor[i] == "missing":
                score += 1.2
            elif spatial_anchor[i] == "vague":
                score += 0.6

            if human_anchor[i] == "absent":
                score += 1.0

            # Interaction: both anchors missing is the nightmare
            if (spatial_anchor[i] == "missing") and (human_anchor[i] == "absent"):
                score += 1.8

            # Ambiguous dept labels make weak-anchor cases worse
            if dept_label_quality[i] == "ambiguous" and spatial_anchor[i] != "specific":
                score += 0.4

        p = logistic(score)
        pl = int(rng.random() < p)
        punchlisted[i] = pl
        revisit_required[i] = pl  # start simple

    df = pd.DataFrame(
        {
            "ticket_id": np.arange(1, N_ROWS + 1),
            "task_applicability": task_applicability,
            "dept_label_quality": dept_label_quality,
            "spatial_anchor": spatial_anchor,
            "human_anchor": human_anchor,
            "punchlisted": punchlisted,
            "revisit_required": revisit_required,
        }
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    # Quick sanity print so it feels real
    overall = df["punchlisted"].mean()
    valid = df[df["task_applicability"] == "valid_target"]["punchlisted"].mean()
    invalid = df[df["task_applicability"] == "invalid_target"]["punchlisted"].mean()
    both_missing_valid = df[
        (df["task_applicability"] == "valid_target")
        & (df["spatial_anchor"] == "missing")
        & (df["human_anchor"] == "absent")
    ]["punchlisted"].mean()

    print(f"Saved: {OUT_PATH}")
    print(f"Overall punchlisted rate: {overall:.3f}")
    print(f"Valid targets punchlisted rate: {valid:.3f}")
    print(f"Invalid targets punchlisted rate: {invalid:.3f}")
    print(f"Valid targets with both anchors missing punchlisted rate: {both_missing_valid:.3f}")


if __name__ == "__main__":
    main()
