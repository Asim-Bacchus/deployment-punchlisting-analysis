from __future__ import annotations

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
    P_EXECUTION_CONSTRAINT,
    P_REVISIT_IF_PUNCHLISTED,
    P_REVISIT_SUCCESS_ON_SECOND,
)

OUT_PATH = Path("data/raw/tickets.csv")


def logistic(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def main() -> None:
    rng = np.random.default_rng(SEED)

    # ── Task applicability ────────────────────────────────────────────────────
    task_applicability = np.where(
        rng.random(N_ROWS) < P_INVALID_TARGET, "invalid_target", "valid_target"
    )

    # ── Department label quality ──────────────────────────────────────────────
    dept_label_quality = np.where(
        rng.random(N_ROWS) < P_DEPT_AMBIGUOUS, "ambiguous", "unambiguous"
    )

    # ── Human anchor ─────────────────────────────────────────────────────────
    human_anchor = np.where(rng.random(N_ROWS) < P_HUMAN_PRESENT, "present", "absent")

    # ── Spatial anchor (ambiguous depts degrade specificity slightly) ─────────
    spatial_draw = rng.random(N_ROWS)
    spatial_anchor = np.full(N_ROWS, "missing", dtype=object)
    spatial_anchor[spatial_draw < P_SPATIAL_SPECIFIC] = "specific"
    spatial_anchor[
        (spatial_draw >= P_SPATIAL_SPECIFIC)
        & (spatial_draw < P_SPATIAL_SPECIFIC + P_SPATIAL_VAGUE)
    ] = "vague"

    amb_mask = dept_label_quality == "ambiguous"
    degrade = (spatial_anchor == "specific") & amb_mask & (rng.random(N_ROWS) < 0.25)
    spatial_anchor[degrade] = "vague"

    # ── Execution constraint (INTRINSIC noise: workstation in use, locked, etc.) ──
    # This affects punchlisting regardless of anchor quality — it is structurally
    # unavoidable via better ticketing. This is what makes the intrinsic/preventable
    # distinction real rather than rhetorical.
    execution_constraint = rng.random(N_ROWS) < P_EXECUTION_CONSTRAINT

    # ── Punchlisting probability model ───────────────────────────────────────
    punchlisted = np.zeros(N_ROWS, dtype=int)

    for i in range(N_ROWS):
        # INTRINSIC score: noise that better information cannot eliminate
        intrinsic_score = -2.0  # baseline (low punchlisting)
        if execution_constraint[i]:
            intrinsic_score += 1.5  # locked room / workstation in use / patient present

        # PREVENTABLE score: anchor and information failures for valid targets
        preventable_score = 0.0
        if task_applicability[i] == "invalid_target":
            # Invalid targets punchlisted after on-site verification — not preventable
            # by anchor rules (requires fixing task assignment upstream, different problem)
            preventable_score += 3.0
        else:
            # Spatial anchor effect
            if spatial_anchor[i] == "missing":
                preventable_score += 1.2
            elif spatial_anchor[i] == "vague":
                preventable_score += 0.6

            # Human anchor effect
            if human_anchor[i] == "absent":
                preventable_score += 1.0

            # Interaction: both anchors missing is the high-risk scenario
            if (spatial_anchor[i] == "missing") and (human_anchor[i] == "absent"):
                preventable_score += 1.8

            # Ambiguous dept labels compound weak-anchor cases
            if dept_label_quality[i] == "ambiguous" and spatial_anchor[i] != "specific":
                preventable_score += 0.4

        p = logistic(intrinsic_score + preventable_score)
        punchlisted[i] = int(rng.random() < p)

    # ── Revisit logic (decoupled from punchlisted) ────────────────────────────
    # Not all punchlisted tickets become revisits — some are resolved administratively
    # (reassigned, cancelled, handled by phone). Of those that do revisit, most succeed.
    revisit_required = np.zeros(N_ROWS, dtype=int)
    revisit_successful = np.full(N_ROWS, np.nan, dtype=object)

    for i in range(N_ROWS):
        if punchlisted[i] == 1:
            if rng.random() < P_REVISIT_IF_PUNCHLISTED:
                revisit_required[i] = 1
                revisit_successful[i] = (
                    "yes" if rng.random() < P_REVISIT_SUCCESS_ON_SECOND else "no"
                )
            else:
                revisit_required[i] = 0
                revisit_successful[i] = "resolved_admin"

    # ── Assemble dataframe ────────────────────────────────────────────────────
    df = pd.DataFrame(
        {
            "ticket_id": np.arange(1, N_ROWS + 1),
            "task_applicability": task_applicability,
            "dept_label_quality": dept_label_quality,
            "spatial_anchor": spatial_anchor,
            "human_anchor": human_anchor,
            "execution_constraint": execution_constraint.astype(int),
            "punchlisted": punchlisted,
            "revisit_required": revisit_required,
            "revisit_successful": revisit_successful,
        }
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    # ── Sanity check ──────────────────────────────────────────────────────────
    overall = df["punchlisted"].mean()
    valid = df[df["task_applicability"] == "valid_target"]["punchlisted"].mean()
    invalid = df[df["task_applicability"] == "invalid_target"]["punchlisted"].mean()
    both_missing_valid = df[
        (df["task_applicability"] == "valid_target")
        & (df["spatial_anchor"] == "missing")
        & (df["human_anchor"] == "absent")
    ]["punchlisted"].mean()
    revisit_rate = df[df["punchlisted"] == 1]["revisit_required"].mean()

    print(f"Saved: {OUT_PATH}")
    print(f"Overall punchlisted rate:                        {overall:.3f}")
    print(f"Valid targets punchlisted rate:                  {valid:.3f}")
    print(f"Invalid targets punchlisted rate:                {invalid:.3f}")
    print(f"Valid targets / both anchors missing rate:       {both_missing_valid:.3f}")
    print(f"Revisit rate among punchlisted tickets:          {revisit_rate:.3f}")


if __name__ == "__main__":
    main()