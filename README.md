# Healthcare IT Deployment: Preventable Punchlisting Analysis

In large healthcare IT rollouts, deployment tickets routinely arrive with incomplete location and contact data. As a result, **punchlisting** (when a technician cannot complete a deployment in a single visit and has to come back) gets treated as a normal part of the job rather than a fixable problem.

This project asks a narrower question:

> **In a system where punchlisting is expected, what portion of it is structurally preventable through minimum viable information?**

Using a simulation grounded in real field workflows (2,000 synthetic deployment tickets), this analysis separates:

- **Intrinsic punchlisting** - failures caused by unavoidable on-site constraints like locked rooms, workstations currently in use, or patient care in progress. These cannot be fixed by improving the ticket.
- **Preventable punchlisting** - failures caused by missing upstream information that a technician needed before ever leaving for the site. These can be fixed cheaply.

The central finding: requiring just **one reliable anchor** per ticket, either a specific physical location or a human contact who can help on arrival, dramatically reduces punchlisting without needing to overhaul anything else.

**Stack:** Python, pandas, NumPy, statsmodels, matplotlib

---

## Thesis Visualization

![Punchlisting rate by anchor availability](outputs/figures/thesis_anchor_status_punchlist_rate.png)

---

## Key Findings (Valid Deployments Only)

Deployments flagged as invalid targets (devices that turned out not to need work) are excluded here. The focus is on tickets where work was genuinely required but still could not be completed.

| Condition | Punchlisting Rate |
|---|---|
| Both anchors present | 23% |
| One anchor present | 37% |
| **No anchors present** | **90.2%** |
| Overall | 34% |

When both a location and a human contact are missing from a ticket, punchlisting stops being a risk and starts being the expected outcome.

---

## Recommendation

### Minimum Anchor Rule

Require every deployment ticket to include at least one of:

- A specific physical location (room number, wing, floor)
- A designated human point of contact who will be reachable on arrival

Perfect information is not the goal. The data shows that going from zero anchors to one anchor is where the largest reduction happens. Going from one to two anchors is an improvement, but the jump from zero to one is where the policy pays off.

Supporting recommendations:

- Replace free-text department name fields with structured location inputs wherever possible
- When a ticket arrives with no anchor at all, have a coordinator reach out to a department manager to establish one before dispatching a technician

---

## Driver Quantification (Logistic Regression)

To go beyond rates and understand which missing information actually drives punchlisting, a logistic regression was fit on valid deployments using `statsmodels`. Logistic regression is appropriate here because the outcome is binary (punchlisted or not) and the goal is explanation rather than prediction. `statsmodels` was chosen over `scikit-learn` specifically because it produces odds ratios and confidence intervals, which are more useful for communicating effect sizes to non-technical stakeholders.

**Reference category:** ticket with a specific location and human contact present

| Predictor | Odds Ratio | Type |
|---|---|---|
| Spatial anchor missing | **6.32x** | Preventable |
| Spatial anchor vague | 2.32x | Preventable |
| Human contact absent | **3.26x** | Preventable |
| Execution constraint | 4.78x | Intrinsic |

**Pseudo R-squared: 0.14**

This is lower than you would see in a clean predictive model, but it is appropriate here. The simulation itself encodes an irreducible noise floor through the `execution_constraint` variable, which contributes to punchlisting independently of any information quality improvements. A model fit on data with intentional intrinsic variance should not be expected to explain all outcome variation, and one that did would likely be overfitting to noise rather than signal. The goal is to quantify the drivers that are actually actionable, not to predict every ticket outcome.

Key takeaways:

- Missing a specific location multiplies punchlisting odds by more than 6x compared to having one
- Lacking any human contact more than triples the odds
- Intrinsic execution constraints (the unavoidable stuff) also meaningfully increase failure risk, confirming that some punchlisting truly cannot be eliminated through process changes alone

---

## Counterfactual: Minimum Anchor Rule

To translate the findings into a practical impact estimate, a conservative counterfactual was computed: if all valid deployment tickets satisfied a **Minimum Anchor Rule** (at least one anchor required), how many punchlisted tickets would have been avoided?

Non-compliant tickets were assumed to perform at the current compliant-ticket rate if brought into compliance. This is a deliberate underestimate because it:

- Does not eliminate intrinsic constraints
- Does not assume perfect data
- Does not assume any change in technician behavior

Estimated impact:

> **Approximately 87 revisits avoided per 2,000 valid deployments, a 13% relative reduction**

That is meaningful downstream labor savings from a change that only requires adding one field to a ticket form.

---

## The Core Distinction

| Type | Cause | Fixable via Better Ticketing? |
|---|---|---|
| **Intrinsic** | Workstation in use, locked room, patient present | No |
| **Preventable** | Missing location, no contact, ambiguous department label | Yes |

This distinction is encoded structurally in the simulation rather than just described in the writeup. The `execution_constraint` variable affects punchlisting probability completely independently of anchor quality, representing the irreducible noise floor. Anchor and ambiguity variables layer on top of that baseline and represent the preventable signal.

> There exists a minimum viable information threshold. Below it, human judgment and field experience cannot compensate for missing data at scale.

---

## Operational Context

This analysis models a real healthcare IT deployment workflow:

- Ticketing system similar to FileMaker, operating at roughly 40% upstream data accuracy
- Field technicians deploying peripherals like tap badges and signature pads across large hospital departments with complex physical layouts
- Leadership explicitly acknowledged the data quality problem and treated punchlisting as an expected operational cost rather than a fixable one

The simulation preserves structural co-occurrences that reflect how real ticket failures cluster. Missing inventory data tends to co-occur with missing contact and location data. Ambiguous department labels (for example, a department name that maps to three different physical areas of the hospital) correlate with degraded spatial specificity. Intrinsic constraints operate independently of information quality because a locked room is a locked room regardless of how good the ticket is.

---

## Data Design

2,000 synthetic deployment tickets, one row per ticket.

| Variable | Description |
|---|---|
| `task_applicability` | Whether the deployment was a valid target or an incorrect assignment. Analysis focuses on valid targets only. |
| `spatial_anchor` | Quality of location information: specific (room number), vague (department name only), or missing entirely |
| `human_anchor` | Whether a reachable human contact was listed on the ticket |
| `dept_label_quality` | Whether the department label maps to one location (unambiguous) or multiple physical areas (ambiguous) |
| `execution_constraint` | Binary flag for intrinsic blockers like locked rooms or occupied workstations, present on roughly 12% of tickets |
| `punchlisted` | Primary outcome: was the technician unable to complete the deployment in one visit? |
| `revisit_required` | Probabilistic outcome: about 80% of punchlisted tickets generate an actual return visit. The remaining 20% are resolved administratively (reassigned, cancelled, handled by phone). |
| `revisit_successful` | Whether the return visit completed the deployment successfully |

---

## Methodology

**Descriptive analysis**
Punchlisting rates broken down by anchor availability, department label quality, and task applicability to establish baseline patterns before modeling.

**Logistic regression (statsmodels)**
Model fit on valid targets using spatial anchor (categorical), human anchor, department label quality, and execution constraint as predictors. The focus is on effect size and interpretability, not predictive accuracy.

**Counterfactual simulation**
Non-compliant tickets (those with no anchor) are assumed to reach the compliant-ticket punchlisting rate under a policy change. This produces a conservative lower-bound estimate of the policy impact.

---

## How to Reproduce

```bash
pip install -r requirements.txt
python src/generate_data.py
python src/analyze.py
```

Outputs are written to `outputs/figures/`, `outputs/tables/`, and `outputs/summary.json`.

---

## Project Structure

```
deployment-punchlisting-analysis/
├── src/
│   ├── config.py           # all parameters and base rates
│   ├── generate_data.py    # simulation with labeled intrinsic/preventable score components
│   ├── analyze.py          # rates, figures, logistic regression, counterfactual
│   └── utils.py            # shared helpers
├── data/raw/tickets.csv
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── summary.json
├── requirements.txt
└── README.md
```

---

## Limitations

- Simulated data cannot capture full behavioral variability. An unresponsive contact, for example, would appear as "human anchor present" in the data but still cause punchlisting in practice. This means the true impact of the human anchor variable is likely understated, and the counterfactual reduction estimate is conservative for this reason as well.
- Findings are explanatory, not predictive. The model quantifies which information gaps drive punchlisting, but it should not be used to forecast whether any individual ticket will be punchlisted.
- The counterfactual assumes anchor compliance is enforceable upstream, which depends on the ticketing system and team workflow. In environments where submitters routinely skip fields, the policy impact would require enforcement mechanisms beyond the rule itself.
- No temporal or technician-level variation is modeled. Day-of-week effects, shift patterns, and individual technician experience are all real factors not captured here. A production version of this analysis would want random effects for technician to isolate information quality from individual performance.
