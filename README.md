Healthcare IT Deployment: Preventable Punchlisting Analysis

In large healthcare IT rollouts, deployment tickets routinely arrive with incomplete location and contact data. As a result, punchlisting — the inability to complete a deployment in a single visit — is treated as operationally normal.

This project asks a narrower question:

In a system where punchlisting is expected, what portion of it is structurally preventable through minimum viable information?

Using a structured simulation grounded in real field workflows (2,000 synthetic deployment tickets), this analysis separates:

Intrinsic punchlisting — unavoidable execution constraints (locked rooms, workstations in use, patient present)

Preventable punchlisting — failures caused by insufficient upstream information

The central finding: requiring just one reliable anchor — either a specific physical location or a human contact — dramatically reduces punchlisting without requiring perfect upstream data.

Thesis Visualization

Key Findings (Valid Deployments Only)
Condition	Punchlisting Rate
Both anchors present	19%
One anchor present	33%
No anchors present	90.2%
Overall	33.5%

When both spatial and human anchors are missing, punchlisting becomes structurally likely.

Driver Quantification (Logistic Regression)

Logistic regression was fit using statsmodels on valid deployments only.

Reference category: specific location + human contact present

Predictor	Odds Ratio	Interpretation
Spatial anchor missing	6.32×	Preventable driver
Spatial anchor vague	2.32×	Preventable driver
Human contact absent	3.26×	Preventable driver
Execution constraint	4.78×	Intrinsic — unavoidable

Pseudo R²: 0.14

Appropriate for noisy operational systems where intrinsic variability is expected.

Interpretation:

Missing spatial information multiplies punchlisting odds by over 6×.

Lack of a human contact more than triples odds.

Intrinsic execution constraints significantly increase failure risk — confirming that some punchlisting is structurally unavoidable.

Counterfactual: Minimum Anchor Rule

A conservative counterfactual was computed:

If all valid deployment tickets satisfied a Minimum Anchor Rule (at least one anchor required), non-compliant tickets were assumed to perform at the current compliant-ticket punchlisting rate.

This assumption:

Does not eliminate intrinsic constraints

Does not assume perfect data

Does not assume behavioral change

Estimated impact:

~87 revisits avoided per 2,000 valid deployments (~13% reduction)

This represents downstream labor savings without redesigning the entire workflow.

The Core Distinction
Type	Cause	Fixable via Better Ticketing?
Intrinsic	Workstation in use, locked room, patient present	No
Preventable	Missing location, no contact, ambiguous department label	Yes

This distinction is encoded structurally in the simulation:

execution_constraint affects punchlisting probability independently of anchor quality (intrinsic noise).

Anchor and ambiguity variables layer on top of that baseline (preventable signal).

The goal is not to eliminate punchlisting — it is to quantify the portion that better ticketing can realistically fix.

There exists a minimum viable information threshold below which human judgment fails to scale.

Recommendation
Minimum Anchor Rule

Require deployment tickets to include at least one of:

A specific physical location

A designated human point of contact

Perfect information is unnecessary. However, when both anchors are missing, failure becomes structurally predictable.

Supporting recommendations:

Replace free-text department descriptions with structured location fields

When no anchor exists, coordinate with a department manager before dispatching a technician

Operational Context

This analysis models a real healthcare IT deployment workflow:

Ticketing system similar to FileMaker (~40% upstream data accuracy)

Field technicians deploying peripherals across large hospital departments

Leadership acknowledgment that punchlisting was operationally normal

The simulation preserves structural co-occurrences observed in practice:

Missing inventory data co-occurs with missing contact/location data

Ambiguous department labels degrade spatial specificity

Intrinsic constraints operate independently of information quality

This is not randomized data — it is logically structured.

Data Design

2,000 synthetic deployment tickets.

Variable	Description
task_applicability	valid_target / invalid_target (analysis focuses on valid)
spatial_anchor	specific / vague / missing
human_anchor	present / absent
dept_label_quality	ambiguous / unambiguous
execution_constraint	intrinsic blocker (~12% baseline)
punchlisted	binary outcome
revisit_required	probabilistic — ~80% of punchlisted generate revisit
revisit_successful	yes / no / resolved_admin
Methodology

Descriptive analysis
Punchlisting rates analyzed by anchor availability and department label quality.

Logistic regression (statsmodels)
Model fit on valid targets using:

spatial_anchor

human_anchor

dept_label_quality

execution_constraint

This is an explanatory model focused on effect size and interpretability, not predictive optimization.

Counterfactual simulation
Non-compliant tickets assumed to reach the compliant-ticket punchlisting rate under policy enforcement. This produces a conservative lower-bound estimate.

Reproducing the Analysis
pip install -r requirements.txt
python src/generate_data.py
python src/analyze.py

Outputs written to:

outputs/figures/

outputs/tables/

outputs/summary.json

Limitations

Simulated data cannot capture full behavioral variability (e.g., unresponsive contacts)

Findings are explanatory, not predictive

Counterfactual assumes anchor compliance is enforceable upstream

No temporal or technician-level variation modeled
