SEED = 42
N_ROWS = 2000

SPATIAL_LEVELS = ["specific", "vague", "missing"]
HUMAN_LEVELS = ["present", "absent"]
TASK_LEVELS = ["valid_target", "invalid_target"]

# Base rates
P_INVALID_TARGET = 0.15
P_HUMAN_PRESENT = 0.65
P_SPATIAL_SPECIFIC = 0.30
P_SPATIAL_VAGUE = 0.50
P_DEPT_AMBIGUOUS = 0.30
P_EXECUTION_CONSTRAINT = 0.12  # intrinsic: locked room, workstation in use, etc.

# Revisit probabilities
P_REVISIT_IF_PUNCHLISTED = 0.80      # not all punchlisted tickets generate a revisit
P_REVISIT_SUCCESS_ON_SECOND = 0.70   # of revisits, 70% succeed on second attempt