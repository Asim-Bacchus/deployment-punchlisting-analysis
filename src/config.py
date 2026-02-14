# src/config.py
SEED = 42
N_ROWS = 2000

SPATIAL_LEVELS = ["specific", "vague", "missing"]
HUMAN_LEVELS = ["present", "absent"]
TASK_LEVELS = ["valid_target", "invalid_target"]

# Base rates (tune later)
P_INVALID_TARGET = 0.20
P_HUMAN_PRESENT = 0.60
P_SPATIAL_SPECIFIC = 0.40
P_SPATIAL_VAGUE = 0.40
P_DEPT_AMBIGUOUS = 0.30
