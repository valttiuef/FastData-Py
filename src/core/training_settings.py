# --- @ai START ---
# model: gpt-5
# tool: codex
# role: architectural-refactor
# reviewed: yes VT20260303
# date: 2026-03-03
# --- @ai END ---
"""Global settings for model-training data filtering rules."""

# Drop feature columns when more than this ratio is missing (NaN).
TRAINING_SPARSE_FEATURE_NAN_RATIO_THRESHOLD = 0.95

# Drop feature columns when non-null values contain this many or fewer unique values.
TRAINING_STATIC_FEATURE_MAX_UNIQUE_NON_NULL = 1

# If enabled, try merging under-sized stratification groups before falling back from
# Stratified K-Fold to regular K-Fold.
TRAINING_STRATIFIED_KFOLD_MERGE_SMALL_GROUPS = 1
