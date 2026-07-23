# v4 repeated nested-CV snapshot

v4 was completed on 2026-07-23 using 312 labeled patients (190 stable and 122
unstable) and the same v7 feature tables used in v3.

Locked candidate combinations:

- LightGBM + 2D (`E2_2d`)
- XGBoost + combined 2D/3D (`E3_combined`)
- ElasticNet + 3D (`E1_3d_level3`)

Validation design:

- Ten prespecified repeat seeds: 2026 through 2035.
- Five outer stratified folds per repeat and four inner folds for tuning.
- Fifty independent outer test predictions per candidate in total.
- The primary estimate is the mean of the ten repeat-level pooled out-of-fold
  metrics, with hierarchical patient-and-repeat bootstrap confidence intervals.

Contents:

- `code/`: exact source, entry point, configuration, dependency, and README snapshot.
- `results/v4_repeated_nested_cv/`: all final tables, logs, audit files, and ten
  completed per-seed checkpoints.

The archived primary performance table has SHA-256
`C8966D7728E1EF800F6FD7BB917B9544B46538B9E6A78F395444C3891DD0B94E`,
identical to the live v4 result at archive creation time.
