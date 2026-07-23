# v5 Pearson-factor experiment snapshot

v5 was completed on 2026-07-23 using 312 labeled patients and all 601 eligible
v7 combined 2D/3D patient-level features.

- Absolute Pearson threshold: 0.15, fitted inside every training partition.
- Reduction: standardized varimax factor analysis.
- Candidate factor counts: 3, 5, 8, and 10, chosen by inner CV.
- Models: ElasticNet, XGBoost, and LightGBM.
- Validation: ten seeds, five outer folds, and four inner folds.

The original 0.25 threshold was rejected after a documented feasibility audit:
48/50 outer training folds retained zero features. At 0.15, all outer and inner
training partitions retained usable features.

Contents:

- `code/`: exact v5 source, entry point, configuration, dependencies, and README.
- `results/v5_factor_analysis/`: final tables, plots, audits, logs, explanatory
  outputs, and ten completed per-seed checkpoints.

The archived primary performance table has SHA-256
`CBDD8A82C91EC4072D89423776E3F546202C178EB24E41E8A9FB2DC0040008A0`,
identical to the live result at archive creation time.
