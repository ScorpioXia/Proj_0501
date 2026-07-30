# v6 Pearson-to-six-factor replication snapshot

v6 freezes the experiment completed on 2026-07-30 using 219 labeled patients
and 631 eligible v7 2D/3D/segment-derived features.

- Exact requested absolute Pearson threshold: 0.25.
- Feasibility result: zero full-cohort features reached 0.25.
- Executable fallback threshold: 0.15.
- Reduction: six varimax-rotated factors.
- Models: L2 Logistic, RBF-SVM, and XGBoost.
- Validation: ten seeds, five outer folds, and four inner folds.
- Analysis-order control: invalid full-cohort supervised screening versus valid
  training-fold-only screening and factor construction.

Contents:

- `code/`: exact source snapshot, entry point, dependencies, and README.
- `results/v6_pearson_factor_replication/`: predictions, performance,
  threshold audits, factor loadings, feature stability, logs, and Chinese
  interpretation.

The archived aggregate performance table has SHA-256
`732CBEB2E044BE6E64E4E7A082BDCB944F38622A02C0F0D1CD7F584F27592BEE`,
identical to the live result at archive creation time.

