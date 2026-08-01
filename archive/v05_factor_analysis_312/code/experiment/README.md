# Current v5 experiment package

- `build_features.py`: dynamic patient-ID cohort loading and deterministic v7
  2D/3D/cross-layer/multi-muscle patient-level feature construction.
- `feature_audit.py`: cohort, formula, range, duplicate-texture, and ratio audits.
- `preprocessing.py`: training-fold-only preprocessing and feature selection.
- `modeling.py`: model pipelines, nested CV, OOF prediction, metrics, and plots.
- `runner.py`: retained single-run orchestration used by the frozen v3 snapshot.
- `repeated_runner.py`: v4 repeated nested CV, per-seed checkpoints,
  hierarchical bootstrap, paired comparisons, and 50-split feature stability.
- `factor_threshold_audit.py`: feasibility audit for fixed Pearson thresholds.
- `factor_modeling.py`: training-fold-only Pearson screening, varimax factor
  analysis, nested model tuning, factor importance, and loading extraction.
- `factor_runner.py`: v5 repeated nested CV, checkpoints, hierarchical
  inference, selection stability, and clearly separated descriptive factors.
- `config.json`: locked tuning candidates and reproducibility settings.

The root `run_experiment.py` is the v5 entry point. It locks the Pearson
threshold at 0.15, tests 3/5/8/10 factors, compares ElasticNet, XGBoost, and
LightGBM, and uses ten random seeds. Current code does not import anything from
`archive/`.
