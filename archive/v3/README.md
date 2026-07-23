# v3 single nested-CV snapshot

v3 was completed on 2026-07-22 using 312 labeled patients (190 stable and 122
unstable) and v7 features from `features_312_20260722/`.

- Outer validation: one five-fold stratified split with seed 2026.
- Inner tuning: four-fold stratified CV inside each outer training fold.
- Formal comparison: ElasticNet, XGBoost, and LightGBM across 3D, 2D, and
  combined feature sets (nine combinations).
- Best apparent v3 result: LightGBM + 2D, ROC-AUC 0.5958.

Contents:

- `code/`: exact source, entry point, configuration, dependency, and README snapshot.
- `results/formal_v7_run_01/`: 26 formal result files.
- `results/validation_only/`: pre-run data and feature validation outputs.

v4 was introduced because the v3 result used only one random outer-fold split.
