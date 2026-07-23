# Current v4 experiment package

- `build_features.py`: dynamic patient-ID cohort loading and deterministic v7
  2D/3D/cross-layer/multi-muscle patient-level feature construction.
- `feature_audit.py`: cohort, formula, range, duplicate-texture, and ratio audits.
- `preprocessing.py`: training-fold-only preprocessing and feature selection.
- `modeling.py`: model pipelines, nested CV, OOF prediction, metrics, and plots.
- `runner.py`: retained single-run orchestration used by the frozen v3 snapshot.
- `repeated_runner.py`: v4 repeated nested CV, per-seed checkpoints,
  hierarchical bootstrap, paired comparisons, and 50-split feature stability.
- `config.json`: locked tuning candidates and reproducibility settings.

The root `run_experiment.py` is the v4 entry point. It locks exactly three
candidate model/feature combinations and ten random seeds. Current code does not
import anything from `archive/`.
