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

## Independent v6 segment pilot

- `segment_pilot.py`: aligns the 30-patient slice-level annotation table with
  v7 2D muscle features, constructs global/target/adjacent/mechanistic feature
  sets, and compares them on identical repeated nested-CV splits.
- Root `run_segment_pilot.py`: editable PyCharm entry point. It writes only to
  `results/v6_segment_pilot_test30` and does not change the v5 main pipeline.

## v7 compact 219-patient segment validation

- `segment_validation_219.py`: creates a protocol-inferred 219-patient
  annotation table from the verified patient-77 mapping, builds locked compact
  global/target/gradient panels, and performs a primary validation in the 189
  patients not used by the v6 pilot.
- Root `run_segment_validation_219.py`: editable PyCharm entry point. The
  all-219 analysis is saved as a secondary sensitivity analysis.

## v8 nested feature discovery

- `feature_discovery.py`: constructs a label-independent 2D/3D/segment candidate
  universe, performs training-fold correlation clustering and stability
  screening, compares nested best 10/15/20 subsets, tests PCA aggregation, and
  runs a reduced full-search label-permutation negative control.
- Root `run_feature_discovery.py`: editable PyCharm entry point. Outputs include
  hard exclusions, feature evidence, retain/exclude recommendations, all OOF
  predictions, and reproducibility metadata.
## v9 Pearson-to-six-factor replication

`run_pearson_factor_replication.py` audits the requested absolute Pearson
threshold of 0.25, then runs the previously approved 0.15 fallback with six
varimax factors.  It reports an intentionally optimistic full-cohort
screening/factor-construction workflow beside a leakage-safe repeated nested-CV
workflow.  Only `nested_train_only` is a valid generalization estimate.
