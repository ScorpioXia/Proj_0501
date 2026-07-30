# Current results

- `v4_repeated_nested_cv/`: completed v4 10-seed repeated nested-CV experiment.
- `v6_segment_pilot_test30/`: completed 30-patient segment-localisation
  feasibility experiment using v7 slice-level 2D features.
- `v7_segment_validation_219/`: compact segment validation with a primary
  189-patient confirmation cohort and secondary all-219 sensitivity analysis.
- `v8_nested_feature_discovery/`: exploratory nested 10/15/20-feature subset
  discovery with stability evidence and negative controls.
- `pre_refactor_v7_20260722/`: pre-refactor v7 historical output; not a current
  formal result because it used the earlier pipeline.

The v3 single-seed formal run and validation-only output are frozen under
`archive/v3/results/`.
## v9_pearson_factor_replication

Reproduction and leakage audit of Pearson screening followed by six-factor
analysis and machine learning.  The folder separates invalid
`optimistic_global` pseudo-OOF estimates from valid `nested_train_only`
predictions.
## v7_stability_lasso_replication

Repeated outer validation of the near-zero variance -> redundancy ->
univariate logistic -> Stability LASSO -> at most seven variables workflow.
Only `nested_train_only` is valid predictive evidence.
