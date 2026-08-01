# Lumbar Stability Prediction from Paraspinal Muscle MRI

The current mainline is version 4 (v4). It predicts lumbar stability from v7
MRI-derived muscle features and uses only `patient_id` and binary `label` from
`PATIENT_LIST_FILE.csv`. Patient names and pinyin are never used as identifiers.

## Current layout

```text
Proj_0501/
|- run_experiment.py              # v4 PyCharm entry point
|- experiment/                    # current v4 pipeline
|- features_312_20260722/         # current v7 feature CSV files
|- PATIENT_LIST_FILE.csv          # cohort and labels
|- results/v4_repeated_nested_cv/ # completed v4 results
|- archive/v3/                    # frozen v3 code and results
|- archive/v1/, archive/v2/       # older versions
|- requirements.txt
`- feature-name reference workbook
```

## v4 protocol

The following model/feature combinations were locked before the v4 run:

- LightGBM + 2D features
- XGBoost + combined 2D/3D features
- ElasticNet + 3D/level-3 features

The protocol uses 10 random seeds (`2026` through `2035`). Each seed runs a
five-fold outer nested CV, with four-fold inner tuning. Each candidate therefore
has 50 outer validation splits and 3,120 repeated OOF predictions covering the
same 312 patients ten times.

The primary result is the mean of the ten repeat-level OOF metrics. Confidence
intervals use a hierarchical bootstrap that resamples both patients and repeat
seeds. Mean OOF predictions across repeats are reported separately as a
secondary cross-fitted ensemble analysis.

## Run in PyCharm

Use this interpreter:

```text
C:\ProgramData\anaconda3\envs\nnUNet-master\python.exe
```

Run `run_experiment.py` directly. The output directory is
`results/v4_repeated_nested_cv`. `RESUME=True` loads completed seed checkpoints,
so an interrupted run continues without refitting completed repeats.

## Methodological safeguards

- The labeled cohort size is dynamic and comes from `PATIENT_LIST_FILE.csv`.
- All patient joins use normalized `patient_id` only.
- Imputation, clipping, variance filtering, Spearman redundancy removal,
  point-biserial top-k selection, scaling, and tuning occur inside training folds.
- `patient_id`, `csf_value`, pixel spacing, slice thickness, threshold/QC fields,
  and absolute peak-slice indices are excluded from predictors.
- Corrected v7 GLCM, `SA_V`, and `3D_Shape_Index` fields are eligible.
- Still-duplicated GLRLM/GLSZM mappings remain excluded.
- Denominator-sensitive multi-muscle ratios use log or bounded transforms.

See `results/v4_repeated_nested_cv/RESULTS_SUMMARY.md` for the completed v4
results and `archive/v3/README.md` for the frozen v3 protocol.
