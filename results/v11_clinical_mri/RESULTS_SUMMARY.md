# v11 clinical plus MRI experiment

All performance values are based on repeated out-of-fold predictions. The result snapshot was originally run under the internal name `v8_clinical_mri` and was promoted to canonical project version v11 during repository organization.

## Same-cohort model comparison

| Cohort | Clinical AUC (95% CI) | MRI AUC (95% CI) | Combined AUC (95% CI) | Combined minus MRI (95% CI) |
|---|---:|---:|---:|---:|
| complete4_n95 | 0.539 (0.415–0.663) | 0.621 (0.498–0.739) | 0.627 (0.504–0.744) | +0.006 (-0.043 to +0.054) |
| age_bmi_n130 | 0.442 (0.336–0.549) | 0.650 (0.550–0.744) | 0.641 (0.543–0.734) | -0.010 (-0.023 to +0.003) |
| all_native_missing_n312 | 0.494 (0.428–0.562) | 0.552 (0.486–0.615) | 0.537 (0.471–0.603) | -0.015 (-0.037 to +0.006) |
| locked7_complete4_overlap_n68 | **0.432 (0.287–0.584)** | **0.686 (0.539–0.823)** | **0.708 (0.564–0.833)** | +0.023 (-0.058 to +0.104) |

The 0.708 result is present in `aggregate_performance.csv` for `locked7_complete4_overlap_n68 / combined_locked7_clinical`. It is non-independent because the MRI variables were selected using labels from the overlapping 219-patient cohort.

## Additional locked-panel reproducibility result

On all 219 patients available to the locked seven-feature panel, L2 Logistic achieved AUC 0.665 (95% CI 0.588–0.739). This is a same-cohort reproducibility estimate, not external validation.

## Interpretation constraints

- Values 0 and spreadsheet error strings in clinical fields were converted to missing.
- No clinical mean/median imputation was used. The 312-patient secondary analysis uses XGBoost native missing-value routing and explicit missingness indicators.
- The global mechanistic MRI seven-variable panel was prespecified without current-label screening.
- The locked seven-variable panel was selected using the same 219 labels before this experiment; neither its 0.665 nor the n=68 combined 0.708 is independent validation.
- Slip segment was omitted from the 312-patient analysis.
