# Data-leakage analysis

Leakage means information from evaluation subjects influencing anything the model is trained on
or selected with. The table goes through every step that touches data. *Scope* says where its
statistics come from.

| Step | Scope of statistics | Leakage risk | Notes |
|---|---|---|---|
| DICOM sorting, conversion | none (rules on metadata) | none | Rules were fixed before modelling. |
| Target grid 1 × 1 × 5 mm | fixed a priori | none | Not derived from cohort medians; the spacing audit is descriptive only. |
| Registration, resampling | within subject | none | |
| Masks (Li / Otsu thresholds) | per image | none | |
| N4 bias correction | per image | none | |
| Z-score (T1c, T1, T2, B0, DWI) | per image | none | Instance normalisation. |
| ADC unit harmonisation | fixed physical rule | none | Order-of-magnitude rule decided in advance. |
| Crop / pad to 320², ROI boxes | per subject (fiducial) | none | See *reader bias* below. |
| **ADC scaling to [0, 1]** | **cohort** | **yes, if fitted on all subjects** | Fitted on the training subjects of each fold only: `FoldwiseADCScaler`. |
| CV splitting | cohort | **yes, if not patient-grouped** | Duplicate archives (`<ID>_2`) must share a fold: `assert_group_disjoint(groups=...)`. |
| Model / hyper-parameter selection (e.g. box size 48–90 mm, which sequences) | cohort | **yes, if selected and reported on the same folds** | Use nested CV or a held-out test set that is touched once. |

## The ADC scaler contract

```python
from hn_mri_prep.normalization.adc import fit_fold_scalers

scalers = fit_fold_scalers(adc_by_subject, folds, groups=subject_to_patient, method="percentile")
for k, (train_ids, val_ids) in enumerate(folds):
    scalers[k].save(f"fold{k}_adc_scaler.json")                  # audit trail per fold
    x_train = [scalers[k].transform(adc_by_subject[s], s, evaluation=False) for s in train_ids]
    x_val = [scalers[k].transform(adc_by_subject[s], s) for s in val_ids]   # raises if s was in train
```

For a final model evaluated on a held-out test set, fit one scaler on the entire development
set and apply it to the test set.

## Not leakage, but bias to report (STARD 2015 / CLAIM 2024)

- **Reader bias in fiducial placement.** Who placed the tumour fiducials, how experienced they
  were, and whether they were blinded to histopathology (STARD items 13a/13b). The fiducial
  only centres the box, but a reviewer will ask.
- **Preprocessing failures.** Subjects excluded because registration failed or a sequence was
  missing belong in the participant flow diagram (STARD item 19). `preprocess_audit.csv` and
  `roi_manifest.csv` give the counts and reasons.
- **Reference standard.** Histopathology after surgery or biopsy, and the interval between MRI and
  the reference standard (STARD items 11, 14).
