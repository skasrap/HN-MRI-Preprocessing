# Data leakage

Everything in stages 1–4 is computed per image or per subject (registration, masks, N4,
z-scores, crops) or fixed in advance (target grid, ADC unit rule). None of it uses statistics
from other subjects.

The places where cohort information can leak in are all at training time:

| Step | Risk | What to do |
|---|---|---|
| ADC scaling to [0, 1] | range estimated on all subjects | fit per fold on the training subjects (`FoldwiseADCScaler`) |
| CV splits | the same patient in two folds | group by patient; `sort_manifest.csv` flags repeated PatientIDs (`<ID>_2`) |
| Model choices (box size, sequences, hyper-parameters) | chosen and reported on the same folds | nested CV, or a test set used once |

```python
from hn_mri_prep.normalization.adc import fit_fold_scalers

scalers = fit_fold_scalers(adc_by_subject, folds, groups=subject_to_patient, method="percentile")
for k, (train_ids, val_ids) in enumerate(folds):
    scalers[k].save(f"fold{k}_adc_scaler.json")
    x_train = [scalers[k].transform(adc_by_subject[s], s, evaluation=False) for s in train_ids]
    x_val = [scalers[k].transform(adc_by_subject[s], s) for s in val_ids]  # raises if s was in train
```

For the final model, fit one scaler on the whole development set and apply it to the test set.
