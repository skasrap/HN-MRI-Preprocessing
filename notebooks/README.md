# Notebooks

Exploratory notebooks only. The pipeline itself lives in `src/hn_mri_prep` and runs from the CLI.

Rules:

1. **Strip outputs before committing.** Outputs can contain patient identifiers, file paths and
   images. `pre-commit install` adds `nbstripout`, which does this automatically.
2. Refer to subjects by pseudonym (`hn-mri-prep sort-dicom --pseudonymize PG ...`), never by
   hospital ID.
3. Read data paths from an environment variable or a local, git-ignored config. Do not hard-code them.
