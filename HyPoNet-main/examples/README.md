# Expected Data Layout

After preprocessing, each horizon should look like:

```text
data/5min/
  fold1/
    train/
      ART.h5
      ECG.h5
      PLETH.h5
      STATIC.h5
      HARD_LABELS.h5
      LABELS_CONCAVE.h5
      extracted_features.h5
    val/
    test/
  ...
  fold5/
```

Create the patient manifests in `data/patient_folds/` before generating these
segments. Reuse them across the 5-, 10-, and 15-minute datasets.

The static clinical array used by `extract_handcrafted_features.py` should be
indexed by `case_id - 1`. Its first four columns are age, sex code (0=male,
1=female), height in cm, and weight in kg; BMI is derived from height and
weight by the extractor.
