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

The static clinical array used by `extract_handcrafted_features.py` should be
indexed by `case_id - 1`, matching the original preprocessing convention.
