# Example Data Layout

HyPo-Net expects pre-windowed 60-second segments stored as HDF5 files.
One fold should look like:

```text
fold1/
  train/
    ART.h5                  # dataset key: ART, shape [N, 6000]
    ECG.h5                  # dataset key: ECG, shape [N, 6000]
    PLETH.h5                # dataset key: PLETH, shape [N, 6000]
    extracted_features.h5   # first dataset, shape [N, 81]
    HARD_LABELS.h5          # dataset key: HARD_LABELS, shape [N]
    LABELS_CONCAVE.h5       # dataset key: LABELS_CONCAVE, shape [N]
  val/
  test/
```

The repository does not include patient data. Replace these files with your
own preprocessed segments following the same layout.
