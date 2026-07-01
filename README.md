# HyPo-Net Core Code

This folder contains the core code needed to reproduce the main method in the
paper: waveform segmentation, probability-label construction, patient-level
fold generation, 81-dimensional handcrafted feature extraction, and HyPo-Net
training.

It intentionally excludes reviewer-response experiments, plotting scripts,
checkpoints, private data, logs, and unrelated baseline experiments.

## Included Components

- `hyponet/preprocessing/segmentation.py`
  - Reads raw VitalDB `.vital` files.
  - Extracts ART, ECG II, and PLETH at 100 Hz.
  - Detects hypotension events using 60-second ART windows and MAP < 65 mmHg.
  - Generates 60-second waveform segments.
  - Assigns hard labels and soft probability labels.

- `hyponet/preprocessing/soft_labels.py`
  - Implements the selected concave probability mapping.
  - Combines MAP, time-to-event, and trend probabilities using MAP:time:trend
    = 5:4:1, i.e. 0.5, 0.4, and 0.1.

- `hyponet/preprocessing/folds.py`
  - Builds patient-level train/validation/test folds.
  - Keeps segments from the same case within one split.
  - Saves each split as HDF5 files.

- `hyponet/preprocessing/handcrafted_features.py`
  - Extracts the 81-dimensional ML feature vector used by the model.
  - Uses ABP/ART, ECG, PLETH/PPG, and static patient information.
  - Saves features as `extracted_features.h5`.

- `hyponet/models/hyponet.py`
  - Main HyPo-Net architecture.
  - Uses temporal Patch-Conv, DWT-based frequency representation, and
    handcrafted features.
  - The final fusion dimension is 384.

- `hyponet/train.py`
  - Probability-fitting training loop using KL loss against soft labels.

## Installation

Install PyTorch for your CUDA version, then:

```bash
pip install -r requirements.txt
```

`vitaldb` is only required if you regenerate segments from raw `.vital` files.

## Reproduction Workflow

### 1. Generate 60-second segments and probability labels

For 5-minute prediction:

```bash
python scripts/make_segments.py \
  --input-dir data/raw_vital_files \
  --output-dir data/segments_5min \
  --horizon-minutes 5 \
  --n-jobs 8
```

For 10- and 15-minute horizons:

```bash
python scripts/make_segments.py --input-dir data/raw_vital_files --output-dir data/segments_10min --horizon-minutes 10 --n-jobs 8
python scripts/make_segments.py --input-dir data/raw_vital_files --output-dir data/segments_15min --horizon-minutes 15 --n-jobs 8
```

The saved files are per-case NumPy arrays such as:

```text
0001_ART.npy
0001_ECG.npy
0001_PLETH.npy
0001_HARD_LABELS.npy
0001_LABELS_CONCAVE.npy
0001_STATIC.npy
```

By default, the interval scan follows the original scripts. It stores fixed
60-second chunks and uses the horizon-specific settings from the manuscript
code. If you want a conventional sliding event scan, add
`--sliding-interval-scan`.

### 2. Build patient-level five-fold HDF5 datasets

```bash
python scripts/build_folds.py \
  --segments-dir data/segments_5min \
  --clinical-csv data/clinical_data.csv \
  --output-root data/5min
```

Repeat for `data/segments_10min` and `data/segments_15min`.

### 3. Extract the 81-dimensional handcrafted features

```bash
python scripts/extract_handcrafted_features.py \
  --data-root data/5min \
  --static-data data/clinical_selected.npy
```

This creates `extracted_features.h5` in every fold split.

### 4. Train HyPo-Net

```bash
python scripts/train_cv.py \
  --data-root data/5min \
  --output-dir outputs/5min \
  --wp-level 3 \
  --alpha 1.0
```

Change `--data-root` to `data/10min` or `data/15min` for other horizons.

## Label Definitions

Each 60-second segment has:

- `HARD_LABELS`: binary label. A segment is positive if it is a hypotension
  event segment or lies within the prediction horizon before a hypotension
  event.
- `LABELS_CONCAVE`: selected soft probability target.

The selected soft label is:

```text
P = 0.5 * P_MAP + 0.4 * P_time + 0.1 * P_trend
```

where the order corresponds to MAP:time:trend = 5:4:1.

## Notes Before Release

- Add a `LICENSE` file before publishing.
- Do not commit patient data, `.vital` files, `.npy`/`.h5` data, checkpoints,
  TensorBoard logs, or reviewer-only experiment scripts.
- If manuscript metadata is finalized, add a citation section here.
