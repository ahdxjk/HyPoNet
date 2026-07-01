# HyPo-Net Core Code

This folder contains the core code for the HyPo-Net probability-fitting model
for perioperative hypotension prediction. It is intentionally smaller than the
research workspace: baseline models, reviewer-response experiments, plotting
scripts, checkpoints, logs, and private data are not included.

## What Is Included

- `hyponet/models/hyponet.py`: the main HyPo-Net model.
- `hyponet/models/xresnet1d.py`: temporal Patch-Conv/xResNet backbone.
- `hyponet/data.py`: HDF5 dataset loader for 60-second waveform windows.
- `hyponet/losses.py`: soft probability-fitting loss.
- `hyponet/metrics.py`: AUROC, F1, KL, soft-target MSE, Brier score, and ECE.
- `hyponet/labels.py`: concave mapping and probability-component weighting utilities.
- `scripts/train_cv.py`: five-fold training entry point.

## Model Inputs

The model uses:

- ART waveform: 60 seconds at 100 Hz, shape `[6000]`.
- ECG waveform: 60 seconds at 100 Hz, shape `[6000]`.
- PLETH waveform: 60 seconds at 100 Hz, shape `[6000]`.
- Handcrafted physiological features: shape `[81]`.

Internally, HyPo-Net combines a temporal Patch-Conv stream, a DWT-Former
frequency stream, and a handcrafted-feature MLP. The default wavelet packet
decomposition level is `L=3`.

## Data Layout

Prepare each prediction horizon as five patient-level folds:

```text
data/5min/
  fold1/
    train/
    val/
    test/
  fold2/
  ...
  fold5/
```

Each split directory should contain:

```text
ART.h5                  key: ART
ECG.h5                  key: ECG
PLETH.h5                key: PLETH
extracted_features.h5   first dataset, shape [N, 81]
HARD_LABELS.h5          key: HARD_LABELS
LABELS_CONCAVE.h5       key: LABELS_CONCAVE
```

Patient-level splitting should be performed before training. Segments from the
same patient/case must not appear in more than one of train, validation, and
test sets.

## Installation

```bash
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version if GPU acceleration is
needed.

## Training

Run five-fold training for a 5-minute horizon:

```bash
python scripts/train_cv.py --data-root data/5min --output-dir outputs/5min
```

For other horizons, change `--data-root`:

```bash
python scripts/train_cv.py --data-root data/10min --output-dir outputs/10min
python scripts/train_cv.py --data-root data/15min --output-dir outputs/15min
```

Useful options:

```bash
python scripts/train_cv.py \
  --data-root data/5min \
  --output-dir outputs/5min \
  --wp-level 3 \
  --epochs 100 \
  --batch-size 256 \
  --alpha 1.0
```

Outputs:

- `fold_results.csv`: metrics for each fold.
- `summary.csv`: mean and standard deviation across folds.
- `fold*/best_model.pt`: checkpoint for each fold.

## Soft Probability Labels

The default probability target is the concave label `LABELS_CONCAVE`.
For documentation and reproducibility, `hyponet/labels.py` provides:

```python
from hyponet.labels import concave_mapping, combine_probability_components
```

The default component weighting follows the manuscript order
MAP:time:trend = 5:4:1.

## Notes Before Public Release

- Add a `LICENSE` file before publishing.
- Add citation information after the manuscript metadata is finalized.
- Do not commit patient data, checkpoints trained on private data, or local logs.
