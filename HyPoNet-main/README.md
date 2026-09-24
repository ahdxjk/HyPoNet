# HyPoNet Core Code

This repository implements the manuscript's 60-second, three-signal HyPoNet
method. It includes patient-level splitting, segment and target construction,
81 handcrafted features, the dual-stream model, and five-fold training.
Raw VitalDB recordings, clinical data, trained weights, and the supplementary
81-feature definitions are not included.

## Method encoded here

- Input: synchronized ABP/ART, lead-II ECG, and PPG/PLETH at 100 Hz, giving
  a `3 x 6000` waveform tensor per 60-second segment. The model also receives
  81 handcrafted descriptors derived from these signals and age, sex, height,
  weight, and BMI.
- A POH episode requires pulse-derived MAP below 65 mmHg for **more than**
  60 seconds. An input is hard-positive if it contains an ongoing episode or
  its end precedes an onset by at most `H` minutes, where `H` is 5, 10, or 15.
  A non-ongoing input is excluded when its complete future `H`-minute horizon
  was not observed, even if an onset is visible before recording end.
- The soft target is a training annotation, not a calibrated event probability:

  ```text
  x_time  = 1 (ongoing), clip(1 - minutes_to_onset/H) (future onset), or 0 (event-free)
  x_MAP   = clip((105 - pulse_based_MAP)/40)
  r       = clip(5_s_smoothed_ABP_OLS_slope_in_mmHg_per_min/30, -1, 1)
  x_trend = (1 - r)/2
  g(x)    = 2x - x²
  P       = 0.5 g(x_MAP) + 0.4 g(x_time) + 0.1 g(x_trend)
  ```

  `clip` bounds values to `[0, 1]` except for `r`, whose bounds are shown.
  For MAP 85 mmHg and a flat trend, inputs ending four and one minute before
  onset at `H=5` have targets `0.594` and `0.834`, respectively.
- The DWT-Former applies level-three DWT with db4 for ABP/PPG and db2 for ECG,
  embeds frequency-subband patches, uses a Transformer, and learns channel
  attention pooling. Patch-Conv uses 64-sample patches, 64-dimensional ABP
  queries over all ECG and PPG patches, followed by residual 1D convolutions
  and global pooling. Each stream and the handcrafted MLP outputs 128 values;
  their 384-value concatenation enters a two-layer fully connected head.
- Soft supervision minimizes **only** mean Bernoulli KL from `P` to the
  sigmoid score. The optional matched hard-supervision comparator uses BCE on
  the binary endpoint, with the same architecture and training settings.
- The default optimizer is Adam with learning rate `2e-5`, weight decay
  `1e-5`, cosine annealing, dropout `0.2`, and batch size `32`. The five
  patient-level runs use four folds for training and split the held-out fold
  into validation and test subsets. Handcrafted-feature imputation and
  standardization are fitted on the training subset only for each run.

## Installation

Install PyTorch appropriate for your machine, then run:

```bash
pip install -r requirements.txt
```

## Reproduction workflow

Patient assignment must happen before segment generation. If your clinical
table has a patient identifier shared by multiple recordings, pass that column
with `--patient-id-column`. The default `caseid` is appropriate only when
each patient has one case. The clinical table needs `caseid` and `age` columns.
Supply a clinical table already restricted to the manuscript's 3,309-person
candidate cohort; the provided manuscript does not state enough selection
criteria to recover that cohort automatically from all VitalDB cases.

```bash
python scripts/assign_patients.py --clinical-csv data/clinical_data.csv --vital-dir data/raw_vital_files --output-dir data/patient_folds --patient-id-column patient_id --expected-patients 3309
```

This writes fixed `fold1_train_cases.txt` through `fold5_test_cases.txt` and
`all_cases.txt`. Use the same manifests for every horizon and supervision
comparison. For a clinical table with one case per patient and no patient ID
column, omit `--patient-id-column`. The code stratifies assignment by
ten-year age groups; the manuscript specifies patient disjointness and split
proportions but does not specify a stratification algorithm.

Generate per-case segments and `LABELS_CONCAVE` targets:

```bash
python scripts/make_segments.py --input-dir data/raw_vital_files --case-list data/patient_folds/all_cases.txt --output-dir data/segments_5min --horizon-minutes 5 --n-jobs 8
python scripts/make_segments.py --input-dir data/raw_vital_files --case-list data/patient_folds/all_cases.txt --output-dir data/segments_10min --horizon-minutes 10 --n-jobs 8
python scripts/make_segments.py --input-dir data/raw_vital_files --case-list data/patient_folds/all_cases.txt --output-dir data/segments_15min --horizon-minutes 15 --n-jobs 8
```

The default keeps all eligible segments. `--balance-hard-labels` opts into the
old per-case negative downsampling strategy; the manuscript does not specify
the exact negative sampling rule. The pulse-finding details used to detect
episodes are also an implementation choice because the manuscript specifies
the MAP threshold and duration but not the detector algorithm.

Build HDF5 splits from the **existing** patient manifests:

```bash
python scripts/build_folds.py --segments-dir data/segments_5min --splits-dir data/patient_folds --output-root data/5min
```

Repeat for the other horizons, changing the segment and output directories.
Next, extract features. The external `clinical_selected.npy` array must be
indexed by `caseid - 1`; its first four columns must be age, sex code
(`0=male`, `1=female`), height in cm, and weight in kg. BMI is recomputed
from height and weight. The training split supplies each fold's imputation
means and standardization parameters.

```bash
python scripts/extract_handcrafted_features.py --data-root data/5min --static-data data/clinical_selected.npy
```

Train the manuscript model. Run the hard comparator with the same data and
seed in a separate output directory when reproducing the matched experiment:

```bash
python scripts/train_cv.py --data-root data/5min --output-dir outputs/5min_soft --dwt-level 3 --supervision soft --seed 42
python scripts/train_cv.py --data-root data/5min --output-dir outputs/5min_hard --dwt-level 3 --supervision hard --seed 42
```

The training script reports test AUROC, recall and F1 at score cutoff `0.5`
against independently constructed hard labels, and KL against soft targets.
Its `summary.csv` uses the sample standard deviation over five runs.

## Chronological five-minute warning evaluation

Feature extraction also saves `feature_preprocessor.npz` in each fold directory
so raw chronological windows use that fold's training-only feature statistics.
For each trained model and fold, score the validation and test recordings:

```bash
python scripts/score_monitoring.py --input-dir data/raw_vital_files --splits-dir data/patient_folds --fold 1 --fold-dir data/5min/fold1 --checkpoint outputs/5min_soft/fold1/best_model.pt --static-data data/clinical_selected.npy --clinical-csv data/clinical_data.csv --patient-id-column patient_id --output-dir outputs/monitoring_soft/fold1
python scripts/evaluate_monitoring.py --decisions outputs/monitoring_soft/fold1/decisions.csv --events outputs/monitoring_soft/fold1/events.csv --output outputs/monitoring_soft/fold1/warnings.csv --model soft --run-id 1
```

Repeat for the other folds and the hard-supervised checkpoints. Scoring uses
the preceding 60-second input every 20 seconds. The warning analysis excludes
ongoing POH and incomplete observations, selects each threshold on validation
at 0.25, 0.5, 1, and 2 false alarms per hour, then applies it unchanged on
test. It uses threshold crossings, a 300-second refractory interval, and
counts warnings 60–300 seconds before onset as timely. The output includes
event sensitivity, test false-alarm rate, lead time, and pre-onset coverage.
The exact definitions for ambiguous alarm boundary cases are documented in
`hyponet/monitoring.py`.

When there is one recording per patient, omit `--clinical-csv` and
`--patient-id-column`; `case_id` is then used as `patient_id`. The scorer writes
this ID into both CSVs so matched comparisons can resample patients rather
than individual 20-second decisions.

Compare hard and soft supervision on matched test decisions. Each model's
threshold is selected on its own validation scores; the resulting test
differences use 2,000 paired patient-cluster bootstrap resamples:

```bash
python scripts/compare_monitoring.py --hard-decisions outputs/monitoring_hard/fold1/decisions.csv --soft-decisions outputs/monitoring_soft/fold1/decisions.csv --events outputs/monitoring_hard/fold1/events.csv --soft-events outputs/monitoring_soft/fold1/events.csv --output-dir outputs/monitoring_comparison/fold1
```

The paper's MAP-only, trend-only, and MAP-plus-trend logistic references can be
fitted on the same training split and evaluated on the same chronological
recordings. The optional `--matched-decisions` checks that their decision grid
and eligibility flags match HyPoNet's:

```bash
python scripts/run_simple_references.py --input-dir data/raw_vital_files --splits-dir data/patient_folds --fold-dir data/5min/fold1 --fold 1 --clinical-csv data/clinical_data.csv --patient-id-column patient_id --matched-decisions outputs/monitoring_soft/fold1/decisions.csv --output-dir outputs/simple_references/fold1
```

The manuscript does not specify the logistic optimizer or penalty. This
implementation records its fitted coefficients and uses an unpenalized
Bernoulli objective by default (`--l2` changes the penalty).

## Matched score-dynamics analysis

The manuscript also compares hard and soft score variability on the same
chronological test decisions. Add an externally adjudicated Boolean `stable`
column to **both** model decision CSVs; the paper does not give a rule from
which to infer stable physiological periods. The `stable` marks, patient IDs,
decision times, and valid-decision flags must agree across the two models.
Then run, for example:

```bash
python scripts/evaluate_score_dynamics.py --hard-scores outputs/monitoring_hard/fold1/decisions_with_stable.csv --soft-scores outputs/monitoring_soft/fold1/decisions_with_stable.csv --events outputs/monitoring_soft/fold1/events.csv --output-dir outputs/score_dynamics/fold1
```

The evaluator averages the SD of ten consecutive 20-second predictions within
each stable case, assesses pre-onset rise and time-score Spearman across all
eligible events including missed ones, fits percentile and z-score transforms
using validation scores only, and uses 2,000 paired patient-cluster bootstrap
resamples. Rise uses the last minus the first available pre-onset score in the
specified window, an explicit choice because the paper does not define exact
rise endpoints. The output records excluded events and the analysis settings.

After these changes, regenerate segments, HDF5 splits, and handcrafted
features. Existing model checkpoints are incompatible with the corrected
architecture. The published numerical results cannot be checked without the
source records, the exact cohort/partition information, and Supplementary Note
S13's complete definitions of the 81 handcrafted descriptors. The current
feature extractor has 81 dimensions, but exact descriptor equivalence cannot
be claimed from the provided manuscript alone.
