# HyPoNet

PyTorch implementation of **HyPoNet**, a multimodal framework for perioperative hypotension (POH) risk prediction.

Each model input contains a 60-second, 100 Hz window of arterial blood pressure (ABP), lead-II ECG, and photoplethysmography (PPG), represented as a `3 × 6000` waveform tensor, together with 81 handcrafted physiological features. The framework is evaluated for 5-, 10-, and 15-minute prediction horizons.

## Reproducibility and evaluation design

This repository is organized to mirror the evaluation pipeline described in the manuscript, from patient-level partitioning and segment generation to model training and chronological warning analysis.

* **Patient-level data separation.** Patient assignment is performed before segment generation. All recordings and derived 60-second windows from the same patient remain within the same training, validation, or test subset within each evaluation run.

* **Five-fold patient-level evaluation.** Patients are assigned to five non-overlapping folds. In each evaluation run, four folds are used for training, while the remaining fold is divided into validation and test subsets. The held-out fold is rotated across the five evaluation runs.

* **Fixed partitions across comparisons.** The same patient manifests are reused for matched supervision and model comparisons so that observed performance differences are not caused by different patient allocations.

* **No future information at inference.** Future POH onset information is used only retrospectively for target construction and outcome evaluation. At inference time, the model receives only the observed physiological signals and the corresponding available physiological and static features.

* **Soft targets are supervision signals, not calibrated probabilities.** The proposed risk-informed target combines temporal proximity to POH onset, current MAP severity, and short-term MAP trend to provide graded supervision during training. Its numerical value should not be interpreted as a calibrated probability of POH occurrence.

* **Matched hard-versus-soft comparison.** Hard- and soft-supervised HyPoNet models use the same architecture, patient partitions, waveform inputs, feature preprocessing, initialization protocol, and optimization settings. The supervision target is the principal experimental difference.

* **Separate segment-level and chronological evaluations.** Segment-level experiments assess horizon-specific discrimination and include both ongoing POH and inputs preceding POH onset within the corresponding prediction window. The chronological monitoring analysis excludes ongoing POH and separately evaluates strictly pre-onset warning behavior.

* **Validation-only alarm threshold selection.** Alarm thresholds are selected using validation recordings and are then applied unchanged to the corresponding test recordings.

* **Common evaluation grid for simple references.** MAP-only, trend-only, and MAP-plus-trend reference models are evaluated on the same eligible chronological decisions used for HyPoNet, allowing comparisons under a common warning protocol.

These design choices make patient separation, supervision comparisons, and chronological warning evaluation explicit and auditable.

## Evaluation outputs

The repository supports several complementary analyses:

* segment-level AUROC, recall, and F1-score;
* soft-target fitting using Bernoulli KL divergence;
* stable-period rolling score variation;
* pre-onset score rise and time-score correlation;
* event-level warning sensitivity;
* false alarms per evaluable hour;
* warning lead time;
* pre-onset warning coverage;
* paired patient-cluster bootstrap comparisons between hard and soft supervision.

The segment-level and chronological evaluations answer different questions and are therefore implemented as separate analysis pipelines.

## Run order

1. `assign_patients.py`
   Create fixed patient-level train/validation/test manifests before segment generation.

2. `make_segments.py`
   Extract 60-second waveform windows and construct hard and risk-informed soft labels.

3. `build_folds.py`
   Assemble eligible windows into HDF5 training, validation, and test datasets.

4. `extract_handcrafted_features.py`
   Compute the 81 handcrafted features and fit preprocessing statistics using the training subset only.

5. `train_cv.py`
   Train and evaluate HyPoNet across the five fixed patient-level evaluation runs.

6. `score_monitoring.py`
   Generate chronological prediction scores on a 20-second evaluation grid.

7. Run the corresponding evaluation scripts for segment-level, score-dynamics, or continuous-warning analyses.

Install dependencies with:

```bash
pip install -r requirements.txt
```

Run:

```bash
python scripts/<script_name>.py --help
```

for the arguments of each script.

`examples/README.md` describes the expected data layout.

## Scripts

| File                                      | Purpose                                                                                               |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `scripts/assign_patients.py`              | Create fixed patient-level fold and subset manifests.                                                 |
| `scripts/make_segments.py`                | Read VitalDB recordings and generate 60-second segments with hard and soft labels.                    |
| `scripts/build_folds.py`                  | Combine per-recording segments into train, validation, and test HDF5 files.                           |
| `scripts/extract_handcrafted_features.py` | Extract 81 physiological features and apply training-fitted preprocessing within each evaluation run. |
| `scripts/train_cv.py`                     | Train soft-supervised HyPoNet or the matched hard-supervised model across five evaluation runs.       |
| `scripts/score_monitoring.py`             | Generate chronological prediction scores and event annotations.                                       |
| `scripts/evaluate_monitoring.py`          | Select validation alarm thresholds and calculate test warning metrics.                                |
| `scripts/compare_monitoring.py`           | Compare hard- and soft-supervised warning results using paired patient-cluster bootstrap intervals.   |
| `scripts/evaluate_score_dynamics.py`      | Analyze stable-period score variation, pre-onset score rise, and temporal score ordering.             |
| `scripts/run_simple_references.py`        | Fit and evaluate MAP-only, trend-only, and MAP-plus-trend logistic reference models.                  |

## Library modules

| File                                            | Purpose                                                                                        |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `hyponet/preprocessing/segmentation.py`         | Load and clean physiological waveforms, identify POH episodes, and construct labeled segments. |
| `hyponet/preprocessing/soft_labels.py`          | Calculate the MAP, event-time, and trend components of the risk-informed soft target.          |
| `hyponet/preprocessing/folds.py`                | Manage patient-level fold assignment and HDF5 dataset construction.                            |
| `hyponet/preprocessing/handcrafted_features.py` | Compute the 81 handcrafted features and fit training-only preprocessing.                       |
| `hyponet/models/hyponet.py`                     | Define the DWT-Former stream, Patch-Conv stream, feature fusion, and prediction head.          |
| `hyponet/models/xresnet1d.py`                   | Define residual one-dimensional convolution blocks.                                            |
| `hyponet/models/basic_conv1d.py`                | Define supporting one-dimensional convolution layers.                                          |
| `hyponet/data.py`                               | Load waveform, handcrafted-feature, and label tensors from HDF5 files.                         |
| `hyponet/labels.py`                             | Provide soft-label mapping and component-combination utilities.                                |
| `hyponet/losses.py`                             | Implement the Bernoulli KL loss used for soft-target fitting.                                  |
| `hyponet/metrics.py`                            | Calculate segment-level prediction and target-fitting metrics.                                 |
| `hyponet/train.py`                              | Build the model and run fold-level training and evaluation.                                    |
| `hyponet/monitoring_data.py`                    | Construct chronological monitoring windows and prediction inputs.                              |
| `hyponet/monitoring.py`                         | Apply alarm thresholds and calculate event-level warning metrics.                              |
| `hyponet/monitoring_compare.py`                 | Perform matched hard-versus-soft warning comparisons.                                          |
| `hyponet/score_dynamics.py`                     | Calculate rolling score SD, pre-onset rise, and time-score correlation.                        |
| `hyponet/simple_references.py`                  | Fit and score the simple MAP/trend logistic reference models.                                  |

## Configuration and tests

`configs/example_config.yaml` lists the main model, preprocessing, and evaluation settings.

The `tests/` directory contains checks for:

* risk-informed label construction;
* handcrafted-feature preprocessing;
* patient-level fold assignment;
* Bernoulli KL loss;
* chronological monitoring;
* warning evaluation;
* simple reference models.


Raw VitalDB recordings are not redistributed in this repository and must be obtained separately under the applicable VitalDB access conditions.
