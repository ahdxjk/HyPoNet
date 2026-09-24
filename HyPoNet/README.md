# HyPoNet

PyTorch code for perioperative hypotension (POH) risk prediction. Each model input contains a 60-second, 100 Hz window of ABP, lead-II ECG, and PPG (`3 × 6000`), together with 81 handcrafted features. The prediction horizons are 5, 10, and 15 minutes.

## Run order

1. `assign_patients.py`: create patient-level train/validation/test manifests.
2. `make_segments.py`: extract waveform windows and construct hard and soft labels.
3. `build_folds.py`: assemble the windows into HDF5 folds.
4. `extract_handcrafted_features.py`: compute and standardize the 81 features.
5. `train_cv.py`: train and evaluate the five folds.
6. `score_monitoring.py`: score recordings on a chronological 20-second grid.
7. Run the evaluation scripts needed for the analysis.

Install dependencies with `pip install -r requirements.txt`. Run `python scripts/<script_name>.py --help` for each script's arguments. `examples/README.md` shows the expected data layout.

## Reproducibility and evaluation notes

This repository is organized to mirror the evaluation pipeline described in the manuscript, from patient-level partitioning and segment generation to model training and chronological warning analysis.

* **Patient-level data separation.** Patient assignment is performed before segment generation. All recordings and derived 60-second windows from the same patient remain within the same training, validation, or test subset in each evaluation run.
* **Fixed manifests across comparisons.** The same patient manifests are reused across prediction horizons and matched supervision comparisons so that performance differences are not caused by different patient allocations.
* **No future information at inference.** Future POH onset information is used only retrospectively to construct supervision targets and evaluation labels. The model receives only the observed physiological signals, handcrafted features, and available static variables at inference time.
* **Soft targets are supervision signals rather than calibrated probabilities.** The risk-informed target is designed to encode temporal proximity and current hemodynamic state during training and should not be interpreted as a calibrated probability of POH occurrence.
* **Matched hard-versus-soft evaluation.** The hard- and soft-supervised HyPoNet models can be trained under the same architecture, patient partitions, waveform inputs, feature preprocessing, and optimization settings, differing only in the supervision target.
* **Separate segment-level and chronological evaluations.** Segment-level experiments evaluate horizon-specific discrimination, whereas the chronological monitoring pipeline excludes ongoing POH and evaluates strictly pre-onset warning behavior, including event sensitivity, false-alarm burden, lead time, and pre-onset coverage.
* **Validation-only threshold selection.** Alarm thresholds are selected using validation recordings and then applied unchanged to the corresponding test recordings.
* **Common evaluation grid for reference models.** MAP-only, trend-only, and MAP-plus-trend reference models can be evaluated on the same eligible chronological decisions used for HyPoNet, enabling direct comparison under the same warning protocol.

These design choices are intended to keep patient separation, supervision comparisons, and chronological warning evaluation explicit and auditable.

## Manuscript version and result traceability

The manuscript results are associated with a fixed code and configuration snapshot.

* Manuscript version: `HyPoNet revision <VERSION>`
* Code release/tag: `<TAG>`
* Commit: `<COMMIT_SHA>`
* Main configuration: `configs/<PAPER_CONFIG>.yaml`
* Patient split manifests: `<PATH_TO_SPLITS>`
* Reported experiment outputs: `<PATH_TO_RESULTS>`

The same fixed patient manifests should be used when reproducing hard-versus-soft supervision comparisons and model baselines.

Raw VitalDB recordings are not redistributed in this repository and must be obtained separately under the applicable VitalDB access conditions.


## Scripts

| File | Purpose |
| --- | --- |
| `scripts/assign_patients.py` | Create fixed patient-level fold manifests. |
| `scripts/make_segments.py` | Read VitalDB recordings and save 60-second segments with labels. |
| `scripts/build_folds.py` | Combine per-case segments into train, validation, and test HDF5 files. |
| `scripts/extract_handcrafted_features.py` | Extract 81 features and apply training-fitted preprocessing to each fold. |
| `scripts/train_cv.py` | Train soft-supervised HyPoNet or the matched hard-supervised model across five folds. |
| `scripts/score_monitoring.py` | Generate chronological prediction scores and event annotations. |
| `scripts/evaluate_monitoring.py` | Select validation alarm thresholds and calculate test warning metrics. |
| `scripts/compare_monitoring.py` | Compare hard- and soft-supervised warning results with paired patient bootstrap intervals. |
| `scripts/evaluate_score_dynamics.py` | Analyze stable-period score variation and pre-onset score behavior. |
| `scripts/run_simple_references.py` | Fit and evaluate MAP, trend, and MAP-plus-trend logistic reference models. |

## Library modules

| File | Purpose |
| --- | --- |
| `hyponet/preprocessing/segmentation.py` | Load and clean waveforms, detect POH episodes, and create labeled segments. |
| `hyponet/preprocessing/soft_labels.py` | Calculate the MAP, event-time, and trend components of the soft target. |
| `hyponet/preprocessing/folds.py` | Assign patient-level folds and write HDF5 datasets. |
| `hyponet/preprocessing/handcrafted_features.py` | Compute the 81 features and fit training-only feature preprocessing. |
| `hyponet/models/hyponet.py` | Define the DWT-Former, Patch-Conv, feature fusion, and prediction head. |
| `hyponet/models/xresnet1d.py` | Define residual one-dimensional convolution blocks. |
| `hyponet/models/basic_conv1d.py` | Define supporting one-dimensional convolution layers. |
| `hyponet/data.py` | Load waveform, feature, and label tensors from HDF5. |
| `hyponet/labels.py` | Provide soft-label mapping and component-combination helpers. |
| `hyponet/losses.py` | Implement Bernoulli KL training loss. |
| `hyponet/metrics.py` | Calculate segment-level prediction and target-fitting metrics. |
| `hyponet/train.py` | Build the model and run fold-level training and evaluation. |
| `hyponet/monitoring_data.py` | Construct chronological monitoring windows and scores. |
| `hyponet/monitoring.py` | Apply alarm thresholds and calculate warning metrics. |
| `hyponet/monitoring_compare.py` | Calculate matched hard/soft warning comparisons. |
| `hyponet/score_dynamics.py` | Calculate rolling score SD, pre-onset rise, and time-score correlation. |
| `hyponet/simple_references.py` | Fit and score the logistic reference models. |

`configs/example_config.yaml` lists the main settings. `tests/` contains checks for labels, feature preprocessing, patient folds, losses, monitoring, and reference models.
