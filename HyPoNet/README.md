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
