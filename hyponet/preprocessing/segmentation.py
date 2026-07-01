"""Segment extraction and label generation from raw waveform records."""

from __future__ import annotations

import argparse
import gc
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from joblib import Parallel, delayed
from scipy.signal import butter, filtfilt, find_peaks

from .soft_labels import add_soft_labels_to_segments


SOFT_LABEL_OUTPUTS = {
    "LABELS_LINEAR": "soft_label_linear",
    "LABELS_CONCAVE": "soft_label_concave",
    "LABELS_CONVEX": "soft_label_convex",
    "LABELS_SIGMOID": "soft_label_sigmoid",
    "LABELS_SINE": "soft_label_sine",
    "LABELS_COSINE": "soft_label_cosine",
    "LABELS_REVERSE_Z_131": "soft_label_reverse_z_131",
    "LABELS_REVERSE_Z_212": "soft_label_reverse_z_212",
}


@dataclass(frozen=True)
class SegmentationConfig:
    sample_rate: int = 100
    horizon_minutes: int = 5
    chunk_seconds: int = 60
    hypotension_threshold: float = 65.0
    event_window_seconds: int = 60
    event_step_seconds: int = 60
    balance_hard_labels: bool = True
    legacy_interval_scan: bool = True


def horizon_default_step_seconds(horizon_minutes: int) -> int:
    """Return the step setting used in the original horizon scripts."""

    return 20 if horizon_minutes == 10 else 60


def clean_art_pressure(art: np.ndarray) -> np.ndarray:
    """Set physiologically implausible arterial pressure values to NaN."""

    art = np.asarray(art, dtype=np.float32).copy()
    art[(art < 30) | (art > 250)] = np.nan
    return art


def is_valid_art_segment(segment: np.ndarray) -> bool:
    """Check whether a 60-second ART segment is usable."""

    if len(segment) == 0 or np.isnan(segment).all():
        return False
    mean_value = np.nanmean(segment)
    if mean_value < 10 or mean_value > 150:
        return False
    if np.isnan(segment).sum() > 100:
        return False
    if np.nanmax(segment) - np.nanmin(segment) > 100:
        return False
    return True


def calculate_map_from_waveform(art_segment: np.ndarray, sample_rate: int = 100) -> float:
    """Estimate MAP from an ART waveform using SBP/DBP peak detection."""

    if len(art_segment) < sample_rate or np.isnan(art_segment).all():
        return float("nan")

    art = np.asarray(art_segment, dtype=np.float64)
    finite = np.isfinite(art)
    if finite.sum() < sample_rate:
        return float("nan")
    if not finite.all():
        x = np.arange(len(art))
        art = np.interp(x, x[finite], art[finite])

    try:
        b, a = butter(3, 10 / (sample_rate / 2.0), btype="lowpass")
        smoothed = filtfilt(b, a, art)
    except Exception:
        return float("nan")

    peak_distance = sample_rate // 2
    sbp_indices, _ = find_peaks(
        smoothed,
        height=np.mean(smoothed) + 0.05 * np.std(smoothed),
        distance=peak_distance,
    )
    dbp_indices, _ = find_peaks(
        -smoothed,
        height=np.mean(-smoothed) + 0.05 * np.std(-smoothed),
        distance=peak_distance,
    )

    if len(dbp_indices) < 3 or len(sbp_indices) == 0:
        return float("nan")

    dbp_diffs = np.diff(dbp_indices)
    lower = np.mean(dbp_diffs) - 1.8 * np.std(dbp_diffs)
    upper = np.mean(dbp_diffs) + 1.8 * np.std(dbp_diffs)

    valid_dbp = []
    for i, dbp_idx in enumerate(dbp_indices):
        if i == 0 or i == len(dbp_indices) - 1:
            valid_dbp.append(dbp_idx)
            continue
        prev_diff = dbp_indices[i] - dbp_indices[i - 1]
        next_diff = dbp_indices[i + 1] - dbp_indices[i]
        if lower <= prev_diff <= upper or lower <= next_diff <= upper:
            valid_dbp.append(dbp_idx)

    map_values = []
    for sbp_idx, dbp_idx in zip(sbp_indices, valid_dbp):
        map_values.append((art[sbp_idx] + 2.0 * art[dbp_idx]) / 3.0)

    return float(np.mean(map_values)) if map_values else float("nan")


def merge_intervals(intervals: Iterable[dict]) -> list[dict]:
    """Merge overlapping or adjacent intervals."""

    sorted_intervals = sorted(intervals, key=lambda item: item["start_idx"])
    if not sorted_intervals:
        return []

    merged = []
    current_start = sorted_intervals[0]["start_idx"]
    current_end = sorted_intervals[0]["end_idx"]
    for interval in sorted_intervals[1:]:
        start = interval["start_idx"]
        end = interval["end_idx"]
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            merged.append({"start_idx": current_start, "end_idx": current_end})
            current_start, current_end = start, end
    merged.append({"start_idx": current_start, "end_idx": current_end})
    return merged


def detect_hypotension_intervals(art: np.ndarray, config: SegmentationConfig) -> list[dict]:
    """Detect 60-second windows whose estimated MAP is below threshold.

    With ``legacy_interval_scan=True`` this follows the original scripts:
    event candidates are aligned to non-overlapping 60-second windows, while
    ``event_step_seconds`` controls how many windows are skipped after a
    positive detection. Set ``legacy_interval_scan=False`` for a conventional
    sliding-window scan using ``event_step_seconds`` as the actual stride.
    """

    intervals: list[dict] = []
    total_length = len(art)
    window_samples = config.event_window_seconds * config.sample_rate
    step_samples = config.event_step_seconds * config.sample_rate

    if config.legacy_interval_scan:
        n_windows = total_length // window_samples
        processed = np.zeros(n_windows, dtype=bool)
        current_window = 0
        while current_window < n_windows:
            if processed[current_window]:
                current_window += 1
                continue

            start = current_window * window_samples
            end = start + window_samples
            segment = art[start:end]
            processed[current_window] = True

            if not is_valid_art_segment(segment):
                current_window += 1
                continue

            map_value = calculate_map_from_waveform(segment, config.sample_rate)
            if map_value < config.hypotension_threshold:
                intervals.append({"start_idx": start, "end_idx": end})
                current_window += max(1, config.event_window_seconds // max(1, config.event_step_seconds))
            else:
                current_window += 1
        return intervals

    for start in range(0, total_length - window_samples + 1, step_samples):
        end = start + window_samples
        segment = art[start:end]
        if not is_valid_art_segment(segment):
            continue
        map_value = calculate_map_from_waveform(segment, config.sample_rate)
        if map_value < config.hypotension_threshold:
            intervals.append({"start_idx": start, "end_idx": end})
    return intervals


def chunk_signal_into_segments(
    art: np.ndarray,
    ecg: np.ndarray,
    pleth: np.ndarray,
    intervals: list[dict],
    config: SegmentationConfig,
) -> list[dict]:
    """Create 60-second event and pre-event segments."""

    segments: list[dict] = []
    used_ranges: set[tuple[int, int]] = set()
    chunk_samples = config.chunk_seconds * config.sample_rate

    def add_chunk(start: int, end: int, label: int, direction: str) -> None:
        if (start, end) in used_ranges:
            return
        if start < 0 or end > len(art):
            return
        art_segment = art[start:end]
        if not is_valid_art_segment(art_segment):
            return
        segments.append(
            {
                "art": art_segment.astype(np.float32, copy=False),
                "ecg": ecg[start:end].astype(np.float32, copy=False),
                "pleth": pleth[start:end].astype(np.float32, copy=False),
                "label": int(label),
                "global_start_idx": int(start),
                "global_end_idx": int(end),
                "map": calculate_map_from_waveform(art_segment, config.sample_rate),
                "direction": direction,
            }
        )
        used_ranges.add((start, end))

    for interval in merge_intervals(intervals):
        start_idx = interval["start_idx"]
        end_idx = interval["end_idx"]

        current = start_idx
        while current + chunk_samples <= end_idx:
            add_chunk(current, current + chunk_samples, config.horizon_minutes, "in_event")
            current += 60 * config.sample_rate

        for minute in range(1, config.horizon_minutes + 1):
            lead_end = start_idx - (minute - 1) * 60 * config.sample_rate
            lead_start = lead_end - chunk_samples
            if lead_start >= 0:
                add_chunk(lead_start, lead_end, config.horizon_minutes - minute, "leading")

    return segments


def label_remaining_chunks_as_zero(
    art: np.ndarray,
    ecg: np.ndarray,
    pleth: np.ndarray,
    already_labeled: list[dict],
    config: SegmentationConfig,
) -> list[dict]:
    """Label valid non-event 60-second chunks as normal."""

    chunk_samples = config.chunk_seconds * config.sample_rate
    processed = np.zeros(len(art), dtype=bool)
    for segment in already_labeled:
        processed[segment["global_start_idx"] : segment["global_end_idx"]] = True

    normal_segments: list[dict] = []
    for start in range(0, len(art) - chunk_samples + 1, chunk_samples):
        end = start + chunk_samples
        if np.any(processed[start:end]):
            continue
        art_segment = art[start:end]
        if not is_valid_art_segment(art_segment):
            continue
        normal_segments.append(
            {
                "art": art_segment.astype(np.float32, copy=False),
                "ecg": ecg[start:end].astype(np.float32, copy=False),
                "pleth": pleth[start:end].astype(np.float32, copy=False),
                "label": 0,
                "global_start_idx": int(start),
                "global_end_idx": int(end),
                "map": calculate_map_from_waveform(art_segment, config.sample_rate),
                "direction": "none",
            }
        )
    return normal_segments


def assign_hard_labels(segments: list[dict], config: SegmentationConfig) -> list[dict]:
    """Assign binary POH labels.

    Event chunks and all chunks ending within the prediction horizon before an
    event are hard-positive. All remaining chunks are hard-negative.
    """

    if not segments:
        return []

    sorted_segments = sorted(segments, key=lambda item: item["global_start_idx"])
    pre_event_samples = config.horizon_minutes * 60 * config.sample_rate
    event_starts = [
        segment["global_start_idx"]
        for segment in sorted_segments
        if segment["label"] == config.horizon_minutes
    ]

    for segment in sorted_segments:
        segment["hard_label"] = 0

    for segment in sorted_segments:
        if segment["label"] == config.horizon_minutes:
            segment["hard_label"] = 1
            continue
        segment_end = segment["global_end_idx"]
        if any(segment_end <= event_start and segment_end > event_start - pre_event_samples for event_start in event_starts):
            segment["hard_label"] = 1

    return sorted_segments


def balance_segments_by_hard_label(segments: list[dict], config: SegmentationConfig) -> list[dict]:
    """Downsample normal segments to balance hard-positive and hard-negative classes."""

    if not segments:
        return []

    sorted_segments = sorted(segments, key=lambda item: item["global_start_idx"])
    positives = [segment for segment in sorted_segments if segment["hard_label"] == 1]
    negatives = [segment for segment in sorted_segments if segment["hard_label"] == 0]
    if not positives or not negatives:
        return []

    pre_event_samples = config.horizon_minutes * 60 * config.sample_rate
    event_starts = [segment["global_start_idx"] for segment in sorted_segments if segment["label"] == config.horizon_minutes]

    leading_negatives = []
    other_negatives = []
    for segment in negatives:
        end = segment["global_end_idx"]
        if any(end <= event_start and end > event_start - pre_event_samples for event_start in event_starts):
            leading_negatives.append(segment)
        else:
            other_negatives.append(segment)

    n_pos = len(positives)
    if len(leading_negatives) > n_pos:
        leading_negatives = random.sample(leading_negatives, n_pos)
    remaining_needed = min(n_pos - len(leading_negatives), len(other_negatives))
    sampled_other = random.sample(other_negatives, remaining_needed) if remaining_needed > 0 else []

    balanced = positives + leading_negatives + sampled_other
    random.shuffle(balanced)
    return balanced


def process_waveform_arrays(
    art: np.ndarray,
    ecg: np.ndarray,
    pleth: np.ndarray,
    config: SegmentationConfig,
) -> list[dict]:
    """Run the complete segment-label pipeline on in-memory waveforms."""

    art = clean_art_pressure(art)
    intervals = detect_hypotension_intervals(art, config)
    segments = chunk_signal_into_segments(art, ecg, pleth, intervals, config)
    segments.extend(label_remaining_chunks_as_zero(art, ecg, pleth, segments, config))
    segments = add_soft_labels_to_segments(segments, config.horizon_minutes)
    segments = assign_hard_labels(segments, config)
    if config.balance_hard_labels:
        segments = balance_segments_by_hard_label(segments, config)
    return segments


def save_case_segments(case_name: str, case_id: int, segments: list[dict], output_dir: Path) -> None:
    """Save one case's segments as per-case NumPy arrays."""

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(case_name).stem

    arrays = {
        "ART": np.asarray([segment["art"] for segment in segments], dtype=np.float32),
        "ECG": np.asarray([segment["ecg"] for segment in segments], dtype=np.float32),
        "PLETH": np.asarray([segment["pleth"] for segment in segments], dtype=np.float32),
        "HARD_LABELS": np.asarray([segment.get("hard_label", 0) for segment in segments], dtype=np.int32),
        "STATIC": np.asarray([case_id for _ in segments], dtype=np.int32),
    }
    for output_name, field_name in SOFT_LABEL_OUTPUTS.items():
        arrays[output_name] = np.asarray([segment.get(field_name, 0.0) for segment in segments], dtype=np.float32)

    for name, value in arrays.items():
        np.save(output_dir / f"{stem}_{name}.npy", value)


def load_vital_waveforms(vital_file: Path, sample_rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load ART, ECG II, and PLETH from a VitalDB .vital file."""

    import vitaldb

    tracks = ["SNUADC/ART", "SNUADC/ECG_II", "SNUADC/PLETH"]
    vf = vitaldb.VitalFile(str(vital_file), tracks)
    art = vf.to_numpy(["SNUADC/ART"], 1 / sample_rate).flatten()
    ecg = vf.to_numpy(["SNUADC/ECG_II"], 1 / sample_rate).flatten()
    pleth = vf.to_numpy(["SNUADC/PLETH"], 1 / sample_rate).flatten()
    return art, ecg, pleth


def process_vital_file(vital_file: Path, output_dir: Path, config: SegmentationConfig) -> None:
    """Process a single .vital file and save per-case NumPy arrays."""

    match = re.match(r"(\d+)", vital_file.stem)
    case_id = int(match.group(1)) if match else -1
    art, ecg, pleth = load_vital_waveforms(vital_file, config.sample_rate)
    segments = process_waveform_arrays(art, ecg, pleth, config)
    save_case_segments(vital_file.name, case_id, segments, output_dir)
    del art, ecg, pleth, segments
    gc.collect()


def process_vital_directory(input_dir: Path, output_dir: Path, config: SegmentationConfig, n_jobs: int = 1) -> None:
    """Process all .vital files in a directory."""

    files = sorted(path for path in input_dir.iterdir() if path.suffix.lower() == ".vital")
    Parallel(n_jobs=n_jobs)(
        delayed(process_vital_file)(vital_file, output_dir, config)
        for vital_file in files
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate HyPo-Net waveform segments and soft labels.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing .vital files.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for per-case .npy outputs.")
    parser.add_argument("--horizon-minutes", type=int, required=True, choices=[5, 10, 15])
    parser.add_argument("--sample-rate", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=65.0)
    parser.add_argument("--event-step-seconds", type=int, default=None)
    parser.add_argument("--no-balance", action="store_true", help="Disable within-case hard-label balancing.")
    parser.add_argument("--sliding-interval-scan", action="store_true", help="Use event_step_seconds as a true sliding stride.")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    step_seconds = args.event_step_seconds
    if step_seconds is None:
        step_seconds = horizon_default_step_seconds(args.horizon_minutes)
    config = SegmentationConfig(
        sample_rate=args.sample_rate,
        horizon_minutes=args.horizon_minutes,
        hypotension_threshold=args.threshold,
        event_step_seconds=step_seconds,
        balance_hard_labels=not args.no_balance,
        legacy_interval_scan=not args.sliding_interval_scan,
    )
    process_vital_directory(args.input_dir, args.output_dir, config, args.n_jobs)


if __name__ == "__main__":
    main()
