"""Segment extraction and label generation from raw waveform records."""

from __future__ import annotations

import argparse
import csv
import gc
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
    "LABELS_CONCAVE": "soft_label_concave",
}


@dataclass(frozen=True)
class SegmentationConfig:
    sample_rate: int = 100
    horizon_minutes: int = 5
    chunk_seconds: int = 60
    hypotension_threshold: float = 65.0
    balance_hard_labels: bool = False


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
    """Mean pulse MAP from the same pulse detector used for POH episodes."""

    pulses = _pulse_map_intervals(art_segment, sample_rate)
    return float(np.mean([value for _, _, value in pulses])) if pulses else float("nan")


def _pulse_map_intervals(art: np.ndarray, sample_rate: int) -> list[tuple[int, int, float]]:
    """Return observed pulse spans and (systolic + 2*diastolic)/3 values.

    Missing pressure samples split the record. Each pulse is bounded by two
    consecutive diastolic troughs, so episode timing and segment MAP use the
    same explicit detector. The manuscript leaves its peak-finding settings
    unspecified; the cutoff, spacing, and prominence here are implementation
    choices.
    """

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    art = np.asarray(art, dtype=np.float64).reshape(-1)
    finite = np.isfinite(art)
    boundaries = np.diff(np.r_[False, finite, False].astype(np.int8))
    run_starts = np.flatnonzero(boundaries == 1)
    run_ends = np.flatnonzero(boundaries == -1)
    pulses: list[tuple[int, int, float]] = []

    for run_start, run_end in zip(run_starts, run_ends):
        waveform = art[run_start:run_end]
        if waveform.size < 3 * sample_rate:
            continue
        try:
            cutoff_hz = min(10.0, 0.45 * sample_rate)
            b, a = butter(3, cutoff_hz / (sample_rate / 2.0), btype="lowpass")
            smoothed = filtfilt(b, a, waveform)
        except ValueError:
            continue

        troughs, _ = find_peaks(
            -smoothed,
            distance=max(1, int(0.5 * sample_rate)),
            prominence=3.0,
        )
        for left, right in zip(troughs[:-1], troughs[1:]):
            if right <= left + 1:
                continue
            peak = left + int(np.argmax(smoothed[left:right]))
            if smoothed[peak] - smoothed[left] < 3.0:
                continue
            systolic = waveform[peak]
            diastolic = waveform[left]
            pulses.append((
                int(run_start + left), int(run_start + right),
                float((systolic + 2.0 * diastolic) / 3.0),
            ))
    return pulses


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
    """Find episodes with pulse-derived MAP <65 mmHg for *more than* 60 s.

    A 60-second window below threshold is insufficient under the manuscript's
    episode definition. Instead, estimate MAP for each complete ABP pulse and
    find continuous below-threshold runs. Pulse detection details are an
    implementation choice because the manuscript does not specify them.
    """

    sample_rate = config.sample_rate
    art = np.asarray(art, dtype=np.float64)
    pulse_map = np.full(art.size, np.nan, dtype=np.float64)
    # Do not bridge missing or implausible pressure readings: gaps interrupt
    # the observed duration of a hypotensive episode.
    for left, right, map_value in _pulse_map_intervals(art, sample_rate):
        pulse_map[left:right] = map_value

    below = pulse_map < config.hypotension_threshold
    boundaries = np.diff(np.r_[False, below, False].astype(np.int8))
    starts = np.flatnonzero(boundaries == 1)
    ends = np.flatnonzero(boundaries == -1)
    minimum_samples = 60 * sample_rate
    return [
        {"start_idx": int(start), "end_idx": int(end)}
        for start, end in zip(starts, ends)
        if end - start > minimum_samples
    ]


def chunk_signal_into_segments(
    art: np.ndarray,
    ecg: np.ndarray,
    pleth: np.ndarray,
    intervals: list[dict],
    config: SegmentationConfig,
) -> list[dict]:
    """Create the existing sampled event and pre-event chunks.

    The integer ``label`` is retained only for the original sampling and
    balancing scheme. Event timing and final labels are derived separately
    from detected intervals, not from this coarse integer.
    """

    segments: list[dict] = []
    used_ranges: set[tuple[int, int]] = set()
    chunk_samples = config.chunk_seconds * config.sample_rate

    def add_chunk(start: int, end: int, label: int, direction: str) -> None:
        if (start, end) in used_ranges:
            return
        if start < 0 or end > min(len(art), len(ecg), len(pleth)):
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
        if end > min(len(ecg), len(pleth)):
            continue
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


def annotate_segment_event_context(
    segments: list[dict],
    intervals: list[dict],
    record_samples: int,
    config: SegmentationConfig,
) -> list[dict]:
    """Identify ongoing POH and the next onset from the event intervals.

    Non-ongoing windows with less than a full observed future horizon are
    excluded even when an onset occurs before the recording ends. This follows
    the manuscript's matched-experiment eligibility rule. The existing sampled
    segment cadence is intentionally unchanged.
    """

    events = merge_intervals(intervals)
    horizon_samples = config.horizon_minutes * 60 * config.sample_rate
    annotated: list[dict] = []
    for segment in segments:
        start = segment["global_start_idx"]
        end = segment["global_end_idx"]
        ongoing = any(
            event["start_idx"] < end and event["end_idx"] > start
            for event in events
        )
        future_complete = end + horizon_samples <= record_samples
        if not ongoing and not future_complete:
            continue

        next_onset = next(
            (event["start_idx"] for event in events if event["start_idx"] >= end),
            None,
        )
        if next_onset is not None and next_onset - end <= horizon_samples:
            time_to_onset = (next_onset - end) / (60.0 * config.sample_rate)
        else:
            time_to_onset = None

        annotated_segment = dict(segment)
        annotated_segment["ongoing_event"] = ongoing
        annotated_segment["future_horizon_complete"] = future_complete
        annotated_segment["time_to_next_onset_minutes"] = time_to_onset
        annotated.append(annotated_segment)
    return annotated


def assign_hard_labels(segments: list[dict], config: SegmentationConfig) -> list[dict]:
    """Assign binary POH labels.

    An input containing ongoing POH or ending at most H minutes before the
    next onset is positive. An onset at exactly H minutes is included.
    """

    if not segments:
        return []

    sorted_segments = sorted(segments, key=lambda item: item["global_start_idx"])
    for segment in sorted_segments:
        if "ongoing_event" not in segment or "time_to_next_onset_minutes" not in segment:
            raise ValueError("Event context must be annotated before hard labeling")
        time_to_onset = segment["time_to_next_onset_minutes"]
        segment["hard_label"] = int(
            segment["ongoing_event"]
            or (time_to_onset is not None and time_to_onset <= config.horizon_minutes)
        )
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
    segments = annotate_segment_event_context(segments, intervals, len(art), config)
    segments = add_soft_labels_to_segments(
        segments, config.horizon_minutes, sample_rate=config.sample_rate,
    )
    segments = assign_hard_labels(segments, config)
    if config.balance_hard_labels:
        segments = balance_segments_by_hard_label(segments, config)
    return segments


def save_case_segments(
    case_name: str,
    case_id: int,
    segments: list[dict],
    output_dir: Path,
    sample_rate: int = 100,
) -> None:
    """Save one case's segments as per-case NumPy arrays."""

    if case_id < 0:
        raise ValueError(f"A numeric case ID is required for {case_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{case_id:04d}"
    waveform_samples = 60 * sample_rate

    def waveform_array(field: str) -> np.ndarray:
        if not segments:
            return np.empty((0, waveform_samples), dtype=np.float32)
        return np.asarray([segment[field] for segment in segments], dtype=np.float32)

    arrays = {
        "ART": waveform_array("art"),
        "ECG": waveform_array("ecg"),
        "PLETH": waveform_array("pleth"),
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
    if match is None:
        raise ValueError(f"Vital filename has no leading numeric case ID: {vital_file.name}")
    case_id = int(match.group(1))
    art, ecg, pleth = load_vital_waveforms(vital_file, config.sample_rate)
    segments = process_waveform_arrays(art, ecg, pleth, config)
    save_case_segments(vital_file.name, case_id, segments, output_dir, config.sample_rate)
    del art, ecg, pleth, segments
    gc.collect()


def _case_key(value: str) -> str:
    """Match a case by its leading numeric ID or, otherwise, exact stem."""

    stem = Path(value.strip()).stem
    match = re.match(r"^(\d+)", stem)
    return str(int(match.group(1))) if match else stem.lower()


def load_case_list(path: Path) -> set[str]:
    """Load case IDs from a text file or a CSV with a case_id column."""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        if path.suffix.lower() == ".csv":
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError(f"Case list is empty: {path}")
            id_column = next(
                (name for name in reader.fieldnames if name.lower().strip() in {"case_id", "caseid"}),
                None,
            )
            if id_column is None:
                raise ValueError("CSV case list requires a case_id column")
            values = [row[id_column] for row in reader]
        else:
            values = [line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#")]
    case_keys = {_case_key(value) for value in values if value and value.strip()}
    if not case_keys:
        raise ValueError(f"Case list has no case IDs: {path}")
    return case_keys


def process_vital_directory(
    input_dir: Path,
    output_dir: Path,
    config: SegmentationConfig,
    n_jobs: int = 1,
    case_list: Path | None = None,
) -> None:
    """Process listed .vital cases, or all cases when no list is supplied."""

    files = sorted(path for path in input_dir.iterdir() if path.suffix.lower() == ".vital")
    if case_list is not None:
        selected = load_case_list(case_list)
        files = [path for path in files if _case_key(path.stem) in selected]
        if not files:
            raise ValueError(f"No .vital files matched case list: {case_list}")
    case_keys = [_case_key(path.stem) for path in files]
    if len(case_keys) != len(set(case_keys)):
        raise ValueError("Multiple .vital files share a case ID and would overwrite one output")
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
    parser.add_argument("--case-list", type=Path, help="Text IDs or CSV with case_id; filter before segmentation.")
    parser.add_argument(
        "--balance-hard-labels", action="store_true",
        help="Opt into the original within-case negative downsampling strategy.",
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    config = SegmentationConfig(
        sample_rate=args.sample_rate,
        horizon_minutes=args.horizon_minutes,
        hypotension_threshold=args.threshold,
        balance_hard_labels=args.balance_hard_labels,
    )
    process_vital_directory(args.input_dir, args.output_dir, config, args.n_jobs, args.case_list)


if __name__ == "__main__":
    main()
