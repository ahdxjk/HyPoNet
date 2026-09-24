"""Chronological 20-second monitoring inputs and score annotations.

Event timing is computed from the full ABP record solely for retrospective
annotations. The network receives only the preceding 60 seconds of signals
and static information at each decision time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch

from .preprocessing.handcrafted_features import MLFeatureExtractor, TrainingFeaturePreprocessor
from .preprocessing.segmentation import (
    SegmentationConfig,
    calculate_map_from_waveform,
    clean_art_pressure,
    detect_hypotension_intervals,
    is_valid_art_segment,
)


DECISION_COLUMNS = (
    "split", "case_id", "patient_id", "time_s", "score", "ongoing", "input_complete", "future_complete",
)
EVENT_COLUMNS = ("split", "case_id", "patient_id", "onset_s")


@dataclass
class MonitoringWindow:
    end_sample: int
    ongoing: bool
    input_complete: bool
    future_complete: bool
    signals: np.ndarray | None


def _fill_within_window(values: np.ndarray) -> np.ndarray | None:
    """Interpolate observed samples inside one input; never use future samples."""
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return None
    if finite.all():
        return values
    positions = np.arange(len(values))
    return np.interp(positions, positions[finite], values[finite]).astype(np.float32)


def iter_monitoring_windows(
    art: np.ndarray,
    ecg: np.ndarray,
    pleth: np.ndarray,
    intervals: list[dict],
    *,
    horizon_minutes: int = 5,
    sample_rate: int = 100,
    step_seconds: int = 20,
) -> Iterator[MonitoringWindow]:
    """Yield all 20-second grid decisions with a preceding 60-second input."""
    if sample_rate <= 0 or step_seconds <= 0 or horizon_minutes <= 0:
        raise ValueError("sample_rate, step_seconds, and horizon_minutes must be positive")

    input_samples = 60 * sample_rate
    step_samples = step_seconds * sample_rate
    horizon_samples = horizon_minutes * 60 * sample_rate
    art = np.asarray(art, dtype=np.float32)
    ecg = np.asarray(ecg, dtype=np.float32)
    pleth = np.asarray(pleth, dtype=np.float32)

    for end in range(input_samples, len(art) + 1, step_samples):
        start = end - input_samples
        ongoing = any(
            episode["start_idx"] < end and episode["end_idx"] > start
            for episode in intervals
        )
        future_complete = end + horizon_samples <= len(art)
        signals = None
        if end <= len(ecg) and end <= len(pleth):
            art_window = art[start:end]
            if is_valid_art_segment(art_window) and np.isfinite(
                calculate_map_from_waveform(art_window, sample_rate)
            ):
                channels = (
                    _fill_within_window(art_window),
                    _fill_within_window(ecg[start:end]),
                    _fill_within_window(pleth[start:end]),
                )
                if all(channel is not None for channel in channels):
                    signals = np.stack(channels, axis=0)
        yield MonitoringWindow(
            end_sample=end,
            ongoing=ongoing,
            input_complete=signals is not None,
            future_complete=future_complete,
            signals=signals,
        )


def score_case_arrays(
    art: np.ndarray,
    ecg: np.ndarray,
    pleth: np.ndarray,
    static_row: np.ndarray,
    model: torch.nn.Module,
    preprocessor: TrainingFeaturePreprocessor,
    *,
    split: str,
    case_id: str,
    patient_id: str | None = None,
    device: torch.device,
    batch_size: int = 32,
    horizon_minutes: int = 5,
    sample_rate: int = 100,
) -> tuple[list[dict], list[dict]]:
    """Score one recording and return decision and episode-onset CSV rows."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if patient_id is None:
        patient_id = case_id
    clean_art = clean_art_pressure(art)
    intervals = detect_hypotension_intervals(
        clean_art,
        SegmentationConfig(sample_rate=sample_rate, horizon_minutes=horizon_minutes),
    )
    events = [
        {"split": split, "case_id": case_id, "patient_id": patient_id,
         "onset_s": episode["start_idx"] / sample_rate}
        for episode in intervals
    ]

    rows: list[dict] = []
    pending_indices: list[int] = []
    pending_signals: list[np.ndarray] = []
    extractor = MLFeatureExtractor()
    static_row = np.asarray(static_row, dtype=np.float32).reshape(1, -1)

    def flush_pending() -> None:
        if not pending_signals:
            return
        signals = np.stack(pending_signals, axis=0)
        static_rows = np.repeat(static_row, len(signals), axis=0)
        raw_features = extractor.extract_features(signals, static_rows)
        features = preprocessor.transform(raw_features)
        with torch.inference_mode():
            logits = model(
                torch.from_numpy(signals).to(device),
                torch.from_numpy(features).to(device),
            )
            scores = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        if not np.isfinite(scores).all():
            raise ValueError(f"Model produced nonfinite scores for case {case_id}.")
        for row_index, score in zip(pending_indices, scores):
            rows[row_index]["score"] = float(score)
        pending_indices.clear()
        pending_signals.clear()

    model.eval()
    for window in iter_monitoring_windows(
        clean_art, ecg, pleth, intervals,
        horizon_minutes=horizon_minutes, sample_rate=sample_rate,
    ):
        rows.append({
            "split": split,
            "case_id": case_id,
            "patient_id": patient_id,
            "time_s": window.end_sample / sample_rate,
            "score": None,
            "ongoing": window.ongoing,
            "input_complete": window.input_complete,
            "future_complete": window.future_complete,
        })
        if window.signals is not None:
            pending_indices.append(len(rows) - 1)
            pending_signals.append(window.signals)
            if len(pending_signals) >= batch_size:
                flush_pending()
    flush_pending()
    return rows, events
