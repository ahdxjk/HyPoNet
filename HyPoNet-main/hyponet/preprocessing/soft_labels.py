"""Risk-informed soft-target construction used by HyPoNet.

The selected manuscript target applies g(x) = 2x - x**2 to normalized MAP,
event-proximity, and pressure-trend cues, then weights them 0.5:0.4:0.1.
The result is a training target, not a calibrated event probability.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np


MappingFn = Callable[..., float]


def _normalize(x: float, x_min: float, x_max: float) -> float:
    if x_max == x_min:
        return 0.0
    return float(np.clip((x - x_min) / (x_max - x_min), 0.0, 1.0))


def linear_mapping(x: float, x_min: float, x_max: float, *unused) -> float:
    return _normalize(x, x_min, x_max)


def concave_mapping(x: float, x_min: float, x_max: float, *unused) -> float:
    """Quadratic concave mapping: f(z)=1-(1-z)^2."""

    z = _normalize(x, x_min, x_max)
    return float(1.0 - (1.0 - z) ** 2)


def convex_mapping(x: float, x_min: float, x_max: float, *unused) -> float:
    z = _normalize(x, x_min, x_max)
    return float(z**2)


def sigmoid_mapping(x: float, x_min: float, x_max: float, x_mu: float = 3.0, x_k: float = 0.5) -> float:
    raw = 1.0 / (1.0 + np.exp(-x_k * (x - x_mu)))
    low = 1.0 / (1.0 + np.exp(-x_k * (x_min - x_mu)))
    high = 1.0 / (1.0 + np.exp(-x_k * (x_max - x_mu)))
    if high == low:
        return 0.0
    return float(np.clip((raw - low) / (high - low), 0.0, 1.0))


def sine_mapping(x: float, x_min: float, x_max: float, *unused) -> float:
    z = _normalize(x, x_min, x_max)
    return float((np.sin(np.pi * z - np.pi / 2.0) + 1.0) / 2.0)


def cosine_mapping(x: float, x_min: float, x_max: float, *unused) -> float:
    z = _normalize(x, x_min, x_max)
    return float((1.0 - np.cos(np.pi * z)) / 2.0)


def reverse_z_131_mapping(x: float, x_min: float, x_max: float, *unused) -> float:
    z = _normalize(x, x_min, x_max)
    if z < 0.2:
        return 0.0
    if z < 0.8:
        return float((z - 0.2) / 0.6)
    return 1.0


def reverse_z_212_mapping(x: float, x_min: float, x_max: float, *unused) -> float:
    z = _normalize(x, x_min, x_max)
    if z < 0.4:
        return 0.0
    if z < 0.6:
        return float((z - 0.4) / 0.2)
    return 1.0


MAPPING_FUNCTIONS: Dict[str, MappingFn] = {
    "linear": linear_mapping,
    "concave": concave_mapping,
    "convex": convex_mapping,
    "sigmoid": sigmoid_mapping,
    "sine": sine_mapping,
    "cosine": cosine_mapping,
    "reverse_z_131": reverse_z_131_mapping,
    "reverse_z_212": reverse_z_212_mapping,
}


def calculate_trend_slope(art_segment: np.ndarray, window_seconds: int = 60, sample_rate: int = 100) -> float:
    """Return r = clip(a / 30, -1, 1), with a in mmHg/min.

    A five-second moving average uses only complete averaging windows. The
    ordinary least-squares slope is fitted against physical time in minutes.
    Missing samples can invalidate some smoothing windows; at least two valid
    smoothed points are required.
    """

    if sample_rate <= 0 or window_seconds <= 0:
        raise ValueError("sample_rate and window_seconds must be positive")
    art = np.asarray(art_segment, dtype=np.float64).reshape(-1)
    if art.size < window_seconds * sample_rate:
        return float("nan")

    smooth_window = 5 * sample_rate
    kernel = np.ones(smooth_window, dtype=np.float64) / smooth_window
    smoothed = np.convolve(art, kernel, mode="valid")
    valid = np.isfinite(smoothed)
    if np.count_nonzero(valid) < 2:
        return float("nan")

    time_minutes = np.flatnonzero(valid) / (sample_rate * 60.0)
    slope_mmhg_per_min, _ = np.polyfit(time_minutes, smoothed[valid], 1)
    return float(np.clip(slope_mmhg_per_min / 30.0, -1.0, 1.0))


def component_probabilities(
    time_value: float | None,
    map_value: float,
    trend_slope: float,
    pre_event_minutes: int,
    mapping_func: MappingFn = concave_mapping,
    *,
    ongoing_event: bool = False,
) -> Dict[str, float]:
    """Map the three manuscript cues to bounded target components.

    ``time_value`` is minutes from the input end to the next onset within the
    horizon, or None when the fully observed horizon contains no onset.
    An ongoing event takes priority over any future onset.
    ``trend_slope`` is the normalized r value from calculate_trend_slope.
    """

    if pre_event_minutes <= 0:
        raise ValueError("pre_event_minutes must be positive")
    if not np.isfinite(map_value) or not np.isfinite(trend_slope):
        raise ValueError("MAP and trend must be finite")

    if ongoing_event:
        time_cue = 1.0
    elif time_value is None:
        time_cue = 0.0
    else:
        if not np.isfinite(time_value) or time_value < 0:
            raise ValueError("time to onset must be finite and nonnegative")
        time_cue = float(np.clip(1.0 - time_value / pre_event_minutes, 0.0, 1.0))

    map_cue = float(np.clip((105.0 - map_value) / 40.0, 0.0, 1.0))
    trend_cue = (1.0 - float(np.clip(trend_slope, -1.0, 1.0))) / 2.0
    return {
        "time": float(mapping_func(time_cue, 0.0, 1.0)),
        "map": float(mapping_func(map_cue, 0.0, 1.0)),
        "trend": float(mapping_func(trend_cue, 0.0, 1.0)),
    }


def combined_probability(
    time_value: float | None,
    map_value: float,
    trend_slope: float,
    pre_event_minutes: int,
    mapping_func: MappingFn = concave_mapping,
    map_weight: float = 0.5,
    time_weight: float = 0.4,
    trend_weight: float = 0.1,
    *,
    ongoing_event: bool = False,
) -> float:
    """Combine MAP, time, and trend components into one soft target."""

    parts = component_probabilities(
        time_value, map_value, trend_slope, pre_event_minutes,
        mapping_func, ongoing_event=ongoing_event,
    )
    total_weight = map_weight + time_weight + trend_weight
    if total_weight <= 0:
        raise ValueError("At least one probability component weight must be positive.")
    value = (
        map_weight * parts["map"]
        + time_weight * parts["time"]
        + trend_weight * parts["trend"]
    ) / total_weight
    return float(np.clip(value, 0.0, 1.0))


def add_soft_labels_to_segments(
    segments: list[dict],
    pre_event_minutes: int,
    mapping_names: tuple[str, ...] = ("concave",),
    *,
    sample_rate: int = 100,
) -> list[dict]:
    """Attach soft targets to segments carrying explicit event context.

    Non-ongoing inputs require a complete future horizon, including inputs
    whose next onset is visible before a truncated recording ends.
    """

    converted: list[dict] = []
    for segment in segments:
        if "ongoing_event" not in segment or "future_horizon_complete" not in segment:
            raise ValueError("Segment must include ongoing_event and future_horizon_complete")
        if not segment["ongoing_event"] and not segment["future_horizon_complete"]:
            continue

        map_value = segment.get("map", np.nan)
        if not np.isfinite(map_value):
            continue
        time_value = segment.get("time_to_next_onset_minutes")
        trend_slope = calculate_trend_slope(segment["art"], sample_rate=sample_rate)
        if not np.isfinite(trend_slope):
            continue
        new_segment = dict(segment)
        new_segment["trend_slope"] = trend_slope

        for name in mapping_names:
            func = MAPPING_FUNCTIONS[name]
            new_segment[f"soft_label_{name}"] = combined_probability(
                time_value=time_value,
                map_value=map_value,
                trend_slope=trend_slope,
                pre_event_minutes=pre_event_minutes,
                mapping_func=func,
                ongoing_event=bool(segment["ongoing_event"]),
            )

        converted.append(new_segment)

    return converted
