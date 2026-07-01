"""Soft probability-label construction used by HyPo-Net.

The selected manuscript design combines three probability components in the
order MAP:time:trend = 5:4:1. In normalized weights this is
map_weight=0.5, time_weight=0.4, and trend_weight=0.1.
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
    """Return a normalized MAP trend proxy in [-1, 1].

    Negative slope means decreasing arterial pressure and should increase risk.
    """

    window_samples = window_seconds * sample_rate
    if len(art_segment) < window_samples:
        return 0.0

    smooth_window = max(1, 5 * sample_rate)
    kernel = np.ones(smooth_window, dtype=np.float64) / smooth_window
    smoothed = np.convolve(np.asarray(art_segment, dtype=np.float64), kernel, mode="valid")
    if smoothed.size < 2 or np.isnan(smoothed).all():
        return 0.0

    valid = np.isfinite(smoothed)
    if valid.sum() < 2:
        return 0.0

    x = np.arange(smoothed.size, dtype=np.float64)[valid]
    y = smoothed[valid]
    slope, _ = np.polyfit(x, y, 1)
    max_expected_slope = 30.0 / (window_seconds * sample_rate)
    return float(np.clip(slope / max_expected_slope, -1.0, 1.0))


def component_probabilities(
    time_value: float,
    map_value: float,
    trend_slope: float,
    pre_event_minutes: int,
    mapping_func: MappingFn = concave_mapping,
) -> Dict[str, float]:
    """Compute time, MAP, and trend probability components."""

    if mapping_func is sigmoid_mapping:
        time_prob = mapping_func(time_value, 1, pre_event_minutes, 3, 0.5)
    else:
        time_prob = mapping_func(time_value, 1, pre_event_minutes)

    if map_value < 65:
        map_prob = 1.0
    elif map_value > 105:
        map_prob = 0.0
    else:
        if mapping_func is sigmoid_mapping:
            map_prob = 1.0 - sigmoid_mapping(map_value, 65, 105, 85, 0.2)
        else:
            map_prob = 1.0 - mapping_func(map_value, 65, 105)

    normalized_slope = (trend_slope + 1.0) / 2.0
    if mapping_func is sigmoid_mapping:
        trend_prob = sigmoid_mapping(trend_slope, -1, 1, 0, 2)
    else:
        trend_prob = mapping_func(normalized_slope, 0, 1)
    if trend_slope < 0:
        trend_prob = 1.0 - trend_prob

    return {
        "time": float(np.clip(time_prob, 0.0, 1.0)),
        "map": float(np.clip(map_prob, 0.0, 1.0)),
        "trend": float(np.clip(trend_prob, 0.0, 1.0)),
    }


def combined_probability(
    time_value: float,
    map_value: float,
    trend_slope: float,
    pre_event_minutes: int,
    mapping_func: MappingFn = concave_mapping,
    map_weight: float = 0.5,
    time_weight: float = 0.4,
    trend_weight: float = 0.1,
) -> float:
    """Combine MAP, time, and trend components into one soft target."""

    parts = component_probabilities(time_value, map_value, trend_slope, pre_event_minutes, mapping_func)
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
    mapping_names: tuple[str, ...] = (
        "linear",
        "concave",
        "convex",
        "sigmoid",
        "sine",
        "cosine",
        "reverse_z_131",
        "reverse_z_212",
    ),
) -> list[dict]:
    """Attach soft-label fields to segment dictionaries."""

    converted: list[dict] = []
    for segment in segments:
        map_value = segment.get("map", np.nan)
        if not np.isfinite(map_value):
            continue

        discrete_label = int(segment.get("label", 0))
        if discrete_label <= 0:
            time_value = 1
        elif discrete_label >= pre_event_minutes:
            time_value = pre_event_minutes
        else:
            time_value = discrete_label

        trend_slope = calculate_trend_slope(segment["art"])
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
            )

        converted.append(new_segment)

    return converted
