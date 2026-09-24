"""Utilities for the soft probability-label framework."""

from __future__ import annotations

import numpy as np


def saturate_01(x):
    return np.clip(x, 0.0, 1.0)


def concave_mapping(x):
    """Concave mapping used in the manuscript: g(x)=2x-x^2."""

    x = saturate_01(np.asarray(x, dtype=np.float32))
    return 2.0 * x - x**2


def combine_probability_components(p_map, p_time, p_trend, weights=(0.5, 0.4, 0.1)):
    """Combine MAP, time, and trend components.

    Parameters
    ----------
    p_map, p_time, p_trend:
        Component probabilities in [0, 1].
    weights:
        Weights in manuscript order: MAP:time:trend. The default is 5:4:1.
    """

    w_map, w_time, w_trend = weights
    p = w_map * np.asarray(p_map) + w_time * np.asarray(p_time) + w_trend * np.asarray(p_trend)
    return saturate_01(p)
