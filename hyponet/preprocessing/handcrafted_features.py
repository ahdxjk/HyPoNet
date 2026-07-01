"""Offline handcrafted feature extraction for HyPo-Net.

The manuscript model uses 81 handcrafted features per 60-second window. They
are extracted from ABP/ART, ECG, PLETH/PPG, and static patient information,
then saved as ``extracted_features.h5`` with dataset key ``features``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler


FEATURE_DIM = 81


def _safe_mean(values: Iterable[float], default: float = 0.0) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0 or not np.isfinite(arr).any():
        return default
    return float(np.nanmean(arr))


def compute_bsa(height_cm: float, weight_kg: float) -> float:
    if height_cm <= 0 or weight_kg <= 0:
        return 0.0
    return float(np.sqrt((height_cm * weight_kg) / 3600.0))


def calculate_hemodynamic_properties(gender: str, age: float, pe_values: Iterable[float]) -> tuple[list[float], list[float]]:
    if gender.lower() == "female":
        a_max = 4.12
        p0 = 72 - 0.89 * age
        p1 = 57 - 0.44 * age
    else:
        a_max = 5.62
        p0 = 76 - 0.89 * age
        p1 = 57 - 0.44 * age

    if abs(p1) < 1e-8:
        p1 = 1e-8

    a_pe, c_pe = [], []
    for pe in pe_values:
        a_value = a_max * (0.5 + math.atan((pe - p0) / p1) / math.pi)
        c_value = (a_max / (math.pi * p1)) / (1.0 + ((pe - p0) / p1) ** 2)
        a_pe.append(a_value)
        c_pe.append(c_value)
    return a_pe, c_pe


def compute_average_psa(smoothed_wave: np.ndarray, dbp_indices: Iterable[int], notch_points: Iterable[int]) -> tuple[list[float], list[float]]:
    psa_values, pe_values = [], []
    for notch, dbp in zip(notch_points, dbp_indices):
        if dbp >= notch:
            continue
        segment = smoothed_wave[dbp : notch + 1]
        if len(segment) == 0:
            continue
        duration = len(segment) / 100.0
        time_vector = np.linspace(0, duration, len(segment))
        psa_values.append(float(np.trapz(segment, time_vector)))
        pe_values.append(float(np.sqrt(np.sum(segment**2) / len(segment))))
    return psa_values, pe_values


def calculate_bp_map(waveform: np.ndarray, notch_points: Iterable[int], dbp_indices: Iterable[int], sbp_indices: Iterable[int]) -> list[float]:
    maps = []
    for notch_idx, dbp_idx in zip(notch_points, dbp_indices):
        candidates = [sbp for sbp in sbp_indices if dbp_idx < sbp < notch_idx]
        if not candidates:
            continue
        sbp = waveform[candidates[0]]
        dbp = waveform[notch_idx]
        maps.append(float((sbp + 2.0 * dbp) / 3.0))
    return maps


def calculate_aortic_impedance(
    bsa: float,
    bp_map: list[float],
    bp_pulse: float,
    psa_values: list[float],
    pe_values: list[float],
    cw_pe_values: list[float],
    a_pe_values: list[float],
    c_pe_values: list[float],
) -> np.ndarray:
    zc_results, zin_results, sv_results, co_results = [], [], [], []
    svr_results, svi_results, ci_results, svri_results, rp_results = [], [], [], [], []
    current_rp = 1.0

    for psa, pe, cw_pe, bp, a_pe, c_pe in zip(psa_values, pe_values, cw_pe_values, bp_map, a_pe_values, c_pe_values):
        if a_pe <= 0 or c_pe == 0 or bsa <= 0:
            continue
        zc = math.sqrt(abs((1.055 / 1333.0) / (a_pe * c_pe)))
        f_heart = bp_pulse / 60.0
        denom = 1.0 + (2.0 * math.pi * f_heart * current_rp * cw_pe) ** 2
        zin_re = zc + current_rp / denom
        zin_im = -2.0 * math.pi * f_heart * current_rp**2 * cw_pe / denom
        zin = math.sqrt(zin_re**2 + zin_im**2)
        if zin <= 0:
            continue

        sv = np.clip(psa / zin, 1, 500)
        co = np.clip(bp_pulse * sv / 1000.0, 0.1, 99)
        svr = np.clip(80.0 * (bp - 7.0) / co, 10, 9999)
        svi = sv / bsa
        ci = co / bsa
        svri = svr * bsa
        current_rp = np.clip(0.06 * (bp - 7.0) / co, 0.1, 7.5)
        if svr > 9000 or svr < 100:
            current_rp = 1.0

        zc_results.append(zc)
        zin_results.append(zin)
        sv_results.append(sv)
        co_results.append(co)
        svr_results.append(svr)
        svi_results.append(svi)
        ci_results.append(ci)
        svri_results.append(svri)
        rp_results.append(current_rp)

    return np.array(
        [
            _safe_mean(zc_results),
            _safe_mean(zin_results),
            _safe_mean(sv_results),
            _safe_mean(co_results),
            _safe_mean(svr_results),
            _safe_mean(svi_results),
            _safe_mean(ci_results),
            _safe_mean(svri_results),
            _safe_mean(rp_results),
        ],
        dtype=np.float32,
    )


def filter_notch_and_dbp_points(notch_points: list[int], dbp_indices: np.ndarray, sampling_rate: int = 100) -> tuple[list[int], list[int]]:
    if len(dbp_indices) < 2 or not notch_points:
        return [], []

    distances = np.diff(dbp_indices)
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    min_distance = max(mean_distance - 1.2 * std_distance, sampling_rate * 0.2)
    max_distance = mean_distance + 1.2 * std_distance

    final_dbp, final_notch = [], []
    notch_idx = 0
    for i in range(len(dbp_indices) - 1):
        left = dbp_indices[i]
        right = dbp_indices[i + 1]
        distance = right - left
        if distance < min_distance or distance > max_distance:
            continue
        while notch_idx < len(notch_points) and notch_points[notch_idx] <= right:
            if left < notch_points[notch_idx] < right:
                final_dbp.append(int(left))
                final_notch.append(int(notch_points[notch_idx]))
                break
            notch_idx += 1
    return final_dbp, final_notch


def compute_dpdt(abp_wave: np.ndarray, sampling_rate: int = 100) -> np.ndarray:
    return np.gradient(abp_wave) / (1.0 / sampling_rate)


def compute_hemodynamic_features(abp_wave: np.ndarray, patient_data: np.ndarray, sampling_rate: int = 100) -> np.ndarray:
    """Return 22 hemodynamic/static ABP-derived features."""

    try:
        if len(abp_wave) < sampling_rate or not np.isfinite(abp_wave).any():
            return np.zeros(22, dtype=np.float32)

        wave = np.asarray(abp_wave, dtype=np.float64)
        finite = np.isfinite(wave)
        if not finite.all():
            x = np.arange(len(wave))
            wave = np.interp(x, x[finite], wave[finite])

        b, a = butter(3, 10 / (sampling_rate / 2.0), btype="lowpass")
        smoothed = filtfilt(b, a, wave)

        peak_distance = sampling_rate // 2
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
        if len(sbp_indices) == 0 or len(dbp_indices) < 3:
            return np.zeros(22, dtype=np.float32)

        dbp_diffs = np.diff(dbp_indices)
        lower = np.mean(dbp_diffs) - 1.8 * np.std(dbp_diffs)
        upper = np.mean(dbp_diffs) + 1.8 * np.std(dbp_diffs)
        valid_dbp = []
        notch_points = []
        for i in range(len(dbp_indices)):
            if i == 0 or i == len(dbp_indices) - 1:
                valid_dbp.append(int(dbp_indices[i]))
                continue
            prev_diff = dbp_indices[i] - dbp_indices[i - 1]
            next_diff = dbp_indices[i + 1] - dbp_indices[i]
            if lower <= prev_diff <= upper or lower <= next_diff <= upper:
                valid_dbp.append(int(dbp_indices[i]))

            start = int(dbp_indices[i] + 0.3 * next_diff)
            end = int(dbp_indices[i] + 0.65 * next_diff)
            if end < len(smoothed) and end > start + 2:
                segment = smoothed[start:end]
                second_diff = np.diff(segment, n=2)
                if len(second_diff):
                    second_diff = np.insert(second_diff, 0, second_diff[0])
                    second_diff = np.insert(second_diff, 0, second_diff[0])
                    notch_points.append(int(start + np.argmax(second_diff)))

        valid_dbp, notch_points = filter_notch_and_dbp_points(notch_points, np.asarray(valid_dbp), sampling_rate)
        if not valid_dbp or not notch_points:
            return np.zeros(22, dtype=np.float32)

        age = float(patient_data[0])
        sex_code = int(patient_data[1])
        height = float(patient_data[2])
        weight = float(patient_data[3])
        sex = "male" if sex_code == 0 else "female"
        bsa = compute_bsa(height, weight)

        bp_map = calculate_bp_map(smoothed, notch_points, valid_dbp, sbp_indices)
        psa_values, pe_values = compute_average_psa(smoothed, valid_dbp, notch_points)
        if not bp_map or not psa_values or not pe_values:
            return np.zeros(22, dtype=np.float32)

        a_pe, c_pe = calculate_hemodynamic_properties(sex, age, pe_values)
        cw_pe = [0.5 * height * c for c in c_pe]
        impedance = calculate_aortic_impedance(bsa, bp_map, len(sbp_indices), psa_values, pe_values, cw_pe, a_pe, c_pe)
        summary = np.array(
            [
                _safe_mean(bp_map),
                _safe_mean(psa_values),
                _safe_mean(pe_values),
                _safe_mean(a_pe),
                _safe_mean(c_pe),
                _safe_mean(cw_pe),
                bsa,
                float(len(sbp_indices)),
            ],
            dtype=np.float32,
        )
        patient = np.asarray(patient_data[:5], dtype=np.float32)
        return np.concatenate([impedance, patient, summary]).astype(np.float32)
    except Exception:
        return np.zeros(22, dtype=np.float32)


def calculate_frequency_features(signals: np.ndarray, fs: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lf_hf_values, ratio_values, total_values = [], [], []
    for signal in signals:
        freqs, psd = welch(signal, fs=fs, nperseg=min(2000, len(signal)))
        lf = (freqs >= 0.04) & (freqs < 0.15)
        hf = (freqs >= 0.15) & (freqs < 0.4)
        lf_power = np.trapz(psd[lf], freqs[lf]) if lf.any() else 0.0
        hf_power = np.trapz(psd[hf], freqs[hf]) if hf.any() else 0.0
        total_power = np.trapz(psd, freqs) if len(freqs) else 0.0
        lf_hf_values.append(lf_power + hf_power)
        ratio_values.append(lf_power / (hf_power + 1e-10))
        total_values.append(total_power)
    return np.asarray(lf_hf_values), np.asarray(ratio_values), np.asarray(total_values)


def calculate_pat(ecg_signal: np.ndarray, ppg_signal: np.ndarray, fs: int = 100) -> np.ndarray:
    values = []
    for ecg, ppg in zip(ecg_signal, ppg_signal):
        r_peaks, _ = find_peaks(ecg, distance=int(0.2 * fs))
        ppg_peaks, _ = find_peaks(ppg, distance=int(0.2 * fs))
        intervals = []
        for r_peak in r_peaks:
            after = ppg_peaks[ppg_peaks > r_peak]
            if len(after):
                intervals.append((after[0] - r_peak) / fs)
        values.append(_safe_mean(intervals, np.nan))
    return np.asarray(values, dtype=np.float32)


def calculate_ptt(ppg_signal: np.ndarray, abp_signal: np.ndarray, fs: int = 100) -> np.ndarray:
    values = []
    for ppg, abp in zip(ppg_signal, abp_signal):
        ppg_peaks, _ = find_peaks(ppg, distance=int(0.2 * fs))
        abp_upstroke, _ = find_peaks(np.gradient(abp), distance=int(0.2 * fs))
        intervals = []
        for ppg_peak in ppg_peaks:
            after = abp_upstroke[abp_upstroke > ppg_peak]
            if len(after):
                intervals.append((after[0] - ppg_peak) / fs)
        values.append(_safe_mean(intervals, np.nan))
    return np.asarray(values, dtype=np.float32)


def _statistical_features(data: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = np.mean(data, axis=1)
    std = np.std(data, axis=1)
    min_value = np.min(data, axis=1)
    max_value = np.max(data, axis=1)
    skewness = skew(data, axis=1)
    kurt = kurtosis(data, axis=1)
    value_range = max_value - min_value
    median = np.median(data, axis=1)
    q1 = np.percentile(data, 25, axis=1)
    q3 = np.percentile(data, 75, axis=1)
    iqr = q3 - q1
    cv = std / (mean + 1e-10)
    return mean, std, min_value, max_value, skewness, kurt, value_range, median, q1, q3, iqr, cv


class MLFeatureExtractor:
    """Extract the 81-dimensional handcrafted feature vector."""

    def extract_abp_features(self, signals: np.ndarray, static: np.ndarray) -> np.ndarray:
        abp = signals[:, 0, :]
        ecg = signals[:, 1, :]
        ppg = signals[:, 2, :]
        pat = calculate_pat(ecg, ppg)
        ptt = calculate_ptt(ppg, abp)
        lf_hf, lf_hf_ratio, power = calculate_frequency_features(abp)
        hemo = np.vstack([compute_hemodynamic_features(wave, patient) for wave, patient in zip(abp, static)])

        mean, std, min_value, max_value, skewness, kurt, value_range, median, q1, q3, iqr, _ = _statistical_features(abp)
        dpdt = np.asarray([compute_dpdt(wave) for wave in abp])
        dpdt_mean = np.mean(dpdt, axis=1)
        dpdt_std = np.std(dpdt, axis=1)
        dpdt_min = np.min(dpdt, axis=1)
        dpdt_max = np.max(dpdt, axis=1)
        dpdt_skew = skew(dpdt, axis=1)
        dpdt_kurt = kurtosis(dpdt, axis=1)
        dpdt_range = dpdt_max - dpdt_min

        abp_basic = np.vstack(
            [
                mean,
                std,
                min_value,
                max_value,
                skewness,
                kurt,
                pat,
                ptt,
                value_range,
                median,
                q1,
                q3,
                iqr,
                dpdt_mean,
                dpdt_std,
                dpdt_min,
                dpdt_max,
                dpdt_skew,
                dpdt_kurt,
                dpdt_range,
                lf_hf,
                lf_hf_ratio,
                power,
            ]
        ).T
        return np.hstack([abp_basic, hemo])

    def extract_ecg_features(self, signals: np.ndarray, fs: int = 100) -> np.ndarray:
        ecg = signals[:, 1, :]
        lf_hf, lf_hf_ratio, power = calculate_frequency_features(ecg)
        mean, std, min_value, max_value, skewness, kurt, value_range, median, q1, q3, iqr, cv = _statistical_features(ecg)

        rr_values, sdnn_values, rmssd_values, ibi_values = [], [], [], []
        for signal in ecg:
            peaks, _ = find_peaks(signal, distance=int(0.2 * fs))
            if len(peaks) < 2:
                rr_values.append(0.0)
                sdnn_values.append(0.0)
                rmssd_values.append(0.0)
                ibi_values.append(0.0)
            else:
                rr = np.diff(peaks) / fs
                rr_values.append(float(np.mean(rr)))
                sdnn_values.append(float(np.std(rr)))
                rmssd_values.append(float(np.sqrt(np.mean(np.diff(rr) ** 2))) if len(rr) > 1 else 0.0)
                ibi_values.append(float(np.mean(rr)))

        return np.vstack(
            [
                mean,
                std,
                min_value,
                max_value,
                skewness,
                kurt,
                value_range,
                median,
                q1,
                q3,
                iqr,
                cv,
                np.asarray(rr_values),
                np.asarray(sdnn_values),
                np.asarray(rmssd_values),
                np.asarray(ibi_values),
                lf_hf,
                lf_hf_ratio,
                power,
            ]
        ).T

    def extract_ppg_features(self, signals: np.ndarray) -> np.ndarray:
        ppg = signals[:, 2, :]
        lf_hf, lf_hf_ratio, power = calculate_frequency_features(ppg)
        mean, std, min_value, max_value, skewness, kurt, value_range, median, q1, q3, iqr, cv = _statistical_features(ppg)
        pulse_amplitude = max_value - min_value
        signal_power = np.mean(ppg**2, axis=1)
        noise_power = np.std(ppg, axis=1) ** 2
        snr = signal_power / (noise_power + 1e-10)
        return np.vstack(
            [
                mean,
                std,
                min_value,
                max_value,
                skewness,
                kurt,
                value_range,
                median,
                q1,
                q3,
                iqr,
                cv,
                pulse_amplitude,
                snr,
                lf_hf,
                lf_hf_ratio,
                power,
            ]
        ).T

    def handle_nan_inf(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        finite = np.isfinite(features)
        if finite.all():
            return features
        column_means = np.nanmean(np.where(finite, features, np.nan), axis=0)
        column_means = np.where(np.isfinite(column_means), column_means, 0.0)
        rows, cols = np.where(~finite)
        features[rows, cols] = column_means[cols]
        return features

    def extract_features(self, signals: np.ndarray | torch.Tensor, static: np.ndarray | torch.Tensor, standardize: bool = True) -> np.ndarray:
        """Extract features from signals with shape [N, 3, 6000] or [N, 6000, 3]."""

        if isinstance(signals, torch.Tensor):
            signals = signals.detach().cpu().numpy()
        if isinstance(static, torch.Tensor):
            static = static.detach().cpu().numpy()

        signals = np.asarray(signals, dtype=np.float32)
        static = np.asarray(static, dtype=np.float32)
        if signals.ndim != 3:
            raise ValueError(f"signals must be 3D, got shape {signals.shape}")
        if signals.shape[1] != 3 and signals.shape[2] == 3:
            signals = np.transpose(signals, (0, 2, 1))
        if signals.shape[1] != 3:
            raise ValueError(f"signals must contain 3 channels, got shape {signals.shape}")

        features = np.hstack(
            [
                self.extract_abp_features(signals, static),
                self.extract_ecg_features(signals),
                self.extract_ppg_features(signals),
            ]
        )
        features = self.handle_nan_inf(features)
        if features.shape[1] != FEATURE_DIM:
            raise RuntimeError(f"Expected {FEATURE_DIM} handcrafted features, got {features.shape[1]}")
        if standardize:
            features = StandardScaler().fit_transform(features)
        return features.astype(np.float32)


def extract_features_for_split(split_dir: Path, static_data: np.ndarray, output_name: str = "extracted_features.h5") -> Path:
    """Read one split directory and save ``extracted_features.h5``."""

    with h5py.File(split_dir / "ART.h5", "r") as h5f:
        art = h5f["ART"][:]
    with h5py.File(split_dir / "ECG.h5", "r") as h5f:
        ecg = h5f["ECG"][:]
    with h5py.File(split_dir / "PLETH.h5", "r") as h5f:
        pleth = h5f["PLETH"][:]
    with h5py.File(split_dir / "STATIC.h5", "r") as h5f:
        static_indices = h5f["STATIC"][:].astype(int).squeeze()

    selected_static = static_data[static_indices - 1]
    signals = np.stack([art, ecg, pleth], axis=1)
    features = MLFeatureExtractor().extract_features(signals, selected_static, standardize=True)

    output_path = split_dir / output_name
    with h5py.File(output_path, "w") as h5f:
        h5f.create_dataset("features", data=features, dtype=features.dtype)
    return output_path


def extract_features_for_folds(data_root: Path, static_data_path: Path, folds: Iterable[int] = (1, 2, 3, 4, 5)) -> None:
    static_data = np.load(static_data_path, allow_pickle=True)
    for fold in folds:
        for split in ("train", "val", "test"):
            split_dir = data_root / f"fold{fold}" / split
            if split_dir.exists():
                output_path = extract_features_for_split(split_dir, static_data)
                print(f"Saved {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract 81-dimensional handcrafted features for HyPo-Net.")
    parser.add_argument("--data-root", required=True, type=Path, help="Root containing fold1...fold5 directories.")
    parser.add_argument("--static-data", required=True, type=Path, help="NumPy file with static clinical data indexed by case ID - 1.")
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    extract_features_for_folds(args.data_root, args.static_data, args.folds)


if __name__ == "__main__":
    main()
