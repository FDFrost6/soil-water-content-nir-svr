"""
Quality Assurance / Safety Guardrails for NIR Spectral Data and Model Predictions.

This module provides functions to validate spectral inputs and model outputs
before and after inference, ensuring physically plausible results.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Spectral Input Checks
# ---------------------------------------------------------------------------

def check_spectral_outliers(spectrum, threshold=3.0):
    """Detect outlier bands in a single NIR spectrum using the z-score method.

    Parameters
    ----------
    spectrum : array-like
        1-D array of reflectance values across wavelengths.
    threshold : float, optional
        Number of standard deviations from the mean above which a band is
        flagged as an outlier (default 3.0).

    Returns
    -------
    dict with keys:
        ``has_outliers`` (bool) – True when at least one band is flagged.
        ``outlier_indices`` (list[int]) – Zero-based indices of outlier bands.
        ``z_scores`` (np.ndarray) – Absolute z-score for every band.
        ``message`` (str) – Human-readable summary.
    """
    spectrum = np.asarray(spectrum, dtype=float)
    if spectrum.size == 0:
        raise ValueError("spectrum must not be empty")

    mean = np.mean(spectrum)
    std = np.std(spectrum)

    if std == 0:
        z_scores = np.zeros_like(spectrum)
    else:
        z_scores = np.abs((spectrum - mean) / std)

    outlier_indices = list(np.where(z_scores > threshold)[0])
    has_outliers = len(outlier_indices) > 0

    if has_outliers:
        message = (
            f"WARNING: {len(outlier_indices)} outlier band(s) detected "
            f"(z-score > {threshold}) at indices {outlier_indices}."
        )
    else:
        message = "OK: No spectral outliers detected."

    return {
        "has_outliers": has_outliers,
        "outlier_indices": outlier_indices,
        "z_scores": z_scores,
        "message": message,
    }


def check_snr(spectrum, noise_indices=None, signal_indices=None):
    """Estimate the signal-to-noise ratio (SNR) of a NIR spectrum.

    When explicit index ranges are not provided the function uses the first
    10 % of bands as a noise reference and the middle 80 % as the signal
    region – a common heuristic for NIR instruments.

    Parameters
    ----------
    spectrum : array-like
        1-D array of reflectance values.
    noise_indices : array-like of int, optional
        Indices of bands used to estimate noise (standard deviation of
        reflectance in a supposedly flat / featureless region).
    signal_indices : array-like of int, optional
        Indices of bands used to estimate signal (mean reflectance in the
        region of interest).

    Returns
    -------
    dict with keys:
        ``snr`` (float) – Estimated signal-to-noise ratio.
        ``signal_mean`` (float) – Mean reflectance in the signal region.
        ``noise_std`` (float) – Std deviation in the noise region.
        ``is_acceptable`` (bool) – True when SNR >= 10 (commonly used threshold).
        ``message`` (str) – Human-readable summary.
    """
    spectrum = np.asarray(spectrum, dtype=float)
    n = spectrum.size
    if n == 0:
        raise ValueError("spectrum must not be empty")

    if noise_indices is None:
        noise_end = max(1, int(np.ceil(n * 0.10)))
        noise_indices = list(range(noise_end))
    if signal_indices is None:
        sig_start = max(1, int(np.floor(n * 0.10)))
        sig_end = max(sig_start + 1, int(np.ceil(n * 0.90)))
        signal_indices = list(range(sig_start, sig_end))

    noise_region = spectrum[noise_indices]
    signal_region = spectrum[signal_indices]

    noise_std = float(np.std(noise_region))
    signal_mean = float(np.mean(signal_region))

    if noise_std == 0:
        snr = float("inf")
    else:
        snr = signal_mean / noise_std

    is_acceptable = snr >= 10.0

    if is_acceptable:
        message = f"OK: SNR = {snr:.1f} (>= 10 threshold)."
    else:
        message = f"WARNING: Low SNR = {snr:.1f} (< 10 threshold). Consider remeasuring."

    return {
        "snr": snr,
        "signal_mean": signal_mean,
        "noise_std": noise_std,
        "is_acceptable": is_acceptable,
        "message": message,
    }


# ---------------------------------------------------------------------------
# Prediction Output Checks
# ---------------------------------------------------------------------------

def check_prediction_range(prediction, min_val=0.0, max_val=100.0):
    """Check whether a model prediction falls within a physically valid range.

    For soil water content the valid range is 0 – 100 %.  Any value outside
    this window is physically impossible and should be flagged.

    Parameters
    ----------
    prediction : float or array-like
        One or more predicted values (e.g. water content in %).
    min_val : float, optional
        Minimum physically plausible value (default 0.0).
    max_val : float, optional
        Maximum physically plausible value (default 100.0).

    Returns
    -------
    dict with keys:
        ``is_valid`` (bool) – True when **all** predictions are in range.
        ``out_of_range_indices`` (list[int]) – Indices of invalid predictions.
        ``clipped`` (np.ndarray) – Values clipped to [min_val, max_val].
        ``message`` (str) – Human-readable summary.
    """
    prediction = np.atleast_1d(np.asarray(prediction, dtype=float))
    below = np.where(prediction < min_val)[0]
    above = np.where(prediction > max_val)[0]
    out_of_range = sorted(set(below.tolist() + above.tolist()))

    is_valid = len(out_of_range) == 0
    clipped = np.clip(prediction, min_val, max_val)

    if is_valid:
        message = (
            f"OK: All {len(prediction)} prediction(s) are within "
            f"[{min_val}, {max_val}]."
        )
    else:
        message = (
            f"WARNING: {len(out_of_range)} prediction(s) out of physically "
            f"valid range [{min_val}, {max_val}] at indices {out_of_range}."
        )

    return {
        "is_valid": is_valid,
        "out_of_range_indices": out_of_range,
        "clipped": clipped,
        "message": message,
    }
