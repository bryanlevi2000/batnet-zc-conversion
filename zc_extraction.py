# zc_extraction.py
import numpy as np


def compute_hysteresis(samples: np.ndarray, percentile: float = 85.0) -> int:
    """Vectorized hysteresis calculation."""
    if samples.size == 0:
        return 1

    h = int(np.percentile(np.abs(samples), percentile))
    return max(1, min(h, 32767))


def wav_to_zc_times(
    samples: np.ndarray,
    divrat: int,
    hysteresis: int,
    dt: float
) -> np.ndarray:
    """
    Optimized ZC timing. Uses a single loop over the array.
    """
    outbuf = []
    divcounter = 0
    T = 0.0

    # Initialize state based on first sample
    y1 = float(samples[0])

    if y1 <= 0:
        new_threshold = float(hysteresis)
        rising = True
    else:
        new_threshold = float(-hysteresis)
        rising = False

    thresh_val = float(hysteresis)

    for i in range(1, len(samples)):
        y2 = float(samples[i])
        T += dt

        if rising:
            if y2 > new_threshold:
                # Sub-sample interpolation
                tz = (T - dt) + (new_threshold - y1) * dt / (y2 - y1)

                divcounter += 1

                if divcounter >= divrat:
                    outbuf.append(int(tz + 0.5))
                    divcounter = 0

                new_threshold = -thresh_val
                rising = False

        else:
            if y2 < new_threshold:
                tz = (T - dt) + (new_threshold - y1) * dt / (y2 - y1)

                divcounter += 1

                if divcounter >= divrat:
                    outbuf.append(int(tz + 0.5))
                    divcounter = 0

                new_threshold = thresh_val
                rising = True

        y1 = y2

    return np.array(outbuf)


def estimate_segment_mean_abs(
    samples: np.ndarray,
    start_time_us: float,
    end_time_us: float,
    sample_period_us: float
) -> float:
    """Mean absolute waveform amplitude across a time interval."""

    if samples.size == 0:
        return 0.0

    if not np.isfinite(start_time_us) or not np.isfinite(end_time_us):
        return np.nan

    start = min(float(start_time_us), float(end_time_us))
    end = max(float(start_time_us), float(end_time_us))

    if end <= start:
        return np.nan

    # Clamp to valid sample window
    start = max(0.0, start)
    end = min(float(samples.size) * sample_period_us, end)

    if end <= start:
        return np.nan

    start_index = start / sample_period_us
    end_index = end / sample_period_us

    left = int(np.floor(start_index))
    right = int(np.ceil(end_index)) - 1

    left = max(0, left)
    right = min(samples.size - 1, right)

    if left > right:
        center_index = int(
            np.clip(
                round((start_index + end_index) / 2.0),
                0,
                samples.size - 1
            )
        )
        return float(np.abs(samples[center_index]))

    total = 0.0
    total_weight = 0.0

    for idx in range(left, right + 1):
        seg_start = idx * sample_period_us
        seg_end = seg_start + sample_period_us

        overlap_start = max(start, seg_start)
        overlap_end = min(end, seg_end)

        if overlap_end <= overlap_start:
            continue

        weight = (overlap_end - overlap_start) / sample_period_us

        total += abs(float(samples[idx])) * weight
        total_weight += weight

    if total_weight <= 0.0:
        center_index = int(
            np.clip(
                round((start_index + end_index) / 2.0),
                0,
                samples.size - 1
            )
        )
        return float(np.abs(samples[center_index]))

    return total / total_weight


def compute_time_frequency_and_amplitude(
    samples: np.ndarray,
    times_us: np.ndarray,
    divrat: int,
    sample_period_us: float
) -> np.ndarray:
    """
    Return columns:

        time_us
        freq_hz
        amplitude
    """

    if len(times_us) < 3:
        return np.empty((0, 3))

    dt_us = times_us[2:] - times_us[:-2]

    valid = dt_us > 0

    freqs = np.zeros_like(dt_us, dtype=float)
    freqs[valid] = (divrat * 1e6) / dt_us[valid]

    amplitudes = np.full(dt_us.shape, np.nan, dtype=float)

    if np.any(valid):
        valid_indices = np.nonzero(valid)[0]

        start_times = times_us[:-2][valid]
        end_times = times_us[2:][valid]

        for idx, (start, end) in zip(
            valid_indices,
            zip(start_times, end_times)
        ):
            amplitudes[idx] = estimate_segment_mean_abs(
                samples,
                start,
                end,
                sample_period_us
            )

    return np.column_stack(
        (times_us[2:], freqs, amplitudes)
    )


def extract_zc(
    samples: np.ndarray,
    sample_rate: int,
    divrat: int = 8,
    hysteresis_percentile: float = 85.0
) -> np.ndarray:
    """
    Extract zero-crossing time, frequency, and amplitude data
    from an audio signal.

    Parameters
    ----------
    samples : np.ndarray
        Audio waveform.
    sample_rate : int
        Audio sample rate in Hz.
    divrat : int
        Zero-crossing division ratio.
    hysteresis_percentile : float
        Percentile used to calculate hysteresis.

    Returns
    -------
    np.ndarray
        Array with columns:

        [time_us, freq_hz, amplitude]
    """

    if samples.size == 0:
        return np.empty((0, 3))

    hysteresis = compute_hysteresis(
        samples,
        hysteresis_percentile
    )

    sample_period_us = 1e6 / sample_rate

    # Calculate zero-crossing times
    times_us = wav_to_zc_times(
        samples,
        divrat,
        hysteresis,
        sample_period_us
    )

    # Calculate frequency and amplitude
    tf_data = compute_time_frequency_and_amplitude(
        samples,
        times_us,
        divrat,
        sample_period_us
    )

    return tf_data
