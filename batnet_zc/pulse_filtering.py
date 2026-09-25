import numpy as np
import pandas as pd


def find_characteristic_frequency(
    pulse_df: pd.DataFrame,
    min_points: int,
    last_fraction: float,
):
    """
    Find the characteristic-frequency point in one pulse.

    The algorithm is:
    1. Sort points by time.
    2. Calculate the actual pulse start and end times.
    3. Keep only the last fraction of the pulse by time.
    4. Calculate slopes between adjacent points.
    5. Ignore zero-frequency-change and positive-slope segments.
    6. Select the least-negative slope.
    7. Return the original dataframe index of the later endpoint.
    """

    pulse_df = pulse_df.sort_values("time_us")

    start_time = pulse_df["time_us"].iloc[0]
    end_time = pulse_df["time_us"].iloc[-1]
    pulse_duration = end_time - start_time

    if pulse_duration <= 0:
        return None

    cutoff_time = (
        start_time
        + (1 - last_fraction) * pulse_duration
    )

    pulse_df = pulse_df[
        pulse_df["time_us"] >= cutoff_time
    ]

    if len(pulse_df) < min_points:
        return None

    x = pulse_df["time_us"].to_numpy(dtype=float)
    y = pulse_df["freq_hz"].to_numpy(dtype=float)
    original_indices = pulse_df.index.to_numpy()

    delta_time = np.diff(x)
    delta_frequency = np.diff(y)

    slopes = delta_frequency / delta_time

    valid_segments = (
        (delta_frequency != 0)
        & (delta_time != 0)
        & (slopes < 0)
    )

    if not np.any(valid_segments):
        return None

    valid_slopes = slopes[valid_segments]
    segment_positions = np.flatnonzero(valid_segments)

    least_negative_position = np.argmax(valid_slopes)

    segment_start = segment_positions[least_negative_position]
    segment_end = segment_start + 1

    return original_indices[segment_end]


def apply_frequency_decrease_cutoff(
    pulse_df: pd.DataFrame,
    characteristic_index,
    cutoff_hz: float,
) -> pd.DataFrame:
    """
    Starting after the characteristic-frequency point, find the first
    adjacent frequency decrease greater than cutoff_hz.

    The point immediately before the excessive decrease is retained.
    The point causing the excessive decrease and all points after it
    are assigned status 1.
    """

    if characteristic_index is None:
        return pulse_df

    pulse_df = pulse_df.sort_values("time_us").copy()

    indices = pulse_df.index.to_numpy()

    characteristic_position = np.flatnonzero(
        indices == characteristic_index
    )

    if len(characteristic_position) == 0:
        return pulse_df

    characteristic_position = characteristic_position[0]

    frequencies = pulse_df["freq_hz"].to_numpy(dtype=float)

    frequency_changes = np.diff(frequencies)

    changes_after_characteristic = frequency_changes[
        characteristic_position:
    ]

    excessive = changes_after_characteristic < -cutoff_hz

    if not np.any(excessive):
        return pulse_df

    first_excessive = np.flatnonzero(excessive)[0]

    cutoff_position = (
        characteristic_position
        + first_excessive
        + 1
    )

    pulse_df.loc[
        indices[cutoff_position:],
        "status"
    ] = 1

    return pulse_df


def calculate_pulse_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate summary metrics for each pulse.

    Only points with status > 1 are included.
    """

    if df.empty:
        return pd.DataFrame(columns=[
            "pulse_id",
            "n_points",
            "duration_us",
            "bandwidth_hz",
            "fmin",
            "fmax",
            "first_freq_hz",
            "last_freq_hz",
        ])

    df = df[df["status"] > 1].copy()

    if df.empty:
        return pd.DataFrame(columns=[
            "pulse_id",
            "n_points",
            "duration_us",
            "bandwidth_hz",
            "fmin",
            "fmax",
            "first_freq_hz",
            "last_freq_hz",
        ])

    df_sorted = df.sort_values(
        ["pulse_id", "time_us"],
        kind="stable"
    )

    grouped = df_sorted.groupby(
        "pulse_id",
        sort=True
    )

    pulse_metrics = grouped.agg(
        n_points=("freq_hz", "size"),
        first_time=("time_us", "first"),
        last_time=("time_us", "last"),
        fmin=("freq_hz", "min"),
        fmax=("freq_hz", "max"),
        first_freq_hz=("freq_hz", "first"),
        last_freq_hz=("freq_hz", "last"),
    )

    pulse_metrics["duration_us"] = (
        pulse_metrics["last_time"]
        - pulse_metrics["first_time"]
    )

    pulse_metrics["bandwidth_hz"] = (
        pulse_metrics["fmax"]
        - pulse_metrics["fmin"]
    )

    pulse_metrics = pulse_metrics.drop(
        columns=["first_time", "last_time"]
    ).reset_index()

    return pulse_metrics[
        [
            "pulse_id",
            "n_points",
            "duration_us",
            "bandwidth_hz",
            "fmin",
            "fmax",
            "first_freq_hz",
            "last_freq_hz",
        ]
    ]


def filter_pulses(
    df: pd.DataFrame,
    min_bandwidth_hz: float,
    max_bandwidth_hz: float,
    min_duration_us: float,
    max_duration_us: float,
    fmin_ranges: tuple[float, float],
    fmax_ranges: tuple[float, float],
) -> pd.DataFrame:
    """
    Calculate pulse metrics and filter pulses.

    Accepted pulse points receive status 2.
    Rejected pulse points receive status 1.
    Characteristic-frequency points retain status 3
    within accepted pulses.
    """

    pulse_metrics = calculate_pulse_metrics(df)

    min_fmin_hz, max_fmin_hz = fmin_ranges
    min_fmax_hz, max_fmax_hz = fmax_ranges

    pulse_metrics["accepted"] = True

    pulse_metrics.loc[
        (pulse_metrics["bandwidth_hz"] < min_bandwidth_hz)
        | (pulse_metrics["bandwidth_hz"] > max_bandwidth_hz),
        "accepted"
    ] = False

    pulse_metrics.loc[
        (pulse_metrics["duration_us"] < min_duration_us)
        | (pulse_metrics["duration_us"] > max_duration_us),
        "accepted"
    ] = False

    pulse_metrics.loc[
        (pulse_metrics["fmin"] < min_fmin_hz)
        | (pulse_metrics["fmin"] > max_fmin_hz),
        "accepted"
    ] = False

    pulse_metrics.loc[
        (pulse_metrics["fmax"] < min_fmax_hz)
        | (pulse_metrics["fmax"] > max_fmax_hz),
        "accepted"
    ] = False

    pulse_metrics.loc[
        pulse_metrics["last_freq_hz"] > pulse_metrics["first_freq_hz"],
        "accepted"
    ] = False

    result = df.copy()

    # Remember characteristic-frequency points before resetting statuses.
    characteristic_indices = result.index[
        result["status"] == 3
    ]

    # Reset all pulse points to rejected.
    result.loc[
        result["pulse_id"].notna(),
        "status"
    ] = 1

    # Assign status 2 to all points belonging to accepted pulses.
    accepted_ids = pulse_metrics.loc[
        pulse_metrics["accepted"],
        "pulse_id"
    ]

    result.loc[
        result["pulse_id"].isin(accepted_ids),
        "status"
    ] = 2

    # Restore status 3 for characteristic-frequency points
    # that belong to accepted pulses.
    result.loc[
        result.index.isin(characteristic_indices)
        & result["pulse_id"].isin(accepted_ids),
        "status"
    ] = 3

    return result
