import pandas as pd
import numpy as np


def calculate_pulse_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate summary metrics for each accepted pulse.

    Only points with status == 3 are included in the calculations.
    Pulses are processed independently by pulse_id.
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
            "increasing_count",
            "decreasing_count",
            "increasing_percent",
            "decreasing_percent",
        ])

    # Only calculate metrics from points that survived
    # the previous filtering stages.
    df = df[df["status"] == 3].copy()

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
            "increasing_count",
            "decreasing_count",
            "increasing_percent",
            "decreasing_percent",
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

    frequency_difference = (
        df_sorted.groupby("pulse_id", sort=True)["freq_hz"]
        .diff()
    )

    increasing = frequency_difference > 0
    decreasing = frequency_difference < 0

    increasing_count = (
        increasing
        .groupby(df_sorted["pulse_id"], sort=True)
        .sum()
    )

    decreasing_count = (
        decreasing
        .groupby(df_sorted["pulse_id"], sort=True)
        .sum()
    )

    pulse_metrics["increasing_count"] = increasing_count
    pulse_metrics["decreasing_count"] = decreasing_count

    denominator = pulse_metrics["n_points"] - 1

    pulse_metrics["increasing_percent"] = np.divide(
        pulse_metrics["increasing_count"],
        denominator,
        out=np.zeros(len(pulse_metrics), dtype=float),
        where=denominator > 0
    )

    pulse_metrics["decreasing_percent"] = np.divide(
        pulse_metrics["decreasing_count"],
        denominator,
        out=np.zeros(len(pulse_metrics), dtype=float),
        where=denominator > 0
    )

    pulse_metrics = pulse_metrics.drop(
        columns=["first_time", "last_time"]
    ).reset_index()

    pulse_metrics = pulse_metrics[
        [
            "pulse_id",
            "n_points",
            "duration_us",
            "bandwidth_hz",
            "fmin",
            "fmax",
            "first_freq_hz",
            "last_freq_hz",
            "increasing_count",
            "decreasing_count",
            "increasing_percent",
            "decreasing_percent",
        ]
    ]

    return pulse_metrics


def filter_pulses(
    df: pd.DataFrame,
    min_bandwidth_hz: float,
    max_bandwidth_hz: float,
    min_duration_us: float,
    max_duration_us: float,
    min_decreasing_percent: float,
    max_increasing_percent: float,
    fmin_ranges: tuple[float, float],
    fmax_ranges: tuple[float, float],
) -> pd.DataFrame:
    """
    Calculate pulse metrics and filter pulses according to specified criteria.

    All input rows are retained. Points belonging to accepted pulses have
    status = 3, while points belonging to rejected pulses have status = 1.
    Rows with no pulse_id are also retained unchanged.
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

    pulse_metrics.loc[
        pulse_metrics["decreasing_percent"] < min_decreasing_percent,
        "accepted"
    ] = False

    pulse_metrics.loc[
        pulse_metrics["increasing_percent"] > max_increasing_percent,
        "accepted"
    ] = False

    accepted_df = df.copy()

    accepted_df.loc[
        accepted_df["pulse_id"].notna(),
        "status"
    ] = 1

    accepted_ids = pulse_metrics.loc[
        pulse_metrics["accepted"],
        "pulse_id"
    ]

    accepted_df.loc[
        accepted_df["pulse_id"].isin(accepted_ids),
        "status"
    ] = 3

    return accepted_df