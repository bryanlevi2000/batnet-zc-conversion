# pulse_filtering.py
import pandas as pd


def calculate_pulse_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate summary metrics for each pulse.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing time_us, freq_hz, and pulse_id.

    Returns
    -------
    pandas.DataFrame
        One row per pulse containing pulse metrics.
    """

    pulse_metrics = []

    for pulse_id, pulse in df.groupby("pulse_id", sort=True):

        pulse = pulse.sort_values("time_us").reset_index(drop=True)

        n_rows = len(pulse)

        first_time = pulse["time_us"].iloc[0]
        last_time = pulse["time_us"].iloc[-1]

        first_frequency = pulse["freq_hz"].iloc[0]
        last_frequency = pulse["freq_hz"].iloc[-1]

        duration = last_time - first_time

        bandwidth = (
            pulse["freq_hz"].max()
            - pulse["freq_hz"].min()
        )

        # Frequency differences between consecutive points
        frequency_difference = pulse["freq_hz"].diff()

        increasing_count = (frequency_difference > 0).sum()
        decreasing_count = (frequency_difference < 0).sum()

        # There is one fewer comparison than there are rows
        denominator = n_rows - 1

        if denominator > 0:
            increasing_percent = increasing_count / denominator
            decreasing_percent = decreasing_count / denominator
        else:
            increasing_percent = 0.0
            decreasing_percent = 0.0

        pulse_metrics.append({
            "pulse_id": pulse_id,
            "n_points": n_rows,
            "duration_us": duration,
            "bandwidth_hz": bandwidth,
            "first_freq_hz": first_frequency,
            "last_freq_hz": last_frequency,
            "increasing_count": increasing_count,
            "decreasing_count": decreasing_count,
            "increasing_percent": increasing_percent,
            "decreasing_percent": decreasing_percent,
        })

    return pd.DataFrame(pulse_metrics)


def filter_pulses(
    df: pd.DataFrame,
    min_bandwidth_hz: float,
    max_bandwidth_hz: float,
    min_duration_us: float,
    max_duration_us: float,
    min_decreasing_percent: float,
    max_increasing_percent: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate pulse metrics and filter pulses according to specified criteria.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing time_us, freq_hz, and pulse_id.

    min_bandwidth_hz : float
        Minimum allowed pulse bandwidth in Hz.

    max_bandwidth_hz : float
        Maximum allowed pulse bandwidth in Hz.

    min_duration_us : float
        Minimum allowed pulse duration in microseconds.

    max_duration_us : float
        Maximum allowed pulse duration in microseconds.

    min_decreasing_percent : float
        Minimum required proportion of decreasing frequency transitions.
        Expressed as a decimal. For example, 0.80 = 80%.

    max_increasing_percent : float
        Maximum allowed proportion of increasing frequency transitions.
        Expressed as a decimal. For example, 0.20 = 20%.

    Returns
    -------
    accepted_df : pandas.DataFrame
        Original point-level dataframe containing only accepted pulses.

    pulse_metrics : pandas.DataFrame
        One row per pulse containing calculated metrics and acceptance status.
    """

    pulse_metrics = calculate_pulse_metrics(df)

    # Start with every pulse accepted
    pulse_metrics["accepted"] = True

    # Bandwidth restriction
    pulse_metrics.loc[
        (pulse_metrics["bandwidth_hz"] < min_bandwidth_hz)
        | (pulse_metrics["bandwidth_hz"] > max_bandwidth_hz),
        "accepted"
    ] = False

    # Duration restriction
    pulse_metrics.loc[
        (pulse_metrics["duration_us"] < min_duration_us)
        | (pulse_metrics["duration_us"] > max_duration_us),
        "accepted"
    ] = False

    # Reject pulses whose final frequency is higher than their initial frequency
    pulse_metrics.loc[
        pulse_metrics["last_freq_hz"] > pulse_metrics["first_freq_hz"],
        "accepted"
    ] = False

    # Minimum percentage of decreasing frequency transitions
    pulse_metrics.loc[
        pulse_metrics["decreasing_percent"] < min_decreasing_percent,
        "accepted"
    ] = False

    # Maximum percentage of increasing frequency transitions
    pulse_metrics.loc[
        pulse_metrics["increasing_percent"] > max_increasing_percent,
        "accepted"
    ] = False

    # Get IDs of accepted pulses
    accepted_ids = pulse_metrics.loc[
        pulse_metrics["accepted"],
        "pulse_id"
    ]

    # Keep only accepted pulses in the original point-level dataframe
    accepted_df = df[
        df["pulse_id"].isin(accepted_ids)
    ].copy()

    return accepted_df, pulse_metrics
