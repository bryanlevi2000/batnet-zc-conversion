# pulse_frequency_filtering.py

import numpy as np
import pandas as pd

def filter_pulse_frequency_gaps(
    df: pd.DataFrame,
    frequency_percentile: float = 0.10,
    upper_decreasing_cutoff_hz: float = 5_000,
    lower_decreasing_cutoff_hz: float = 2_000,
) -> pd.DataFrame:
    """
    Filter pulse points based on decreasing frequency gaps.

    All input rows are retained. Points that fail the frequency-gap
    criterion are reassigned status = 1. Points that pass remain
    unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least:
            - pulse_id
            - freq_hz
            - status

    frequency_percentile : float, default=0.10
        Percentile of the pulse's frequency-point distribution that
        defines the lower-frequency region.

    upper_decreasing_cutoff_hz : float, default=5000
        Maximum allowed decrease between consecutive retained points
        while the current point is above the lower-frequency percentile.

    lower_decreasing_cutoff_hz : float, default=2000
        Maximum allowed decrease between consecutive retained points
        while the current point is at or below the lower-frequency
        percentile.

    Returns
    -------
    pd.DataFrame
        Full input DataFrame with rejected pulse points assigned
        status = 1. No rows are removed.
    """

    if df.empty:
        return df.copy()

    if not 0.0 <= frequency_percentile <= 1.0:
        raise ValueError(
            "frequency_percentile must be between 0 and 1."
        )

    if upper_decreasing_cutoff_hz < 0:
        raise ValueError(
            "upper_decreasing_cutoff_hz must be non-negative."
        )

    if lower_decreasing_cutoff_hz < 0:
        raise ValueError(
            "lower_decreasing_cutoff_hz must be non-negative."
        )

    required_columns = {"pulse_id", "freq_hz", "status"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {sorted(missing)}"
        )

    df_out = df.copy()

    # Process each pulse independently.
    for _, pulse in df.groupby("pulse_id", sort=False):

        pulse = pulse.sort_index()

        frequencies = pulse["freq_hz"].to_numpy(dtype=float)
        indices = pulse.index.to_numpy()

        # Nothing to filter for a one-point pulse.
        if len(pulse) <= 1:
            continue

        # Determine the lower-frequency boundary.
        frequency_boundary = np.nanquantile(
            frequencies,
            frequency_percentile
        )

        # First point is always retained.
        previous_frequency = frequencies[0]

        # Evaluate each subsequent point.
        for idx, frequency in zip(
            indices[1:],
            frequencies[1:]
        ):

            # Increasing or flat frequency is always retained.
            if frequency >= previous_frequency:
                previous_frequency = frequency
                continue

            # Frequency is decreasing.
            decrease = previous_frequency - frequency

            # The current point determines which cutoff applies.
            if frequency <= frequency_boundary:
                cutoff = lower_decreasing_cutoff_hz
            else:
                cutoff = upper_decreasing_cutoff_hz

            # Keep the point if its decrease is within the allowed cutoff.
            if decrease <= cutoff:
                previous_frequency = frequency

            else:
                # Reject the point, but do not remove it.
                df_out.loc[idx, "status"] = 1

                # previous_frequency intentionally remains unchanged
                # so that the next point is compared against the
                # previous retained point.

    return df_out