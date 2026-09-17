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
    Remove pulse points that follow an excessively large decreasing
    frequency gap, using a more restrictive cutoff for points in the
    lower-frequency portion of each pulse.

    The lower-frequency portion is determined from the distribution
    of frequency points within each pulse. For example, with
    frequency_percentile=0.10, points at or below the 10th percentile
    of the pulse's freq_hz distribution use the lower decreasing-
    frequency cutoff.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least:
            - pulse_id
            - freq_hz

    frequency_percentile : float, default=0.10
        Percentile of the pulse's frequency-point distribution that
        defines the lower-frequency region.

        For example, 0.10 means the lowest 10% of frequency values
        within each pulse use the lower decreasing-frequency cutoff.

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
        DataFrame containing only retained points. Original columns,
        row indices, row order, and pulse IDs are preserved.
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

    required_columns = {"pulse_id", "freq_hz"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {sorted(missing)}"
        )

    retained_indices = []

    # Process each pulse independently.
    for _, pulse in df.groupby("pulse_id", sort=False):

        pulse = pulse.sort_index()

        frequencies = pulse["freq_hz"].to_numpy(dtype=float)
        indices = pulse.index.to_numpy()

        # Nothing to filter for a one-point pulse.
        if len(pulse) <= 1:
            retained_indices.extend(indices)
            continue

        # ------------------------------------------------------------
        # Determine the lower-frequency region from the actual
        # distribution of frequency points in this pulse.
        #
        # Example:
        #     frequency_percentile = 0.10
        #
        #     → points at or below the 10th percentile use the
        #       lower decreasing-frequency cutoff.
        # ------------------------------------------------------------

        frequency_boundary = np.nanquantile(
            frequencies,
            frequency_percentile
        )

        # First point is always retained.
        retained = [indices[0]]
        previous_frequency = frequencies[0]

        # ------------------------------------------------------------
        # Evaluate each subsequent point.
        # ------------------------------------------------------------

        for idx, frequency in zip(
            indices[1:],
            frequencies[1:]
        ):

            # Increasing or flat frequency is always retained.
            if frequency >= previous_frequency:
                retained.append(idx)
                previous_frequency = frequency
                continue

            # Frequency is decreasing.
            decrease = previous_frequency - frequency

            # The CURRENT point determines which cutoff applies.
            #
            # Points in the bottom frequency percentile are subject
            # to the more restrictive cutoff.
            if frequency <= frequency_boundary:
                cutoff = lower_decreasing_cutoff_hz
            else:
                cutoff = upper_decreasing_cutoff_hz

            # Keep the point if its decrease is within the
            # allowed cutoff.
            if decrease <= cutoff:
                retained.append(idx)
                previous_frequency = frequency

            # If the decrease exceeds the cutoff, the point is
            # rejected. previous_frequency intentionally remains
            # unchanged so that the next point is compared against
            # the previous retained point.

        retained_indices.extend(retained)

    return df.loc[retained_indices].copy()

