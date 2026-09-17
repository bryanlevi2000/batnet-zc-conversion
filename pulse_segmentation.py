# pulse_segmentation.py
import pandas as pd


def assign_pulse_ids(
    df: pd.DataFrame,
    pulse_time_gap: float,
    pulse_inc_freq_gap: float,
    pulse_dec_freq_gap: float
) -> pd.DataFrame:
    """
    Assign pulse IDs based on time and directional frequency gaps
    between consecutive points.

    A new pulse is started whenever either:

    1. The time difference between consecutive points is greater than
       or equal to pulse_time_gap, OR
    2. The frequency increases by an amount greater than or equal to
       pulse_inc_freq_gap, OR
    3. The frequency decreases by an amount greater than or equal to
       pulse_dec_freq_gap.

    Parameters
    ----------
    df : pandas.DataFrame
        Zero-crossing dataframe containing 'time_us' and 'freq_hz' columns.

    pulse_time_gap : float
        Time gap threshold for separating pulses.
        Units: microseconds (us).

    pulse_inc_freq_gap : float
        Frequency increase threshold for separating pulses.
        Units: Hz.

    pulse_dec_freq_gap : float
        Frequency decrease threshold for separating pulses.
        Units: Hz.

    Returns
    -------
    pandas.DataFrame
        Dataframe with a new 'pulse_id' column.
    """

    df = df.copy()

    # Ensure chronological order
    df = df.sort_values("time_us").reset_index(drop=True)

    # Calculate differences between consecutive points
    time_gap = df["time_us"].diff()
    frequency_change = df["freq_hz"].diff()

    # Separate increasing and decreasing frequency changes
    increasing_frequency_gap = frequency_change.where(
        frequency_change > 0,
        0
    )

    decreasing_frequency_gap = (-frequency_change).where(
        frequency_change < 0,
        0
    )

    # Start a new pulse if any threshold is met
    new_pulse = (
        (time_gap >= pulse_time_gap)
        | (increasing_frequency_gap >= pulse_inc_freq_gap)
        | (decreasing_frequency_gap >= pulse_dec_freq_gap)
    )

    # Cumulative sum creates pulse IDs starting at 1
    df["pulse_id"] = new_pulse.cumsum() + 1

    return df
