# pulse_detection.py
import warnings

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def filter_zc_linearity(
    df,
    lookahead_n=10,
    linearity_cutoff=0.8
):
    """Calculate local point linearity every lookahead_n / 2 points."""

    df_out = (
        df
        if df["time_us"].is_monotonic_increasing
        else df.sort_values("time_us")
    ).copy()

    t = df_out["time_us"].values.astype(np.float64)
    f = df_out["freq_hz"].values.astype(np.float64)

    step = max(1, lookahead_n // 2)

    if len(t) <= lookahead_n:
        df_out["linearity"] = np.nan
        df_out["status"] = 1
        return df_out

    # Create all possible windows, then keep only every `step`th window.
    t_win = sliding_window_view(t, lookahead_n + 1)[::step]
    f_win = sliding_window_view(f, lookahead_n + 1)[::step]

    dt = t_win - t_win.mean(axis=1, keepdims=True)
    df_val = f_win - f_win.mean(axis=1, keepdims=True)

    cov = (dt * df_val).sum(axis=1)
    var_t = (dt ** 2).sum(axis=1)
    var_f = (df_val ** 2).sum(axis=1)

    denom = var_t * var_f

    r2 = np.where(
        denom > 0,
        (cov ** 2) / denom,
        0.0
    )

    linearity = np.full(len(t), np.nan, dtype=np.float64)

    # Put each calculated value at its corresponding point.
    indices = np.arange(0, len(t) - lookahead_n, step)
    linearity[indices] = r2

    df_out["linearity"] = linearity

    df_out["status"] = np.where(
        (linearity >= linearity_cutoff) & np.isfinite(linearity),
        3,
        1
    )

    return df_out


def greedy_walk(
    df,
    time_cutoff,
    inc_freq_cutoff,
    dec_freq_cutoff,
    min_points
):
    """Greedily expand high-linearity points into candidate pulse segments."""

    df_out = (
        df
        if df["time_us"].is_monotonic_increasing
        else df.sort_values("time_us")
    ).copy()

    t = df_out["time_us"].values
    f = df_out["freq_hz"].values
    s = df_out["status"].values

    n, i = len(s), 0

    while i < n:

        idx = np.where(s[i:] == 3)[0]

        if not idx.size:
            break

        base = i + idx[0]

        # Forward walk
        curr = base

        while curr < n - 1:

            dt = t[curr + 1] - t[curr]
            dfreq = f[curr + 1] - f[curr]

            if (
                dt > time_cutoff
                or (dfreq > 0 and dfreq > inc_freq_cutoff)
                or (dfreq <= 0 and -dfreq > dec_freq_cutoff)
            ):
                break

            s[curr + 1] = 3
            curr += 1

        fw_end = curr

        # Backward walk
        curr = base

        while curr > 0:

            dt = t[curr] - t[curr - 1]
            dfreq = f[curr] - f[curr - 1]

            if (
                dt > time_cutoff
                or (dfreq > 0 and dfreq > inc_freq_cutoff)
                or (dfreq <= 0 and -dfreq > dec_freq_cutoff)
            ):
                break

            s[curr - 1] = 3
            curr -= 1

        bw_end = curr

        # Reject segment if it is too short
        if (fw_end - bw_end + 1) < min_points:
            s[bw_end:fw_end + 1] = 1

        i = fw_end + 1

    df_out["status"] = s

    return df_out


def filter_status_three(df):
    """Keep only points with status == 3."""

    df = df.copy()

    df["status"] = pd.to_numeric(
        df["status"],
        errors="coerce"
    )

    df = df.dropna(subset=["status"])

    df["status"] = df["status"].astype(int)

    filtered_df = df[df["status"] == 3].copy()

    return filtered_df.drop(columns=["status"])


def detect_pulse_points(
    df,
    lookahead_n=10,
    linearity_cutoff=0.8,
    time_cutoff=500,
    inc_freq_cutoff=5000,
    dec_freq_cutoff=5000,
    min_points=20
):
    """
    Identify potential bat pulse points from zero-crossing data.

    Parameters
    ----------
    df : pandas.DataFrame
        Zero-crossing dataframe containing time_us, freq_hz, and amplitude.

    lookahead_n : int
        Number of points used for the local linearity calculation.

    linearity_cutoff : float
        Cutoff value for determining high-linearity points.

    time_cutoff : float
        Maximum allowed time difference between adjacent points
        during the greedy walk, in microseconds.

    inc_freq_cutoff : float
        Maximum allowed frequency increase between adjacent points,
        in Hz.

    dec_freq_cutoff : float
        Maximum allowed frequency decrease between adjacent points,
        in Hz.

    min_points : int
        Minimum number of points required for a greedy-walk segment
        to be retained.

    Returns
    -------
    pandas.DataFrame
        Zero-crossing dataframe containing only points identified
        as potential bat pulse points.
    """

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning
        )

        df = filter_zc_linearity(
            df,
            lookahead_n=lookahead_n,
            linearity_cutoff=linearity_cutoff
        )

        df = greedy_walk(
            df,
            time_cutoff=time_cutoff,
            inc_freq_cutoff=inc_freq_cutoff,
            dec_freq_cutoff=dec_freq_cutoff,
            min_points=min_points
        )

        df = filter_status_three(df)

    return df
