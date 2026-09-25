# pipeline.py

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

import librosa
import pandas as pd
from tqdm import tqdm

import config

from batnet_zc.denoise import denoise_audio
from batnet_zc.zc_extraction import extract_zc
from batnet_zc.pulse_detection import filter_zc_linearity, greedy_walk
from batnet_zc.pulse_filtering import (
    find_characteristic_frequency,
    apply_frequency_decrease_cutoff,
    filter_pulses,
)
from batnet_zc.output import write_pulses


# ============================================================
# PROCESS ONE WAV FILE
# ============================================================

def process_wav(source_path):

    source_path = Path(source_path)

    times = {
        "read_wav": 0.0,
        "denoising": 0.0,
        "zc_extraction": 0.0,
        "local_linearity": 0.0,
        "greedy_walk": 0.0,
        "characteristic_frequency": 0.0,
        "frequency_decrease_cutoff": 0.0,
        "pulse_filtering": 0.0,
        "csv_writing": 0.0,
    }

    # --------------------------------------------------------
    # Read WAV
    # --------------------------------------------------------

    start = time.perf_counter()

    y, sample_rate = librosa.load(
        source_path,
        sr=None,
        mono=True
    )

    times["read_wav"] = time.perf_counter() - start

    # --------------------------------------------------------
    # Denoising
    # --------------------------------------------------------

    start = time.perf_counter()

    y = denoise_audio(
        y,
        noise_factor=config.NOISE_FACTOR,
        noise_stride=config.NOISE_STRIDE
    )

    # Convert back to the integer scale expected by ZC extraction
    y = (y * 32767).astype("int16")

    times["denoising"] = time.perf_counter() - start

    # --------------------------------------------------------
    # Zero-crossing extraction
    # --------------------------------------------------------

    start = time.perf_counter()

    zc_data = extract_zc(
        y,
        sample_rate,
        divrat=config.DIVRAT,
        hysteresis_percentile=config.HYSTERESIS_PERCENTILE
    )

    df = pd.DataFrame(
        zc_data,
        columns=[
            "time_us",
            "freq_hz",
            "amplitude"
        ]
    )

    times["zc_extraction"] = time.perf_counter() - start

    # --------------------------------------------------------
    # Local linearity
    # --------------------------------------------------------

    start = time.perf_counter()

    df = filter_zc_linearity(
        df,
        lookahead_n=config.LOOKAHEAD_N,
        linearity_cutoff=config.LINEARITY_CUTOFF
    )

    times["local_linearity"] = time.perf_counter() - start

    # --------------------------------------------------------
    # Greedy walk
    # --------------------------------------------------------

    start = time.perf_counter()

    df = greedy_walk(
        df,
        time_cutoff=config.TIME_CUTOFF,
        inc_freq_cutoff=config.INC_FREQ_CUTOFF_HZ,
        dec_freq_cutoff=config.DEC_FREQ_CUTOFF_HZ,
        min_points=config.MIN_POINTS,
        buffer_us=config.BUFFER_US
    )

    times["greedy_walk"] = time.perf_counter() - start

    # --------------------------------------------------------
    # Characteristic frequency
    # --------------------------------------------------------

    start = time.perf_counter()

    # Work only with candidate pulse points
    candidate_df = df[df["status"] > 1].copy()

    for pulse_id, pulse_df in candidate_df.groupby(
        "pulse_id",
        sort=False
    ):

        characteristic_index = find_characteristic_frequency(
            pulse_df,
            min_points=config.CHARACTERISTIC_MIN_POINTS,
            last_fraction=config.CHARACTERISTIC_LAST_FRACTION
        )

        if characteristic_index is not None:
            df.loc[
                characteristic_index,
                "status"
            ] = 3

    times["characteristic_frequency"] = (
        time.perf_counter() - start
    )

    # --------------------------------------------------------
    # Frequency-decrease cutoff
    # --------------------------------------------------------

    start = time.perf_counter()

    candidate_df = df[df["status"] > 1].copy()

    for pulse_id, pulse_df in candidate_df.groupby(
        "pulse_id",
        sort=False
    ):

        # Find the characteristic-frequency point
        characteristic_points = pulse_df[
            pulse_df["status"] == 3
        ]

        if characteristic_points.empty:
            continue

        characteristic_index = characteristic_points.index[0]

        pulse_result = apply_frequency_decrease_cutoff(
            pulse_df,
            characteristic_index=characteristic_index,
            cutoff_hz=(
                config.CHARACTERISTIC_FREQUENCY_DECREASE_CUTOFF_HZ
            )
        )

        # Copy modified status values back into main dataframe
        df.loc[
            pulse_result.index,
            "status"
        ] = pulse_result["status"]

    times["frequency_decrease_cutoff"] = (
        time.perf_counter() - start
    )

    # --------------------------------------------------------
    # Pulse metric calculation and filtering
    # --------------------------------------------------------

    start = time.perf_counter()

    df = filter_pulses(
        df,
        min_bandwidth_hz=config.MIN_BANDWIDTH_HZ,
        max_bandwidth_hz=config.MAX_BANDWIDTH_HZ,
        min_duration_us=config.MIN_DURATION_US,
        max_duration_us=config.MAX_DURATION_US,
        fmin_ranges=config.FMIN_RANGES,
        fmax_ranges=config.FMAX_RANGES
    )

    times["pulse_filtering"] = (
        time.perf_counter() - start
    )

    # --------------------------------------------------------
    # Keep only accepted pulse points for output
    # --------------------------------------------------------

    output_df = df[df["status"] > 1].copy()

    # --------------------------------------------------------
    # Write CSVs
    # --------------------------------------------------------

    start = time.perf_counter()

    pulse_count = write_pulses(
        output_df,
        output_dir=config.OUTPUT_DIR,
        source_path=source_path,
        input_dir=config.INPUT_DIR,
        output_mode=config.OUTPUT_MODE
    )

    times["csv_writing"] = time.perf_counter() - start

    return times, pulse_count


# ============================================================
# MAIN
# ============================================================

def main():

    input_dir = Path(config.INPUT_DIR)

    # --------------------------------------------------------
    # Find all WAV files
    # --------------------------------------------------------

    wav_files = [
        path
        for path in input_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() == ".wav"
    ]

    print(f"Found {len(wav_files):,} WAV files.")

    if not wav_files:
        return

    # --------------------------------------------------------
    # Initialize timing totals
    # --------------------------------------------------------

    total_times = {
        "read_wav": 0.0,
        "denoising": 0.0,
        "zc_extraction": 0.0,
        "local_linearity": 0.0,
        "greedy_walk": 0.0,
        "characteristic_frequency": 0.0,
        "frequency_decrease_cutoff": 0.0,
        "pulse_filtering": 0.0,
        "csv_writing": 0.0,
    }

    total_pulses = 0

    # --------------------------------------------------------
    # Start wall-clock timer
    # --------------------------------------------------------

    wall_start = time.perf_counter()

    # --------------------------------------------------------
    # Process files in parallel
    # --------------------------------------------------------

    with ProcessPoolExecutor(
        max_workers=config.NUM_WORKERS
    ) as executor:

        futures = [
            executor.submit(process_wav, wav_path)
            for wav_path in wav_files
        ]

        with tqdm(
            total=len(futures),
            desc="Processing WAV files",
            unit="file"
        ) as progress:

            for future in as_completed(futures):

                times, pulse_count = future.result()

                for step in total_times:
                    total_times[step] += times[step]

                total_pulses += pulse_count

                progress.update(1)

    wall_time = time.perf_counter() - wall_start

    # --------------------------------------------------------
    # Timing report
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PIPELINE TIMING")
    print("=" * 60)

    print(f"\nFiles processed: {len(wav_files):,}")
    print(f"Total pulses:    {total_pulses:,}")

    print("\nCumulative processing time:")
    print("-" * 60)

    labels = {
        "read_wav": "Reading WAV files",
        "denoising": "Denoising",
        "zc_extraction": "Zero-crossing extraction",
        "local_linearity": "Local linearity calculation",
        "greedy_walk": "Greedy walk",
        "characteristic_frequency": "Characteristic frequency",
        "frequency_decrease_cutoff": "Frequency decrease cutoff",
        "pulse_filtering": "Pulse metric calculation/filtering",
        "csv_writing": "Writing CSVs",
    }

    for step, label in labels.items():

        seconds = total_times[step]

        print(
            f"{label:<35} "
            f"{seconds:>12.3f} seconds"
        )

    print("-" * 60)

    print(
        f"{'TOTAL CUMULATIVE STAGE TIME':<35} "
        f"{sum(total_times.values()):>12.3f} seconds"
    )

    print(
        f"{'TOTAL WALL-CLOCK TIME':<35} "
        f"{wall_time:>12.3f} seconds"
    )

    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
    