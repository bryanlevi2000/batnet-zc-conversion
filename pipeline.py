# pipeline.py

import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

import config

from denoise import denoise_audio
from zc_extraction import extract_zc
from pulse_detection import detect_pulse_points
from pulse_segmentation import assign_pulse_ids
from pulse_frequency_filtering import filter_pulse_frequency_gaps
from pulse_filtering import filter_pulses
from output import write_pulses


def write_empty_output(wav_path):
    """
    Write an empty output CSV for a WAV file that produced no accepted
    pulses.

    This is primarily used for combined output mode so that every input
    WAV file has a corresponding output CSV.
    """

    empty_df = pd.DataFrame(
        columns=[
            "time_us",
            "freq_hz",
            "amplitude",
            "pulse_id"
        ]
    )

    return write_pulses(
        df=empty_df,
        output_dir=config.OUTPUT_DIR,
        source_path=wav_path,
        input_dir=config.INPUT_DIR,
        output_mode=config.OUTPUT_MODE
    )


def process_file(wav_path):
    """
    Process one WAV file through the complete bat pulse pipeline.

    All intermediate data remain in memory.

    Pipeline:

        WAV
          ↓
        Denoising
          ↓
        Zero-crossing extraction
          ↓
        Pulse point detection
          ↓
        Pulse segmentation
          ↓
        Adaptive pulse frequency-gap filtering
          ↓
        Pulse metric filtering
          ↓
        Output

    In individual output mode, accepted pulses are written to individual
    CSV files.

    In combined output mode, all accepted pulses are written to one CSV
    file per WAV file. An empty CSV is also written when no accepted
    pulses are found.
    """

    wav_path = Path(wav_path)

    try:

        # ------------------------------------------------------------
        # 1. Load WAV
        # ------------------------------------------------------------

        y, sample_rate = librosa.load(
            wav_path,
            sr=None,
            mono=False
        )

        # If stereo, use the first channel.
        if y.ndim > 1:
            y = y[0]

        # ------------------------------------------------------------
        # 2. Denoise
        # ------------------------------------------------------------

        y_denoised = denoise_audio(
            y,
            noise_factor=config.NOISE_FACTOR
        )

        # ------------------------------------------------------------
        # 3. Extract zero crossings
        # ------------------------------------------------------------

        # Convert normalized floating-point audio back to the int16 scale
        # expected by the ZC extraction algorithm.
        y_denoised = np.clip(
            y_denoised * 32767,
            -32768,
            32767
        ).astype(np.int16)

        zc_data = extract_zc(
            y_denoised,
            sample_rate,
            divrat=config.DIVRAT,
            hysteresis_percentile=config.HYSTERESIS_PERCENTILE
        )

        # Convert ZC array to DataFrame.
        zc_df = pd.DataFrame(
            zc_data,
            columns=[
                "time_us",
                "freq_hz",
                "amplitude"
            ]
        )

        # ------------------------------------------------------------
        # No ZC points
        # ------------------------------------------------------------

        if zc_df.empty:

            pulses_written = write_empty_output(wav_path)

            return {
                "file": str(wav_path),
                "success": True,
                "pulses_written": pulses_written,
                "error": ""
            }

        # ------------------------------------------------------------
        # 4. Detect potential pulse points
        # ------------------------------------------------------------

        pulse_points = detect_pulse_points(
            zc_df,
            lookahead_n=config.LOOKAHEAD_N,
            linearity_cutoff=config.LINEARITY_CUTOFF,
            time_cutoff=config.TIME_CUTOFF_US,
            inc_freq_cutoff=config.INC_FREQ_CUTOFF_HZ,
            dec_freq_cutoff=config.DEC_FREQ_CUTOFF_HZ,
            min_points=config.MIN_POINTS
        )

        # ------------------------------------------------------------
        # No potential pulse points
        # ------------------------------------------------------------

        if pulse_points.empty:

            pulses_written = write_empty_output(wav_path)

            return {
                "file": str(wav_path),
                "success": True,
                "pulses_written": pulses_written,
                "error": ""
            }

        # ------------------------------------------------------------
        # 5. Segment into individual pulses
        # ------------------------------------------------------------

        pulse_data = assign_pulse_ids(
            pulse_points,
            pulse_time_gap=config.PULSE_TIME_GAP_US,
            pulse_inc_freq_gap=config.PULSE_INC_FREQ_GAP,
            pulse_dec_freq_gap=config.PULSE_DEC_FREQ_GAP
        )

        # ------------------------------------------------------------
        # No pulse data
        # ------------------------------------------------------------

        if pulse_data.empty:

            pulses_written = write_empty_output(wav_path)

            return {
                "file": str(wav_path),
                "success": True,
                "pulses_written": pulses_written,
                "error": ""
            }

        # ------------------------------------------------------------
        # 6. Adaptive pulse frequency-gap filtering
        # ------------------------------------------------------------

        pulse_data = filter_pulse_frequency_gaps(
            pulse_data,
            frequency_percentile=config.PULSE_FREQUENCY_LOWER_PERCENTILE,
            upper_decreasing_cutoff_hz=(
                config.PULSE_UPPER_DECREASING_FREQ_CUTOFF_HZ
            ),
            lower_decreasing_cutoff_hz=(
                config.PULSE_LOWER_DECREASING_FREQ_CUTOFF_HZ
            )
        )

        # ------------------------------------------------------------
        # No pulse data after adaptive frequency-gap filtering
        # ------------------------------------------------------------

        if pulse_data.empty:

            pulses_written = write_empty_output(wav_path)

            return {
                "file": str(wav_path),
                "success": True,
                "pulses_written": pulses_written,
                "error": ""
            }

        # ------------------------------------------------------------
        # 7. Filter pulses by pulse-level metrics
        # ------------------------------------------------------------

        accepted_pulses, pulse_metrics = filter_pulses(
            pulse_data,
            min_bandwidth_hz=config.MIN_BANDWIDTH_HZ,
            max_bandwidth_hz=config.MAX_BANDWIDTH_HZ,
            min_duration_us=config.MIN_DURATION_US,
            max_duration_us=config.MAX_DURATION_US,
            min_decreasing_percent=(
                config.MIN_DECREASING_PERCENT / 100.0
            ),
            max_increasing_percent=(
                config.MAX_INCREASING_PERCENT / 100.0
            )
        )

        # ------------------------------------------------------------
        # No accepted pulses
        # ------------------------------------------------------------

        if accepted_pulses.empty:

            pulses_written = write_empty_output(wav_path)

            return {
                "file": str(wav_path),
                "success": True,
                "pulses_written": pulses_written,
                "error": ""
            }

        # ------------------------------------------------------------
        # 8. Write accepted pulses
        # ------------------------------------------------------------

        pulses_written = write_pulses(
            df=accepted_pulses,
            output_dir=config.OUTPUT_DIR,
            source_path=wav_path,
            input_dir=config.INPUT_DIR,
            output_mode=config.OUTPUT_MODE
        )

        return {
            "file": str(wav_path),
            "success": True,
            "pulses_written": pulses_written,
            "error": ""
        }

    except Exception as e:

        return {
            "file": str(wav_path),
            "success": False,
            "pulses_written": 0,
            "error": f"{type(e).__name__}: {e}"
        }


def find_wav_files(input_dir):
    """Recursively find all WAV files in input_dir."""

    input_dir = Path(input_dir)

    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() == ".wav"
    )


def write_failure_log(results, output_dir):
    """Write a CSV containing only failed files."""

    failed = [
        result
        for result in results
        if not result["success"]
    ]

    if not failed:
        return None

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    log_path = output_dir / "failed_files.csv"

    pd.DataFrame(failed).to_csv(
        log_path,
        index=False
    )

    return log_path


def run_pipeline():

    input_dir = Path(config.INPUT_DIR)
    output_dir = Path(config.OUTPUT_DIR)

    # ------------------------------------------------------------
    # Validate input directory
    # ------------------------------------------------------------

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist:\n{input_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # Find WAV files
    # ------------------------------------------------------------

    wav_files = find_wav_files(input_dir)

    if not wav_files:
        print("No WAV files found.")
        return

    print(f"Found {len(wav_files):,} WAV files.")

    # ------------------------------------------------------------
    # Parallel processing
    # ------------------------------------------------------------

    results = []

    with ProcessPoolExecutor(
        max_workers=config.NUM_WORKERS
    ) as executor:

        futures = {
            executor.submit(process_file, wav_path): wav_path
            for wav_path in wav_files
        }

        with tqdm(
            total=len(futures),
            desc="Processing",
            unit="file"
        ) as progress:

            for future in as_completed(futures):

                result = future.result()

                results.append(result)

                progress.update(1)

    # ------------------------------------------------------------
    # Write failure log
    # ------------------------------------------------------------

    log_path = write_failure_log(
        results,
        output_dir
    )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    successful = sum(
        result["success"]
        for result in results
    )

    failed = sum(
        not result["success"]
        for result in results
    )

    total_pulses = sum(
        result["pulses_written"]
        for result in results
    )

    print()
    print(f"Completed: {successful:,}/{len(results):,} files")
    print(f"Pulses written: {total_pulses:,}")
    print(f"Failed: {failed:,} files")

    if log_path is not None:
        print(f"Failure log: {log_path}")


if __name__ == "__main__":

    # Suppress unnecessary warnings/messages during batch processing.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        run_pipeline()
