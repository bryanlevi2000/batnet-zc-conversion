# pipeline.py

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import librosa
import pandas as pd
from tqdm import tqdm

import config

from denoise import denoise_audio
from zc_extraction import extract_zc
from pulse_detection import detect_pulse_points
from pulse_cleanup import filter_pulse_frequency_gaps
from pulse_filtering import filter_pulses
from output import write_pulses


# ============================================================
# PROCESS ONE WAV FILE
# ============================================================

def process_wav(source_path):
    """
    Process one WAV file through the complete pulse-detection pipeline.
    """

    source_path = Path(source_path)

    # --------------------------------------------------------
    # Load WAV
    # --------------------------------------------------------

    y, sample_rate = librosa.load(
        source_path,
        sr=None,
        mono=True
    )

    # --------------------------------------------------------
    # Denoising
    # --------------------------------------------------------

    y = denoise_audio(
        y,
        noise_factor=config.NOISE_FACTOR,
        noise_stride=config.NOISE_STRIDE
    )

    # Convert back to the integer scale expected by ZC extraction
    y = (y * 32767).astype("int16")

    # --------------------------------------------------------
    # Zero-crossing extraction
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Pulse detection
    # --------------------------------------------------------

    df = detect_pulse_points(
        df,
        lookahead_n=config.LOOKAHEAD_N,
        linearity_cutoff=config.LINEARITY_CUTOFF,
        time_cutoff=config.TIME_CUTOFF,
        inc_freq_cutoff=config.INC_FREQ_CUTOFF_HZ,
        dec_freq_cutoff=config.DEC_FREQ_CUTOFF_HZ,
        min_points=config.MIN_POINTS,
        buffer_us=config.BUFFER_US
    )

    # --------------------------------------------------------
    # Pulse frequency-gap filtering
    # --------------------------------------------------------

    df = filter_pulse_frequency_gaps(
        df,
        frequency_percentile=config.PULSE_FREQUENCY_LOWER_PERCENTILE,
        upper_decreasing_cutoff_hz=(
            config.PULSE_UPPER_DECREASING_FREQ_CUTOFF_HZ
        ),
        lower_decreasing_cutoff_hz=(
            config.PULSE_LOWER_DECREASING_FREQ_CUTOFF_HZ
        )
    )

    # --------------------------------------------------------
    # Pulse filtering
    # --------------------------------------------------------

    df = filter_pulses(
        df,
        min_bandwidth_hz=config.MIN_BANDWIDTH_HZ,
        max_bandwidth_hz=config.MAX_BANDWIDTH_HZ,
        min_duration_us=config.MIN_DURATION_US,
        max_duration_us=config.MAX_DURATION_US,
        min_decreasing_percent=config.MIN_DECREASING_PERCENT,
        max_increasing_percent=config.MAX_INCREASING_PERCENT,
        fmin_ranges=config.FMIN_RANGES,
        fmax_ranges=config.FMAX_RANGES
    )

    # --------------------------------------------------------
    # Keep only accepted pulse points for output
    # --------------------------------------------------------

    output_df = df[df["status"] == 3].copy()

    # --------------------------------------------------------
    # Write output
    # --------------------------------------------------------

    pulse_count = write_pulses(
        output_df,
        output_dir=config.OUTPUT_DIR,
        source_path=source_path,
        input_dir=config.INPUT_DIR,
        output_mode=config.OUTPUT_MODE
    )

    return pulse_count


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    input_dir = Path(config.INPUT_DIR)

    # --------------------------------------------------------
    # Find all WAV files recursively
    # --------------------------------------------------------

    wav_files = [
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".wav"
    ]

    # --------------------------------------------------------
    # Process all files in parallel
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
                future.result()
                progress.update(1)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
    