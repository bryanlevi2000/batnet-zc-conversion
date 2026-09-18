# ============================================================
# INPUT / OUTPUT
# ============================================================

INPUT_DIR = r"C:\Users\Students\Desktop\pulse detection\wav_files"

OUTPUT_DIR = r"C:\Users\Students\Desktop\pulse detection\custom_zc_14\files"

# Output mode - specify "individual" to have one CSV per pulse,
# or "combined" to have one CSV per WAV file
OUTPUT_MODE = "combined"  # individual or combined


# ============================================================
# DENOISING
# ============================================================

NOISE_FACTOR = 2.0


# ============================================================
# ZC EXTRACTION
# ============================================================

DIVRAT = 8

HYSTERESIS_PERCENTILE = 85.0


# ============================================================
# PULSE DETECTION
# ============================================================

# Local linearity
LOOKAHEAD_N = 10

LINEARITY_CUTOFF = 0.9

# Greedy walk
TIME_CUTOFF = 500  # microseconds

INC_FREQ_CUTOFF_HZ = 1000

DEC_FREQ_CUTOFF_HZ = 5000

MIN_POINTS = 10

BUFFER_US = 10000


# ============================================================
# PULSE CLEANUP
# ============================================================

# Adaptive pulse frequency-gap filtering
PULSE_FREQUENCY_LOWER_PERCENTILE = 0.10

PULSE_UPPER_DECREASING_FREQ_CUTOFF_HZ = 5000

PULSE_LOWER_DECREASING_FREQ_CUTOFF_HZ = 1000


# ============================================================
# PULSE FILTERING
# ============================================================

MIN_BANDWIDTH_HZ = 2000

MAX_BANDWIDTH_HZ = 120000

MIN_DURATION_US = 100

MAX_DURATION_US = 40000

MIN_DECREASING_PERCENT = 0.2

MAX_INCREASING_PERCENT = 1.0

FMIN_RANGES = (5000, 50000)

FMAX_RANGES = (10000, 130000)


# ============================================================
# PARALLEL PROCESSING
# ============================================================

NUM_WORKERS = 12