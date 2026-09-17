# ============================================================
# INPUT / OUTPUT
# ============================================================

INPUT_DIR = r"C:\Users\Students\Desktop\pulse detection\wav_files"

OUTPUT_DIR = r"C:\Users\Students\Desktop\pulse detection\custom_zc_12\files"

# Output mode- Specify individual to have one csv per pulse, or combined to have one csv per wav file
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
TIME_CUTOFF_US = 500

INC_FREQ_CUTOFF_HZ = 1000

DEC_FREQ_CUTOFF_HZ = 5000

MIN_POINTS = 10


# ============================================================
# PULSE SEGMENTATION
# ============================================================

# Gap between points that causes a new pulse
PULSE_TIME_GAP_US = 10000

# Frequency gap in hz between points that causes a new pulse if frequency is increasing
PULSE_INC_FREQ_GAP = 1000000000000

# Frequency gap in hz between points that causes a new pulse if frequency is decreasing
PULSE_DEC_FREQ_GAP = 5000000000000

# ============================================================
# PULSE FREQUENCY FILTERING
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

MIN_DECREASING_PERCENT = 20

MAX_INCREASING_PERCENT = 100

# ============================================================
# PARALLEL PROCESSING
# ============================================================

NUM_WORKERS = 24
