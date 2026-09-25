# ============================================================
# INPUT / OUTPUT
# ============================================================

INPUT_DIR = r"path\to\input\directory"

OUTPUT_DIR = r"path\to\output\directory"

# Output mode - specify "individual" to have one CSV per pulse,
# or "combined" to have one CSV per WAV file
OUTPUT_MODE = "individual"  # individual or combined


# ============================================================
# DENOISING
# ============================================================

NOISE_FACTOR = 2.0
NOISE_STRIDE = 4


# ============================================================
# ZC EXTRACTION
# ============================================================

DIVRAT = 8
HYSTERESIS_PERCENTILE = 85.0


# ============================================================
# PULSE DETECTION
# ============================================================

# Local linearity
LOOKAHEAD_N = 15
LINEARITY_CUTOFF = 0.9

# Greedy walk
TIME_CUTOFF = 500  # microseconds
INC_FREQ_CUTOFF_HZ = 1000
DEC_FREQ_CUTOFF_HZ = 5000
MIN_POINTS = 15
BUFFER_US = 10000

# ============================================================
# PULSE FILTERING
# ============================================================

# Characteristic frequency
CHARACTERISTIC_MIN_POINTS = 2
CHARACTERISTIC_LAST_FRACTION = 0.33

# Maximum allowed frequency decrease after characteristic frequency
CHARACTERISTIC_FREQUENCY_DECREASE_CUTOFF_HZ = 1000

# Pulse metrics
MIN_BANDWIDTH_HZ = 2000
MAX_BANDWIDTH_HZ = 120000

MIN_DURATION_US = 100
MAX_DURATION_US = 40000

FMIN_RANGES = (5000, 50000)
FMAX_RANGES = (10000, 130000)


# ============================================================
# PARALLEL PROCESSING
# ============================================================

NUM_WORKERS = 24