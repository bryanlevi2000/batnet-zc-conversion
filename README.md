# batnet-zc-conversion
Python pipeline to extract zero crossing bat pulses from audio files. The workflow is:
- Read wav file
- Denoise with median subtraction
- Extract zc points
- Get local linearity of every nth point (R2 values)
- Greedy walk algorithm starting at candidate pulse points (with high R2 values), walking forward and backward in time accepting dots within 
    a prespecified time and frequency. When walk ends, restart at next candidate point that is farther away than a buffer time. Each walk becomes a pulse_id
- For each pulse, bottom 10% of points in frequency get a tighter decreasing frequency constraint than the greedy walk
- Get pulse metrics and filter pulses so metrics make semantic sense for bats
