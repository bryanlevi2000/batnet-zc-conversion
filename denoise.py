#denoise.py

import numpy as np
import librosa

def denoise_audio(y, noise_factor=1.0):
    """
    Apply spectral median subtraction to an audio signal.

    Parameters
    ----------
    y : np.ndarray
        Audio signal.
    noise_factor : float
        Scaling factor for the spectral median subtraction.

    Returns
    -------
    y_denoised : np.ndarray
        Denoised audio signal.
    """

    # 1. Compute the Short-Time Fourier Transform
    S = librosa.stft(y, n_fft=512)

    # 2. Separate magnitude and phase
    mag, phase = librosa.magphase(S)

    # 3. Calculate median magnitude across time for each frequency bin
    median_mag = np.median(mag, axis=1, keepdims=True)

    # 4. Subtract the median spectrum
    mag_denoised = mag - (median_mag * noise_factor)

    # 5. Half-wave rectification
    mag_denoised = np.maximum(mag_denoised, 0)

    # 6. Recombine denoised magnitude with original phase
    S_denoised = mag_denoised * phase

    # 7. Convert back to time domain
    y_denoised = librosa.istft(S_denoised)

    return y_denoised
