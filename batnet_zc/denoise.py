# denoise.py

import numpy as np
from scipy import signal


def denoise_audio(y, noise_factor=2.0, n_fft=512, noise_stride=4):
    """Spectral-subtraction denoiser using a per-frequency median noise floor.

    Parameters
    ----------
    y : array-like
        1-D audio signal.
    noise_factor : float
        Multiplier applied to the median noise floor before subtraction.
    n_fft : int
        STFT window length (samples). Hop is n_fft // 2 (50% overlap).
    noise_stride : int
        Use every `noise_stride`-th time frame when estimating the noise
        floor. 1 reproduces the full median over all frames; larger values
        are faster and nearly identical for long, stationary-noise files.

    Returns
    -------
    np.ndarray
        Denoised signal, same length as the input.
    """
    y = np.asarray(y, dtype=np.float32)
    n = len(y)
    noverlap = n_fft // 2

    # Pad with zeros at the edges so the STFT is invertible everywhere
    # (satisfies the NOLA condition; no warning from istft).
    _, _, S = signal.stft(
        y, nperseg=n_fft, noverlap=noverlap, boundary="zeros"
    )

    # Noise floor: per-frequency median magnitude across (a subset of) frames
    mag = np.abs(S)
    noise = np.median(mag[:, ::noise_stride], axis=1, keepdims=True)
    noise *= noise_factor

    # Spectral subtraction as a gain applied directly to the complex spectrum:
    #   gain = max(1 - noise / mag, 0)
    # (equivalent to clip(mag - noise, 0) * S / mag, without the phase
    # round trip). Done in place on `mag` to avoid extra temporaries.
    np.maximum(mag, 1e-10, out=mag)   # avoid division by zero
    np.divide(noise, mag, out=mag)    # noise / mag
    np.subtract(1.0, mag, out=mag)    # 1 - noise / mag
    np.maximum(mag, 0.0, out=mag)     # clip -> gain
    S *= mag

    # Inverse STFT; boundary=True strips the padding added by stft above.
    _, y_denoised = signal.istft(
        S, nperseg=n_fft, noverlap=noverlap, boundary=True
    )

    # stft's default padded=True can leave a few trailing samples
    return y_denoised[:n]
    