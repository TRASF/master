import numpy as np
from numpy.typing import NDArray
import scipy.io.wavfile
from scipy.signal import resample_poly
from math import gcd

TARGET_SR = 8000


def audio_loader(
    path: str,
    target_sr: int = TARGET_SR,
) -> NDArray[np.float32]:
    """
    Load WAV as normalized float32 waveform for ML training.

    Rule:
        - int32 WAV is treated as 24-bit PCM stored left-justified by scipy.
        - int32 is shifted to 16-bit-equivalent using >> 16.
        - int16 WAV is already accepted as native 16-bit.
        - final output is float32 in approximately [-1, 1].
    """
    sr, x = scipy.io.wavfile.read(path)

    # Convert integer PCM to 16-bit-equivalent float32.
    data = bit_shifting(x)
    
    # Convert stereo/multi-channel to mono after normalization
    if data.ndim > 1:
        data = np.mean(data, axis=1).astype(np.float32)

    data = np.clip(data, -1.0, 1.0)

    if sr != target_sr:
        data = resample(data, sr, target_sr)

    return data.astype(np.float32)

def resample(data: NDArray[np.float32], original_sr: int, target_sr: int = TARGET_SR) -> NDArray[np.float32]:
    if original_sr != target_sr:
        factor = gcd(original_sr, target_sr)
        up = target_sr // factor
        down = original_sr // factor

        data = resample_poly(data, up, down).astype(np.float32)

    return data.astype(np.float32)

def bit_shifting(x: NDArray[any]) -> NDArray[np.int16]:
    if x.dtype == np.int32:
        x_i16 = (x >> 16).astype(np.int16)
        data = x_i16.astype(np.float32) / 32768.0

    elif x.dtype == np.int16:
        # Already native 16-bit PCM
        data = x.astype(np.float32) / 32768.0

    elif np.issubdtype(x.dtype, np.floating):
        # Float WAV is usually already normalized.
        data = x.astype(np.float32)
    else:
        raise ValueError(f"Unsupported audio dtype: {x.dtype}")

    return data.astype(np.float32)