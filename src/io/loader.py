import librosa
from numpy import ndarray


def audio_loader(
    path: str, sr: float = 8000, bit_depth: int = 24, target_bit_depth: int = 16
) -> ndarray:

    data, _ = librosa.load(path, sr=sr, mono=True)
    data = bit_shift(data, bit_depth, target_bit_depth)

    return data


def bit_shift(data: ndarray, bits: int, target_bits: int) -> ndarray:

    shift = target_bits - bits

    if shift > 0:
        data = data * (2**shift)
    elif shift < 0:
        data = data / (2 ** abs(shift))

    return data
