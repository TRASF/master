import os
import numpy as np
import librosa
import soundfile as sf

def load_file(path: str):

    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    data, sr = sf.read(path, dtype='float32', always_2d=False)
    info = sf.info(path)
    return data, sr, info.subtype

def to_mono(data: np.ndarray) -> np.ndarray:
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return data.astype(np.float32)

def resample_audio(data: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    if original_sr == target_sr:
        return data.astype(np.float32)
    data = librosa.resample(
        data.astype(np.float32),
        orig_sr=original_sr,
        target_sr=target_sr,
        res_type="kaiser_best",
    )
    return data.astype(np.float32)

class FileLoader:
    def __init__(self, path: str, sample_rate: int = 8000):
        self.path = path
        self.sample_rate = sample_rate

    def load(self) -> np.ndarray:
        data, sr, _ = load_file(self.path)
        data = to_mono(data)
        data = resample_audio(data, sr, self.sample_rate)
        return data.astype(np.float32)
