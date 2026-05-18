import os
import numpy as np
import yaml
import tensorflow as tf
from pathlib import Path
from typing import List, Tuple, Dict, Generator, Optional
from src.io.loader import audio_loader
from librosa.util import frame

class LabelParser:
    """
    Handles label extraction, normalization, and mapping from directory structures.
    """
    def __init__(self, class_dict: Dict[str, int]):
        self.class_dict = class_dict
        self.reverse_dict = {v: k for k, v in class_dict.items()}

    def normalize_folder_name(self, folder_name: str) -> str:
        name_lower = folder_name.lower()
        # Explicitly map known background noise folders to No.Mos
        if name_lower in ('nomos', 'no_mos', 'no.mos', 'cat_noise', 'noises', 'inmp_noises', 'humbug_noises', 'miru_noises'):
            return 'No.Mos'
            
        parts = folder_name.split('_')
        if len(parts) >= 3:
            genus = parts[0]
            species = parts[1].capitalize()
            gender = parts[2][0].upper()
            return f"{genus}.{species}_{gender}"
            
        return folder_name

    def get_label_id(self, folder_name: str) -> int:
        normalized = self.normalize_folder_name(folder_name)
        return self.class_dict.get(normalized, -1)

    def parse_directory(self, root_path: str) -> List[Tuple[str, int]]:
        samples = []
        root = Path(root_path)
        if not root.exists():
            raise FileNotFoundError(f"Dataset root path not found: {root_path}")
            
        # Check if root itself is a valid class folder
        root_label_id = self.get_label_id(root.name)
        if root_label_id != -1:
            for file_path in root.glob('*.wav'):
                samples.append((str(file_path), root_label_id))
            # Also look into subdirectories of a labeled root (e.g. Environmental/noises/Train)
            for sub_folder in root.iterdir():
                if sub_folder.is_dir():
                    for file_path in sub_folder.glob('**/*.wav'):
                        samples.append((str(file_path), root_label_id))
            if samples:
                return samples

        for folder in root.iterdir():
            if not folder.is_dir():
                continue
                
            label_id = self.get_label_id(folder.name)
            if label_id == -1:
                continue
                
            for file_path in folder.glob('**/*.wav'):
                samples.append((str(file_path), label_id))
                
        return samples

class AudioSegmenter:
    """
    Handles audio slicing, padding, and framing logic.
    """
    def __init__(self, sample_rate: int, duration_sec: float, overlap_pct: float):
        self.sample_rate = sample_rate
        self.segment_size = int(duration_sec * sample_rate)
        self.hop_size = int(self.segment_size * (1 - overlap_pct))

    def _pad_audio(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) < self.segment_size:
            pad_width = self.segment_size - len(audio)
            return np.pad(audio, (0, pad_width))
        return audio

    def segment(self, audio: np.ndarray, force_single: bool = True) -> np.ndarray:
        if len(audio) == self.segment_size:
            return np.expand_dims(audio, axis=0)
            
        padded_audio = self._pad_audio(audio)
        segments = frame(padded_audio, frame_length=self.segment_size, hop_length=self.hop_size)
        return segments.T

class AudioDataLoader:
    """
    Orchestrator for loading and segmenting audio data.
    """
    def __init__(self, config_path: str = "configs/defaults.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.parser = LabelParser(self.config['class_dict'])
        audio_cfg = self.config['audio']
        self.segmenter = AudioSegmenter(
            sample_rate=audio_cfg['sample_rate'],
            duration_sec=audio_cfg['duration'],
            overlap_pct=audio_cfg['overlap']
        )

    def load_file(self, file_path: str) -> np.ndarray:
        audio = audio_loader(file_path)
        return self.segmenter.segment(audio)

    def get_samples(self, root_path: str) -> List[Tuple[str, int]]:
        return self.parser.parse_directory(root_path)

    def generate_from_samples(self, samples: List[Tuple[str, int]]) -> Generator[Tuple[np.ndarray, int], None, None]:
        for file_path, label_id in samples:
            segments = self.load_file(file_path)
            for seg in segments:
                yield seg, label_id

    def prepare_segments(self, samples: List[Tuple[str, int]]) -> tf.data.Dataset:
        output_signature = (
            tf.TensorSpec(shape=(self.segmenter.segment_size,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
        
        return tf.data.Dataset.from_generator(
            lambda: self.generate_from_samples(samples),
            output_signature=output_signature
        )
