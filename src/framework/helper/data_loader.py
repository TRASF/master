import os
from pathlib import Path
import numpy as np
from src.io.loader import FileLoader


class DataLoader:
    def __init__(self, dataset_dir: str, sample_rate: int = 8000, classes: list = None):
        self.dataset_dir = dataset_dir
        self.sample_rate = sample_rate 

        if os.path.isdir(self.dataset_dir):
            self.classes = classes if classes is not None else sorted(
                [d.name for d in Path(self.dataset_dir).iterdir() if d.is_dir()]
            )
        
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
            self.num_classes = len(self.classes)
        else:
            self.classes = []
            self.class_to_idx = {}
            self.num_classes = 0
    
    def gather_files(self, directory: Path = None):
        
        if directory is None:
            directory = Path(self.dataset_dir)
        else:
            directory = Path(directory)

        file_paths = []
        labels = []

        for cls_name in self.classes:
            cls_dir = directory / cls_name
            if not cls_dir.is_dir():
                continue

            for file_path in cls_dir.glob("*.wav"):
                file_paths.append(str(file_path))
                labels.append(self.class_to_idx[cls_name])

        return np.array(file_paths), np.array(labels)

    def segment_audio(self, audio: np.ndarray, segment_length: int):
        total_length = len(audio)
        if total_length <= segment_length:
            padding = segment_length - total_length
            audio = np.pad(audio, (0, padding), mode="constant")
            return audio

        start_idx = np.random.randint(0, total_length - segment_length + 1)
        return audio[start_idx : start_idx + segment_length]

    def load_file(self, file_path: str):
        loader = FileLoader(file_path, self.sample_rate)
        return loader.load()
