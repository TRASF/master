import os
from pathlib import Path
import numpy as np
from wingbeat_ml.data.audio import FileLoader


class DataLoader:
    def __init__(
        self,
        dataset_dir: str,
        sample_rate: int = 8000,
        classes: list = None,
        file_ext=(".npy", ".wav"),
        labels_dict: dict = None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.sample_rate = sample_rate
        if isinstance(file_ext, str):
            self.file_exts = (file_ext,)
        else:
            self.file_exts = tuple(file_ext)

        if os.path.isdir(self.dataset_dir):
            if labels_dict:
                self.classes = list(labels_dict.keys())
                self.class_to_idx = labels_dict
                self.num_classes = max(labels_dict.values()) + 1
            else:
                self.classes = classes if classes is not None else sorted(
                    [d.name for d in self.dataset_dir.iterdir() if d.is_dir()]
                )

                self.class_to_idx = {
                    cls_name: idx for idx, cls_name in enumerate(self.classes)
                }
                self.num_classes = len(self.classes)
        else:
            self.classes = []
            self.class_to_idx = {}
            self.num_classes = 0

    def gather_files(self, directory: Path = None):
        directory = Path(directory) if directory is not None else self.dataset_dir

        file_paths = []
        labels = []

        for cls_name in self.classes:
            cls_dir = directory / cls_name

            if not cls_dir.is_dir():
                print(f"Warning: class directory not found: {cls_dir}")
                continue

            class_files = []
            for file_ext in self.file_exts:
                class_files.extend(cls_dir.glob(f"*{file_ext}"))

            for file_path in sorted(set(class_files)):
                file_paths.append(str(file_path))
                labels.append(self.class_to_idx[cls_name])

        return (
            np.array(file_paths, dtype=str),
            np.array(labels, dtype=np.int32),
        )

    def load_file(self, file_path: str):
        file_path = str(file_path)

        if file_path.endswith(".npy"):
            data = np.load(file_path)
            return data.astype(np.float32, copy=False)

        if file_path.endswith(".wav"):
            loader = FileLoader(file_path, self.sample_rate)
            return loader.load().astype(np.float32, copy=False)

        raise ValueError(f"Unsupported file type: {file_path}")


# New code should use this clearer name.
DatasetFileLoader = DataLoader

__all__ = ["DataLoader", "DatasetFileLoader"]
