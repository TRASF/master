import tensorflow as tf
import os
import random
import yaml
from collections import defaultdict
from src.framework.helper.data_loader import AudioDataLoader
from src.framework.helper.augment import AudioAugmentor
from typing import Optional, Dict, List, Tuple

class SupervisedDataset:
    """
    Handles supervised dataset generation, persistence, and optimized loading.
    """
    def __init__(self, data_loader: Optional[AudioDataLoader] = None, config_path: str = 'configs/defaults.yaml'):
        self.data_loader = data_loader if data_loader else AudioDataLoader(config_path)
        self.segment_size = self.data_loader.segmenter.segment_size
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.augmentor = AudioAugmentor(self.config)

    def _split_samples(self, samples: List[Tuple[str, int]], split_ratios: Dict[str, float]) -> Dict[str, List[Tuple[str, int]]]:
        seed = self.config.get('seed', 42)
        rng = random.Random(seed)
        
        # Group by label
        grouped_samples = defaultdict(list)
        for sample in samples:
            grouped_samples[sample[1]].append(sample)
            
        split_names = list(split_ratios.keys())
        splits = {name: [] for name in split_names}
        
        for label_samples in grouped_samples.values():
            rng.shuffle(label_samples)
            total_samples = len(label_samples)
            start_idx = 0
            
            for split_name, ratio in split_ratios.items():
                if split_name == split_names[-1]:
                    num_files = total_samples - start_idx
                else:
                    num_files = int(total_samples * ratio)
                
                end_idx = start_idx + num_files
                splits[split_name].extend(label_samples[start_idx:end_idx])
                start_idx = end_idx
                
        # Shuffle final splits
        for split_samples in splits.values():
            rng.shuffle(split_samples)
            
        return splits

    def _get_raw_samples(self, dataset_selection: Optional[List[str]] = None) -> List[Tuple[str, int]]:
        """
        Collect samples from configured directories.
        Selection can be specific subsets like 'MSB/indoor' or parents like 'MSB'.
        """
        all_samples = []
        dataset_config = self.config.get('dataset', {})
        selection = [s.lower() for s in dataset_selection] if dataset_selection else None

        def is_selected(key_path: str) -> bool:
            if not selection: 
                return True
            kp = key_path.lower()
            # Selected if kp matches any item in selection, or is a child/parent of it
            for s in selection:
                if kp == s or kp.startswith(s + "/") or s.startswith(kp + "/"):
                    return True
            return False

        # Process MSB
        msb_config = dataset_config.get('MSB', [])
        for item in msb_config:
            for key, path in item.items():
                if key in ('indoor', 'outdoor'):
                    if is_selected(f"MSB/{key}"):
                        print(f"Parsing raw samples from MSB/{key}: {path}...")
                        all_samples.extend(self.data_loader.get_samples(path))
                elif key == 'environmental' and isinstance(path, list):
                    if is_selected("MSB/environmental"):
                        for env_item in path:
                            for env_key, env_path in env_item.items():
                                if is_selected(f"MSB/environmental/{env_key}"):
                                    print(f"Parsing raw samples from MSB/environmental/{env_key}: {env_path}...")
                                    all_samples.extend(self.data_loader.get_samples(env_path))

        # Process Philip
        philip_config = dataset_config.get('Philip', [])
        if isinstance(philip_config, list):
            for item in philip_config:
                for key, path in item.items():
                    if is_selected(f"Philip/{key}"):
                        print(f"Parsing raw samples from Philip/{key}: {path}...")
                        all_samples.extend(self.data_loader.get_samples(path))
        elif isinstance(philip_config, dict):
            for key, path in philip_config.items():
                if is_selected(f"Philip/{key}"):
                    print(f"Parsing raw samples from Philip/{key}: {path}...")
                    all_samples.extend(self.data_loader.get_samples(path))

        return all_samples

    def generate_and_save(self, raw_data_path: str, save_path: str, 
                          split_ratios: Optional[Dict[str, float]] = None,
                          dataset_selection: Optional[List[str]] = None):
        if not split_ratios:
            split_ratios = self.config.get('dataset', {}).get('split_ratios')
            
        samples = self._get_raw_samples(dataset_selection)
        print(f"Total samples collected for saving: {len(samples)}")
        if not samples:
            print("WARNING: No samples found for the given selection.")
            return

        split_samples = self._split_samples(samples, split_ratios)
        splits = {}
        for split_name, samples_for_split in split_samples.items():
            splits[split_name] = self.data_loader.prepare_segments(samples_for_split)
            
        self.save_splits(splits, save_path)

    def get_splits(self, raw_data_path: str, split_ratios: Optional[Dict[str, float]] = None,
                   dataset_selection: Optional[List[str]] = None) -> Dict[str, tf.data.Dataset]:
        samples = self._get_raw_samples(dataset_selection)
        
        if not split_ratios:
            return {'all': self.data_loader.prepare_segments(samples)}
            
        split_samples = self._split_samples(samples, split_ratios)
        splits = {}
        for split_name, samples_for_split in split_samples.items():
            splits[split_name] = self.data_loader.prepare_segments(samples_for_split)
            
        return splits

    def save_splits(self, splits: Dict[str, tf.data.Dataset], save_path: str):
        for name, ds in splits.items():
            path = os.path.join(save_path, name) if name != 'all' else save_path
            print(f"Saving split '{name}' to {path}...")
            ds.save(path)
        print(f"Successfully saved {len(splits)} splits to {save_path}.")

    def load_from_raw(self, raw_data_path: str, batch_size: int, split: str = 'train', 
                      shuffle: bool = True, dataset_selection: Optional[List[str]] = None) -> tf.data.Dataset:
        split_ratios = self.config.get('dataset', {}).get('split_ratios')
        samples = self._get_raw_samples(dataset_selection)
        split_samples = self._split_samples(samples, split_ratios)
        target_samples = split_samples.get(split, [])
        
        dataset = self.data_loader.prepare_segments(target_samples)
        return self._apply_pipeline(dataset, batch_size, shuffle)

    def load_for_training(self, save_path: str, batch_size: int, split: Optional[str] = None, shuffle: bool = True) -> tf.data.Dataset:
        load_path = os.path.join(save_path, split) if split else save_path
        dataset = tf.data.Dataset.load(load_path)
        return self._apply_pipeline(dataset, batch_size, shuffle)

    def _apply_pipeline(self, dataset: tf.data.Dataset, batch_size: int, shuffle: bool) -> tf.data.Dataset:
        def process_audio(audio, label):
            if self.augmentor.enabled and shuffle:
                audio, label = self.augmentor.apply(audio, label)
            
            # Mean subtraction
            audio = audio - tf.reduce_mean(audio)
            return audio, label

        dataset = dataset.map(process_audio, num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
            
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
