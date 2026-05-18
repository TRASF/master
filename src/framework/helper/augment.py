import tensorflow as tf
import numpy as np
import scipy.io.wavfile as scipy
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

class AudioAugmentor:
    """
    Handles on-the-fly audio augmentation using TensorFlow operations.
    """
    def __init__(self, config: Dict):
        self.aug_config = config['training']['augmentation']
        self.enabled = self.aug_config.get('enabled', False)
        self.apply_prob = self.aug_config.get('apply_prob', 1.0)
        
        self.p_time = self.aug_config.get('time_shift', {}).get('p', 1.0)
        self.p_pitch = self.aug_config.get('pitch_shift', {}).get('p', 1.0)
        self.p_gain = self.aug_config.get('global_gain', {}).get('p', 1.0)
        self.p_noise = self.aug_config.get('noise', {}).get('p', 1.0)
        
        self.sample_rate = config['audio']['sample_rate']
        self.segment_size = int(config['audio']['duration'] * self.sample_rate)
        
        if self.enabled:
            self._init_noise_bank()

    def _init_noise_bank(self):
        noise_config = self.aug_config.get('noise', {})
        noise_dirs = noise_config.get('noise_dirs', [])
        noise_paths = []
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for d in noise_dirs:
                p = Path(d)
                if not p.exists():
                    continue
                for f in p.glob('**/*.wav'):
                    try:
                        # Validation read
                        scipy.read(str(f), mmap=True)
                        noise_paths.append(str(f))
                    except Exception:
                        continue
        
        if noise_paths:
            self.noise_bank = tf.constant(noise_paths)
            snr_range = noise_config.get('snr_range', [-10, 20])
            self.snr_min = float(snr_range[0])
            self.snr_max = float(snr_range[1])
            print(f"Noise bank initialized with {len(noise_paths)} valid files.")
        else:
            self.noise_bank = None

    @tf.function
    def apply_time_shift(self, audio: tf.Tensor) -> tf.Tensor:
        max_shift = int(self.segment_size * self.aug_config['time_shift']['max_shift_pct'])
        shift = tf.random.uniform([], -max_shift, max_shift, dtype=tf.int32)
        return tf.roll(audio, shift, axis=0)

    @tf.function
    def apply_pitch_shift(self, audio: tf.Tensor) -> tf.Tensor:
        """Achieves pitch shift via speed perturbation (resampling)."""
        p_range = self.aug_config['pitch_shift']['range_pct']
        scale = tf.random.uniform([], 1.0 + p_range[0], 1.0 + p_range[1])
        new_size = tf.cast(tf.cast(self.segment_size, tf.float32) * scale, tf.int32)
        
        audio_reshaped = tf.reshape(audio, [1, -1, 1])
        resampled = tf.image.resize(audio_reshaped, [1, new_size], method='bilinear')
        resampled = tf.reshape(resampled, [-1])
        
        if new_size > self.segment_size:
            return resampled[:self.segment_size]
        else:
            return tf.pad(resampled, [[0, self.segment_size - new_size]])

    @tf.function
    def apply_gain(self, audio: tf.Tensor) -> tf.Tensor:
        gain_range = self.aug_config['global_gain']['range']
        gain = tf.random.uniform([], gain_range[0], gain_range[1])
        return audio * gain

    @tf.function
    def apply_noise(self, audio: tf.Tensor) -> tf.Tensor:
        if self.noise_bank is None:
            return audio
            
        idx = tf.random.uniform([], 0, tf.shape(self.noise_bank)[0], dtype=tf.int32)
        noise_path = tf.gather(self.noise_bank, idx)
        
        noise_binary = tf.io.read_file(noise_path)
        noise_audio, _ = tf.audio.decode_wav(noise_binary)
        noise_audio = tf.reshape(noise_audio, [-1])
        
        noise_len = tf.shape(noise_audio)[0]
        if noise_len > self.segment_size:
            start = tf.random.uniform([], 0, noise_len - self.segment_size, dtype=tf.int32)
            noise_seg = noise_audio[start : start + self.segment_size]
        else:
            noise_seg = tf.pad(noise_audio, [[0, self.segment_size - noise_len]])
            
        target_snr = tf.random.uniform([], self.snr_min, self.snr_max)
        
        rms_signal = tf.sqrt(tf.reduce_mean(tf.square(audio)) + 1e-9)
        rms_noise = tf.sqrt(tf.reduce_mean(tf.square(noise_seg)) + 1e-9)
        
        scalar = (rms_signal / rms_noise) / tf.pow(10.0, target_snr / 20.0)
        return audio + (noise_seg * scalar)

    @tf.function
    def apply(self, audio: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        if not self.enabled:
            return audio, label
            
        if tf.random.uniform([]) > self.apply_prob:
            return audio, label
            
        if tf.random.uniform([]) < self.p_time:
            audio = self.apply_time_shift(audio)
        if tf.random.uniform([]) < self.p_pitch:
            audio = self.apply_pitch_shift(audio)
        if tf.random.uniform([]) < self.p_gain:
            audio = self.apply_gain(audio)
        if tf.random.uniform([]) < self.p_noise:
            audio = self.apply_noise(audio)
            
        return tf.clip_by_value(audio, -1.0, 1.0), label
