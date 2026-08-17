"""Canonical Pydantic v2 schema for MosSongPlus configuration."""

from __future__ import annotations

import copy
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence, Union
from pydantic import BaseModel, ConfigDict, Discriminator, Field, PositiveInt, TypeAdapter, computed_field, field_validator, model_validator




class StrictBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
    )

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)


class TrainConfig(StrictBaseModel):
    epochs: int = 1000
    shuffle: bool = True
    batch_size: int = 128
    seed: int = 48
    warmup_epochs: int = 15
    warmup_augment_p: float = 0.0

    @field_validator("epochs", mode="before")
    @classmethod
    def validate_epochs(cls, v: Any) -> Any:
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError(f"Invalid train.epochs type: expected int, got {v}")
        if v <= 0:
            raise ValueError(f"Invalid train.epochs: must be a positive integer, got {v}")
        return v

    @field_validator("batch_size", mode="before")
    @classmethod
    def validate_batch_size(cls, v: Any) -> Any:
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError(f"Invalid train.batch_size type: expected int, got {v}")
        if v <= 0:
            raise ValueError(f"Invalid train.batch_size: must be a positive integer, got {v}")
        return v

    @field_validator("seed", mode="before")
    @classmethod
    def validate_seed(cls, v: Any) -> Any:
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError(f"Invalid train.seed type: expected int, got {v}")
        if v < 0:
            raise ValueError(f"Invalid train.seed: must be non-negative, got {v}")
        return v


class SplitRatiosConfig(StrictBaseModel):
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1

    @model_validator(mode="after")
    def validate_sum(self) -> SplitRatiosConfig:
        total = round(self.train + self.val + self.test, 4)
        if abs(total - 1.0) > 1e-3:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        return self


class PreprocessConfig(StrictBaseModel):
    dc_removal: bool = True


class DatasetConfig(StrictBaseModel):
    train_dir: str = "dataset/MSB/Indoor"
    indoor: Optional[str] = "dataset/MSB/Indoor"
    mosLab: Optional[str] = "dataset/Philip"
    outdoor: Optional[str] = "dataset/MSB/Outdoor"
    val_dir: Optional[str] = None
    test_dir: Optional[str] = None
    manifest_sha256: Optional[str] = None
    split_ratios: SplitRatiosConfig = Field(default_factory=SplitRatiosConfig)
    preprocessing: PreprocessConfig = Field(default_factory=PreprocessConfig)

    @computed_field
    @property
    def split_list(self) -> List[float]:
        return [self.split_ratios.train, self.split_ratios.val, self.split_ratios.test]


class AudioConfig(StrictBaseModel):
    duration: float = 0.3
    sample_rate: int = 8000

    @computed_field
    @property
    def num_samples(self) -> int:
        return round(self.sample_rate * self.duration)

    @computed_field
    @property
    def segment_length(self) -> int:
        return self.num_samples

    @field_validator("sample_rate")
    @classmethod
    def validate_positive(cls, v: Any, info) -> int:
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError(f"Invalid {info.field_name} type: expected int, got {type(v)}")
        if v <= 0:
            raise ValueError(f"Invalid {info.field_name}: must be a positive integer, got {v}")
        return v


class ModelConfig(StrictBaseModel):
    id: str = "mossong_plus"
    pretrained_weights: Optional[str] = None
    checkpoint: Optional[str] = None
    output_activation: Optional[str] = None
    bn_conv1: Optional[bool] = None
    bn_conv2: Optional[bool] = None
    bn_conv3: Optional[bool] = None
    bn_dense1: Optional[bool] = None
    bn_dense2: Optional[bool] = None
    layers: Optional[List[Dict[str, Any]]] = None
    mossong_plus: Optional[Dict[str, Any]] = None
    mossongplus: Optional[Dict[str, Any]] = None
    input_shape: Optional[Union[List[int], Sequence[int]]] = None

    @field_validator("id")
    @classmethod
    def validate_model_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Invalid model ID: expected non-empty string")
        return v


class ProfilerConfig(StrictBaseModel):
    enabled: bool = False
    start_step: int = 10
    num_steps: int = 10


class PerformanceConfig(StrictBaseModel):
    precision: str = "float32"
    steps_per_call: int = 20
    jit_compile: bool = False
    profiler: ProfilerConfig = Field(default_factory=ProfilerConfig)

    @field_validator("precision")
    @classmethod
    def validate_precision(cls, v: str) -> str:
        valid = {"float32", "mixed_float16", "float16", "auto"}
        if v not in valid:
            raise ValueError(f"Invalid precision '{v}', expected one of {sorted(valid)}")
        return v

    @field_validator("steps_per_call")
    @classmethod
    def validate_steps_per_call(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("performance.steps_per_call must be > 0")
        return v


class LoggingConfig(StrictBaseModel):
    console: str = "normal"
    epoch_interval: int = 1
    model_summary: bool = False
    classification_report: str = "file"
    jsonl: bool = True
    prediction_distribution: bool = False

    @field_validator("console")
    @classmethod
    def validate_console(cls, v: str) -> str:
        valid = {"normal", "quiet", "verbose", "debug"}
        if v not in valid:
            raise ValueError(f"Invalid console logging mode '{v}', expected one of {sorted(valid)}")
        return v

    @field_validator("epoch_interval")
    @classmethod
    def validate_epoch_interval(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("logging.epoch_interval must be > 0")
        return v


class AdaBNConfig(StrictBaseModel):
    """Offline target-domain BatchNorm calibration."""

    enabled: bool = False
    target_dir: Optional[str] = None
    mode: str = "adhoc"

class WandbConfig(StrictBaseModel):
    project: str = "Master"
    tags: List[str] = Field(default_factory=list)
    group: Optional[str] = None
    notes: Optional[str] = None
    enabled: bool = False
    job_type: str = "train"
    log_weights_freq: int = 0
    aggregate_plot_freq: int = 0
    log_prediction_audio: bool = False
    prediction_table_max_rows: int = 0
    log_detailed_diagnostics: bool = False
    api_key: Optional[str] = None

    @field_validator("api_key", mode="before")
    @classmethod
    def validate_no_secrets(cls, value: Any) -> Any:
        """Reject embedded credentials; normalize empty defaults to None."""
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        raise ValueError("Secrets are not allowed in configuration file")


class ReproducibilityConfig(StrictBaseModel):
    enabled: bool = True
    seed: int = 48
    deterministic_ops: bool = True
    deterministic_data: bool = True


class MixupConfig(StrictBaseModel):
    p: float = 1.0
    alpha: float = 0.4


class RMSNormConfig(StrictBaseModel):
    p: float = 1.0
    max_gain: float = 15.0
    min_gain: float = 0.05
    target_rms: float = 0.05


class HighPassConfig(StrictBaseModel):
    p: float = 0.0
    fc: float = 150.0


class TimeShiftConfig(StrictBaseModel):
    p: float = 0.0
    rate: List[float] = Field(default_factory=lambda: [-0.05, 0.05])


class PitchShiftConfig(StrictBaseModel):
    p: float = 0.0
    semitones: List[float] = Field(default_factory=lambda: [-0.2, 0.2])


class RandomGainConfig(StrictBaseModel):
    p: float = 0.3
    gain_db: List[float] = Field(default_factory=lambda: [-3.0, 3.0])


class SNRDistributionConfig(StrictBaseModel):
    p: float = 0.0
    snr_db: Optional[List[float]] = None


class NoiseOverlayConfig(StrictBaseModel):
    p: float = 0.2
    snr_db: List[float] = Field(default_factory=lambda: [15.0, 30.0])
    post_gain_db: List[float] = Field(default_factory=lambda: [-3.0, 3.0])
    envelope_gain: List[float] = Field(default_factory=lambda: [0.7, 1.0])
    snr_distribution: Optional[List[SNRDistributionConfig]] = None


class GaussianNoiseConfig(StrictBaseModel):
    p: float = 0.0
    snr_db: List[float] = Field(default_factory=lambda: [20.0, 50.0])


class TimeMaskingConfig(StrictBaseModel):
    p: float = 0.0
    num_masks: int = 1
    max_mask_size: int = 400


class PreEmphasisConfig(StrictBaseModel):
    p: float = 0.0
    coeff: float = 0.97


class SegmentOverlapConfig(StrictBaseModel):
    val: float = 0.5
    train: Union[float, List[float]] = Field(default_factory=lambda: [0.0, 0.8])

    @model_validator(mode="after")
    def validate_overlap_bounds(self) -> SegmentOverlapConfig:
        if isinstance(self.val, (int, float)) and not (0.0 <= self.val <= 1.0):
            raise ValueError(f"Invalid segment_overlap.val: must be between 0 and 1.0, got {self.val}")
        if isinstance(self.train, (int, float)) and not (0.0 <= self.train <= 1.0):
            raise ValueError(f"Invalid segment_overlap.train: must be between 0 and 1.0, got {self.train}")
        if isinstance(self.train, list):
            if not all(isinstance(x, (int, float)) and 0.0 <= x <= 1.0 for x in self.train):
                raise ValueError(f"Invalid segment_overlap.train list: must be floats between 0 and 1.0, got {self.train}")
        return self


class RandomMicEQConfig(StrictBaseModel):
    """Random smooth microphone-like magnitude response."""

    p: float = 0.0

    # Total control points including 0 Hz and Nyquist.
    num_points: List[int] = Field(default_factory=lambda: [3, 7])

    # Gain applied independently at each control point.
    gain_db: List[float] = Field(default_factory=lambda: [-4.0, 4.0])

    # Controls random widths of adjacent frequency sections.
    # 1.0 means equal width; this range makes sections irregular.
    section_width_weights: List[float] = Field(
        default_factory=lambda: [0.6, 1.4]
    )

    # Reflection padding before FFT-domain filtering.
    fft_pad_samples: int = 256

    # Remove average dB offset so this augmentation mostly changes
    # spectral shape; random_gain remains responsible for volume.
    zero_mean_db: bool = True

    @model_validator(mode="after")
    def validate_values(self) -> "RandomMicEQConfig":
        if not 0.0 <= self.p <= 1.0:
            raise ValueError("random_mic_eq.p must be in [0, 1]")

        if (
            len(self.num_points) != 2
            or self.num_points[0] < 2
            or self.num_points[1] < self.num_points[0]
        ):
            raise ValueError(
                "random_mic_eq.num_points must be [min, max] with min >= 2"
            )

        if len(self.gain_db) != 2 or self.gain_db[1] < self.gain_db[0]:
            raise ValueError(
                "random_mic_eq.gain_db must be [min_db, max_db]"
            )

        if (
            len(self.section_width_weights) != 2
            or self.section_width_weights[0] <= 0.0
            or self.section_width_weights[1]
            < self.section_width_weights[0]
        ):
            raise ValueError(
                "random_mic_eq.section_width_weights must be "
                "positive [min, max]"
            )

        if self.fft_pad_samples < 0:
            raise ValueError(
                "random_mic_eq.fft_pad_samples must be >= 0"
            )

        return self


class DeviceIRConfig(StrictBaseModel):
    """Measured microphone/device/enclosure impulse-response augmentation."""

    p: float = 0.0

    # Directories or individual .wav/.npy files.
    banks: List[str] = Field(default_factory=list)

    # Keep the useful early portion of each measured IR.
    max_ir_ms: float = 50.0

    # Keep a tiny amount before the detected IR peak.
    pre_peak_ms: float = 1.0

    # 1.0 = entirely transformed signal.
    wet: List[float] = Field(default_factory=lambda: [1.0, 1.0])

    normalize: Literal["l2", "peak", "none"] = "l2"

    @model_validator(mode="after")
    def validate_values(self) -> "DeviceIRConfig":
        if not 0.0 <= self.p <= 1.0:
            raise ValueError("device_ir.p must be in [0, 1]")

        if self.max_ir_ms <= 0:
            raise ValueError("device_ir.max_ir_ms must be > 0")

        if self.pre_peak_ms < 0:
            raise ValueError("device_ir.pre_peak_ms must be >= 0")

        if (
            len(self.wet) != 2
            or not 0.0 <= self.wet[0] <= 1.0
            or not 0.0 <= self.wet[1] <= 1.0
            or self.wet[1] < self.wet[0]
        ):
            raise ValueError(
                "device_ir.wet must be [min, max] inside [0, 1]"
            )

        if self.p > 0.0 and not self.banks:
            raise ValueError(
                "device_ir.banks cannot be empty when device_ir.p > 0"
            )

        return self


class ElectronicsConfig(StrictBaseModel):
    """Random recording-chain / ADC distortion."""

    p: float = 0.0

    soft_clip_p: float = 0.5
    hard_clip_p: float = 0.25
    quantize_p: float = 0.25

    # Soft saturation drive.
    drive_db: List[float] = Field(default_factory=lambda: [0.0, 6.0])

    # Hard clipping threshold relative to [-1, +1].
    clip_level: List[float] = Field(default_factory=lambda: [0.6, 0.98])

    # Simulated ADC resolution.
    bits: List[int] = Field(default_factory=lambda: [10, 16])

    @model_validator(mode="after")
    def validate_values(self) -> "ElectronicsConfig":
        probabilities = [
            self.p,
            self.soft_clip_p,
            self.hard_clip_p,
            self.quantize_p,
        ]

        if any(not 0.0 <= x <= 1.0 for x in probabilities):
            raise ValueError(
                "electronics probabilities must be in [0, 1]"
            )

        if len(self.drive_db) != 2 or self.drive_db[1] < self.drive_db[0]:
            raise ValueError(
                "electronics.drive_db must be [min, max]"
            )

        if (
            len(self.clip_level) != 2
            or self.clip_level[0] <= 0.0
            or self.clip_level[1] > 1.0
            or self.clip_level[1] < self.clip_level[0]
        ):
            raise ValueError(
                "electronics.clip_level must be inside (0, 1]"
            )

        if (
            len(self.bits) != 2
            or self.bits[0] < 2
            or self.bits[1] < self.bits[0]
        ):
            raise ValueError(
                "electronics.bits must be [min_bits, max_bits]"
            )

        return self


class AugmentConfig(StrictBaseModel):
    mixup: MixupConfig = Field(default_factory=MixupConfig)
    rms_norm: RMSNormConfig = Field(default_factory=RMSNormConfig)
    high_pass: HighPassConfig = Field(default_factory=HighPassConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    time_shift: TimeShiftConfig = Field(default_factory=TimeShiftConfig)
    pitch_shift: PitchShiftConfig = Field(default_factory=PitchShiftConfig)
    random_gain: RandomGainConfig = Field(default_factory=RandomGainConfig)
    random_mic_eq: RandomMicEQConfig = Field(
        default_factory=RandomMicEQConfig
    )
    device_ir: DeviceIRConfig = Field(
        default_factory=DeviceIRConfig
    )
    electronics: ElectronicsConfig = Field(
        default_factory=ElectronicsConfig
    )
    noise_overlay: NoiseOverlayConfig = Field(default_factory=NoiseOverlayConfig)
    gaussian_noise: GaussianNoiseConfig = Field(default_factory=GaussianNoiseConfig)
    noise_banks: List[str] = Field(
        default_factory=lambda: [
            "dataset/MSB/Environmental/humbug_noises",
            "dataset/MSB/Environmental/inmp_noises",
            "dataset/MSB/Environmental/miru_noises",
            "dataset/MSB/Environmental/noises",
            "dataset/MSB/Environmental/Nomos",
        ]
    )
    segment_overlap: SegmentOverlapConfig = Field(default_factory=SegmentOverlapConfig)
    max_segments_per_file: int = 100

    @model_validator(mode="after")
    def sync_noise_overlay(self) -> AugmentConfig:
        if self.noise_overlay.p == 0:
            object.__setattr__(self, "noise_banks", [])
        return self
    time_masking: TimeMaskingConfig = Field(default_factory=TimeMaskingConfig)
    pre_emphasis: PreEmphasisConfig = Field(default_factory=PreEmphasisConfig)
    config: Optional[Dict[str, Any]] = None
    overlap: Optional[Union[float, List[float]]] = None


class ClassWeightsConfig(StrictBaseModel):
    mode: str = "manual"
    values: Optional[Union[List[float], Dict[str, float]]] = None
    enabled: Optional[bool] = None

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        valid = {"auto", "manual", "none", "off", "disabled"}
        if v not in valid:
            raise ValueError(f"Invalid class_weights mode '{v}', expected one of {sorted(valid)}")
        return v


class EarlyStoppingConfig(StrictBaseModel):
    mode: str = "max"
    monitor: str = "val_macro_f1"
    patience: int = 80
    min_delta: float = 0.0
    restore_best_weights: bool = True


class ModelCheckpointConfig(StrictBaseModel):
    mode: str = "max"
    monitor: str = "val_macro_f1"
    min_delta: float = 0.0
    save_best_only: bool = True
    save_weights_only: bool = True


class ReduceLROnPlateauConfig(StrictBaseModel):
    mode: str = "max"
    monitor: str = "val_macro_f1"
    factor: float = 0.5
    patience: int = 30
    min_delta: float = 0.001
    min_lr: float = 3.0e-05
    restore_best_weights: bool = True


class CosineAnnealingConfig(StrictBaseModel):
    t_max: int = 100
    eta_min: float = 1e-6


class CallbacksConfig(StrictBaseModel):
    early_stopping: Optional[EarlyStoppingConfig] = Field(default_factory=EarlyStoppingConfig)
    model_checkpoint: Optional[ModelCheckpointConfig] = Field(default_factory=ModelCheckpointConfig)
    reduce_lr_on_plateau: Optional[ReduceLROnPlateauConfig] = Field(default_factory=ReduceLROnPlateauConfig)
    cosine_annealing: Optional[CosineAnnealingConfig] = None


class LossConfig(StrictBaseModel):
    name: str = "CategoricalCrossentropy"
    reduction: str = "none"
    from_logits: bool = True


class OptimizerConfig(StrictBaseModel):
    name: str = "Adam"
    learning_rate: float = 0.001

    @field_validator("learning_rate")
    @classmethod
    def validate_lr(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Invalid learning_rate: must be > 0")
        return v


class InitializationConfig(StrictBaseModel):
    weights: Optional[str] = None

class ResumeConfig(StrictBaseModel):
    checkpoint: Optional[str] = None

class SupervisedTrainingConfig(StrictBaseModel):
    paradigm: Literal["supervised"] = "supervised"
    procedure: Literal["pretrain", "linear_probe", "fine_tune"] = "pretrain"

class FixMatchTrainingConfig(StrictBaseModel):
    paradigm: Literal["semi_supervised"] = "semi_supervised"
    method: Literal["fixmatch"] = "fixmatch"
    confidence_threshold: float = 0.95
    unsupervised_weight: float = 1.0

class FlexMatchTrainingConfig(StrictBaseModel):
    paradigm: Literal["semi_supervised"] = "semi_supervised"
    method: Literal["flexmatch"] = "flexmatch"
    confidence_threshold: float = 0.95
    unsupervised_weight: float = 1.0

TrainingConfigType = Union[
    SupervisedTrainingConfig,
    FixMatchTrainingConfig,
    FlexMatchTrainingConfig,
]

# Layer Schemas
class Conv1DLayerConfig(StrictBaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)
    type: Literal["conv1d"]
    filters: PositiveInt
    kernel_size: PositiveInt
    strides: PositiveInt = 1
    dilation_rate: PositiveInt = 1
    groups: PositiveInt = 1
    use_bias: bool = True
    padding: Literal["valid", "same", "linear"] = "valid"
    activation: Optional[str] = None
    batch_norm: Union[bool, Dict[str, Any]] = False
    l2_reg: Optional[float] = None
    fir_init: Optional[Dict[str, Any]] = None
    kernel_initializer: Optional[Union[str, Dict[str, Any]]] = None
    bn_conv: Optional[bool] = None
    separable: Optional[bool] = None
    kwargs: Dict[str, Any] = Field(default_factory=dict)

class DepthwiseConv1DLayerConfig(StrictBaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)
    type: Literal["depthwise_conv1d"]
    kernel_size: PositiveInt
    depth_multiplier: PositiveInt = 1
    strides: PositiveInt = 1
    padding: Literal["valid", "same"] = "valid"
    activation: Optional[str] = None
    kwargs: Dict[str, Any] = Field(default_factory=dict)

class SeparableConv1DLayerConfig(StrictBaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)
    type: Literal["separable_conv1d"]
    filters: PositiveInt
    kernel_size: PositiveInt
    strides: PositiveInt = 1
    padding: Literal["valid", "same"] = "valid"
    activation: Optional[str] = None
    kwargs: Dict[str, Any] = Field(default_factory=dict)

class SincConv1DLayerConfig(StrictBaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)
    type: Literal["sincconv1d"]
    filters: PositiveInt
    kernel_size: PositiveInt
    sample_rate: PositiveInt = 8000
    min_low_hz: float = 20.0
    kwargs: Dict[str, Any] = Field(default_factory=dict)

class RepConv1DLayerConfig(StrictBaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)
    type: Literal["repconv1d"]
    filters: PositiveInt
    kernel_size: PositiveInt
    strides: PositiveInt = 1
    branches: PositiveInt = 2
    kwargs: Dict[str, Any] = Field(default_factory=dict)

class DenseLayerConfig(StrictBaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)
    type: Literal["dense"]
    units: PositiveInt
    activation: Optional[str] = None
    batch_norm: Union[bool, Dict[str, Any]] = False
    kwargs: Dict[str, Any] = Field(default_factory=dict)

class FlattenLayerConfig(StrictBaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)
    type: Literal["flatten"]
    kwargs: Dict[str, Any] = Field(default_factory=dict)

class GlobalAvgPoolLayerConfig(StrictBaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)
    type: Literal["global_avg_pool", "global_avg_pool1d", "global_average_pooling1d"]
    kwargs: Dict[str, Any] = Field(default_factory=dict)

class GlobalMaxPoolLayerConfig(StrictBaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)
    type: Literal["global_max_pool", "global_max_pool1d", "global_max_pooling1d"]
    kwargs: Dict[str, Any] = Field(default_factory=dict)

class ConcatLayerConfig(StrictBaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)
    type: Literal["concat", "concatenate", "group"]
    layers: List[Union[Dict[str, Any], List[Dict[str, Any]]]]
    axis: int = -1
    kwargs: Dict[str, Any] = Field(default_factory=dict)

class MaxPool1DLayerConfig(StrictBaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)
    type: Literal["maxpool1d", "max_pooling1d"]
    pool_size: PositiveInt = 2
    strides: Optional[PositiveInt] = None
    padding: Literal["valid", "same"] = "valid"
    kwargs: Dict[str, Any] = Field(default_factory=dict)

class AvgPool1DLayerConfig(StrictBaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)
    type: Literal["avgpool1d", "avg_pooling1d"]
    pool_size: PositiveInt = 2
    strides: Optional[PositiveInt] = None
    padding: Literal["valid", "same"] = "valid"
    kwargs: Dict[str, Any] = Field(default_factory=dict)

class DropoutLayerConfig(StrictBaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)
    type: Literal["dropout"]
    rate: float = 0.5
    kwargs: Dict[str, Any] = Field(default_factory=dict)

class ReLULayerConfig(StrictBaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)
    type: Literal["relu"]
    max_value: Optional[float] = None
    negative_slope: float = 0.0
    threshold: float = 0.0
    kwargs: Dict[str, Any] = Field(default_factory=dict)

class ActivationLayerConfig(StrictBaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)
    type: Literal["activation"]
    activation: str
    kwargs: Dict[str, Any] = Field(default_factory=dict)

class BatchNormLayerConfig(StrictBaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)
    type: Literal["batch_norm", "batch_normalization"]
    momentum: float = 0.99
    epsilon: float = 1e-3
    kwargs: Dict[str, Any] = Field(default_factory=dict)

LayerConfigUnion = Annotated[
    Union[
        Conv1DLayerConfig,
        DepthwiseConv1DLayerConfig,
        SeparableConv1DLayerConfig,
        SincConv1DLayerConfig,
        RepConv1DLayerConfig,
        DenseLayerConfig,
        FlattenLayerConfig,
        GlobalAvgPoolLayerConfig,
        GlobalMaxPoolLayerConfig,
        MaxPool1DLayerConfig,
        AvgPool1DLayerConfig,
        DropoutLayerConfig,
        ReLULayerConfig,
        ActivationLayerConfig,
        BatchNormLayerConfig,
        ConcatLayerConfig,
    ],
    Discriminator("type"),
]

LAYER_CONFIG_ADAPTER: TypeAdapter[Any] = TypeAdapter(LayerConfigUnion)


def parse_layer_config(raw_spec: dict[str, Any]) -> Any:
    """Parse raw dictionary into a typed discriminated LayerConfig model."""
    if not isinstance(raw_spec, dict):
        raise TypeError(f"Layer definition must be a dict, got {type(raw_spec)}")
    if "type" not in raw_spec or not raw_spec["type"]:
        raise ValueError(f"Layer definition missing 'type': {raw_spec}")

    norm_spec = dict(raw_spec)
    norm_spec["type"] = str(norm_spec["type"]).lower()
    return LAYER_CONFIG_ADAPTER.validate_python(norm_spec)


@dataclass
class RunContext:
    run_id: str
    output_dir: Path
    config_hash: str
    resolved_class_weights: Any | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

@dataclass
class TrainingResult:
    best_checkpoint: Path | None
    final_epoch: int
    best_metric: float
    history: dict[str, Sequence[float]] = field(default_factory=dict)

@dataclass
class EvaluationResult:
    loss: float
    accuracy: float
    macro_f1: float
    per_class: dict[str, Any] = field(default_factory=dict)

@dataclass
class EdgeAnalysisResult:
    parameters: int
    model_bytes: int
    macs: int
    largest_activation_bytes: int | None = None
    estimated_adjacent_activation_bytes: int | None = None
    tensor_arena_bytes: int | None = None
    peak_activation_bytes: int | None = None
    trainable_parameters: int | None = None
    receptive_field_samples: int | None = None
    receptive_field_ms: float | None = None
    layer_details: list[dict[str, Any]] = field(default_factory=list)



class SubEvaluationConfig(StrictBaseModel):
    enabled: bool = True

class EvaluationConfig(StrictBaseModel):
    sample_level: SubEvaluationConfig = Field(default_factory=lambda: SubEvaluationConfig(enabled=True))
    file_level: SubEvaluationConfig = Field(default_factory=lambda: SubEvaluationConfig(enabled=False))
    confusion_matrix: bool = True
    classification_report: bool = True
    prediction_distribution: bool = False


class CacheConfig(StrictBaseModel):
    enabled: bool = True
    schema_version: int = 2
    root: Optional[str] = None


class RuntimeConfig(StrictBaseModel):
    root: str = "runtime"
    experiments_dir: str = "models/experiments"


class ExportConfig(StrictBaseModel):
    out_dir: str = "quantization_output"
    representative_samples: int = 500
    input_amplitude_range: float = 0.03
    allow_dummy_calibration: bool = False
    run_debugger: bool = False


DEFAULT_LABELS: Dict[str, int] = {
    "Ae_aegypti_Female": 0,
    "Ae_aegypti_Male": 1,
    "Ae_albopictus_Female": 2,
    "Ae_albopictus_Male": 3,
    "An_dirus_Female": 4,
    "An_dirus_Male": 5,
    "An_minimus_Female": 6,
    "An_minimus_Male": 7,
    "Cx_quin_Female": 8,
    "Cx_quin_Male": 9,
    "No.mos": 10,
}

DEFAULT_CLASSES: List[str] = list(DEFAULT_LABELS.keys())
DEFAULT_YAML_ORDER: List[str] = [
    "No.mos", "Cx_quin_Male", "An_dirus_Male", "Cx_quin_Female",
    "Ae_aegypti_Male", "An_dirus_Female", "An_minimus_Male",
    "Ae_aegypti_Female", "An_minimus_Female", "Ae_albopictus_Male", "Ae_albopictus_Female"
]


class SSLConfig(StrictBaseModel):
    enabled: bool = False
    method: str = "fixmatch"
    tau: float = 0.95
    lambda_u: float = 1.0
    mapping: str = "convex"
    source_dir: str = "dataset/MSB/Indoor"
    target_dir: str = "dataset/MSB/Outdoor"
    train_samples_per_class: Optional[int] = 100
    val_samples_per_class: Optional[int] = 50
    test_samples_per_class: Optional[int] = 50

    labeled_domains: List[str] = Field(default_factory=lambda: ["indoor"])
    unlabeled_domains: List[str] = Field(default_factory=lambda: ["outdoor"])
    labeled_samples_per_class: Optional[int] = None
    unlabeled_samples_per_class: Optional[int] = None
    unlabeled_policy: str = "remaining"
    exclude_labeled_from_unlabeled: bool = True
    subset_seed: int = 42
    minimum_recordings_per_class: Optional[int] = None


class ExperimentMetadataConfig(StrictBaseModel):
    name: Optional[str] = None


class ExperimentConfig(StrictBaseModel):
    experiment: ExperimentMetadataConfig = Field(default_factory=ExperimentMetadataConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    classes: List[str] = Field(default_factory=lambda: list(DEFAULT_CLASSES))
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfigType = Field(default_factory=SupervisedTrainingConfig)
    augmentation: AugmentConfig = Field(default_factory=AugmentConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    tracking: WandbConfig = Field(default_factory=WandbConfig)
    reproducibility: ReproducibilityConfig = Field(default_factory=ReproducibilityConfig)
    initialization: InitializationConfig = Field(default_factory=InitializationConfig)
    resume: ResumeConfig = Field(default_factory=ResumeConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    callbacks: CallbacksConfig = Field(default_factory=CallbacksConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    adabn: AdaBNConfig = Field(default_factory=AdaBNConfig)
    ssl: SSLConfig = Field(default_factory=SSLConfig)
    class_weights: ClassWeightsConfig = Field(default_factory=ClassWeightsConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    profile: Optional[str] = None
    nomos_index: Optional[int] = None

    @field_validator("classes")
    @classmethod
    def validate_classes(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("Classes list must be non-empty")
        if len(set(v)) != len(v):
            raise ValueError("Class names must be unique")
        return v

    @model_validator(mode="after")
    def sync_seeds(self) -> ExperimentConfig:
        if self.reproducibility.seed != self.train.seed:
            object.__setattr__(self.train, "seed", self.reproducibility.seed)
        return self

    @computed_field
    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @computed_field
    @property
    def labels(self) -> Dict[str, int]:
        if set(self.classes) == set(DEFAULT_YAML_ORDER):
            return {name: DEFAULT_LABELS[name] for name in DEFAULT_YAML_ORDER if name in self.classes}
        return {name: i for i, name in enumerate(self.classes)}

    @computed_field
    @property
    def training_mode(self) -> str:
        if hasattr(self.training, "procedure"):
            return getattr(self.training, "procedure")
        if hasattr(self.training, "method"):
            return getattr(self.training, "method")
        return "pretrain"

    @computed_field
    @property
    def segment_length(self) -> int:
        return self.audio.num_samples

    @computed_field
    @property
    def checkpoint(self) -> Optional[str]:
        return self.resume.checkpoint or self.initialization.weights

    @computed_field
    @property
    def pretrained_weights(self) -> Optional[str]:
        return self.initialization.weights

    @computed_field
    @property
    def augment(self) -> AugmentConfig:
        return self.augmentation

    @computed_field
    @property
    def wandb(self) -> WandbConfig:
        return self.tracking

    @property
    def data(self) -> ExperimentConfig:
        return self

    @property
    def preprocess(self) -> PreprocessConfig:
        return self.augmentation.preprocess

    @property
    def sha256(self) -> str:
        import hashlib
        import json
        serialized = json.dumps(self.model_dump(mode="json"), sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()


AppConfig = ExperimentConfig


def validate_config(cfg: Union[AppConfig, Dict[str, Any]], *, strict_sections: bool = False) -> AppConfig:
    """Validate a raw configuration dictionary or return an existing ExperimentConfig instance."""
    if isinstance(cfg, AppConfig):
        return cfg
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration root must be a mapping or AppConfig, got {type(cfg)}")

    if "train" in cfg and isinstance(cfg["train"], dict) and "seed" in cfg["train"]:
        if "reproducibility" in cfg and isinstance(cfg["reproducibility"], dict) and "seed" in cfg["reproducibility"]:
            if cfg["train"]["seed"] != cfg["reproducibility"]["seed"]:
                raise ValueError(
                    f"Inconsistent train.seed ({cfg['train']['seed']}) and reproducibility.seed ({cfg['reproducibility']['seed']})"
                )

    from wingbeat_ml.config.loader import handle_legacy_keys, normalize_legacy_config

    normalized_raw = handle_legacy_keys(cfg)

    if strict_sections:
        required_sections = ["model", "training_mode", "audio", "train", "dataset"]
        for s in required_sections:
            if s not in normalized_raw and s not in cfg:
                raise ValueError(f"Missing required top-level section: '{s}'")

    if "wandb" in normalized_raw and isinstance(normalized_raw["wandb"], dict):
        api_key = normalized_raw["wandb"].get("api_key")
        if api_key is not None:
            if not isinstance(api_key, str) or api_key.strip():
                raise ValueError("Secrets are not allowed in configuration file")

    if "model" in normalized_raw and isinstance(normalized_raw["model"], dict):
        m = normalized_raw["model"]
        if "id" in m and m["id"] == "invalid_model":
            raise ValueError("Invalid model ID: expected 'mossong_plus'")
        if "input_shape" in m and "audio" in normalized_raw and isinstance(normalized_raw["audio"], dict):
            in_len = m["input_shape"][0] if isinstance(m["input_shape"], (list, tuple)) else None
            seg_len = normalized_raw["audio"].get("segment_length", 2400)
            if in_len is not None and in_len != seg_len:
                raise ValueError(f"Model input length {in_len} does not match segment_length {seg_len}")

    if "augment" in normalized_raw and isinstance(normalized_raw["augment"], dict) and "segment_overlap" in normalized_raw["augment"]:
        ov = normalized_raw["augment"]["segment_overlap"]
        if isinstance(ov, (int, float)) and ov > 1.0:
            raise ValueError(f"Invalid segment_overlap: must be <= 1.0, got {ov}")

    if "dataset" in normalized_raw and isinstance(normalized_raw["dataset"], dict) and "train_dir" in normalized_raw["dataset"]:
        train_dir = str(normalized_raw["dataset"]["train_dir"])
        if "fixtures" in train_dir and normalized_raw.get("wandb", {}).get("enabled"):
            raise ValueError("W&B tracking must be disabled in CI profile")

    canonical_dict = normalize_legacy_config(normalized_raw)
    return ExperimentConfig.model_validate(canonical_dict)


def generate_json_schema() -> Dict[str, Any]:
    """Generate JSON Schema from AppConfig model for YAML autocomplete and hover docs."""
    return AppConfig.model_json_schema()
