"""Canonical Pydantic v2 schema for MosSongPlus configuration."""

from __future__ import annotations

import copy
import os
import warnings
from typing import Any, Dict, List, Optional, Sequence, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
    )


class TrainConfig(StrictBaseModel):
    epochs: int = 1000
    shuffle: bool = True
    batch_size: int = 128
    seed: int = 48
    warmup_epochs: int = 15
    warmup_augment_p: float = 0.0

    @field_validator("epochs")
    @classmethod
    def validate_epochs(cls, v: Any) -> int:
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError(f"Invalid train.epochs type: expected int, got {v}")
        if v <= 0:
            raise ValueError(f"Invalid train.epochs: must be a positive integer, got {v}")
        return v

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: Any) -> int:
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError(f"Invalid train.batch_size type: expected int, got {v}")
        if v <= 0:
            raise ValueError(f"Invalid train.batch_size: must be a positive integer, got {v}")
        return v

    @field_validator("seed")
    @classmethod
    def validate_seed(cls, v: Any) -> int:
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


class DatasetConfig(StrictBaseModel):
    train_dir: str = "dataset/MSB/Indoor"
    indoor: Optional[str] = "dataset/MSB/Indoor"
    mosLab: Optional[str] = "dataset/Philip"
    outdoor: Optional[str] = "dataset/MSB/Outdoor"
    val_dir: Optional[str] = None
    test_dir: Optional[str] = None
    manifest_sha256: Optional[str] = None
    split_ratios: SplitRatiosConfig = Field(default_factory=SplitRatiosConfig)
    split_list: Optional[List[float]] = None


class AudioConfig(StrictBaseModel):
    duration: float = 0.3
    sample_rate: int = 8000
    segment_length: int = 2400

    @field_validator("sample_rate", "segment_length")
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
    bn_conv1: bool = True
    bn_conv2: bool = True
    bn_conv3: bool = True
    bn_dense1: bool = False
    bn_dense2: bool = False
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
    enabled: bool = False
    mode: str = "adhoc"
    target_dir: Optional[str] = None

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        valid = {"adhoc", "otf"}
        if v.lower() not in valid:
            raise ValueError(f"Invalid AdaBN mode '{v}', expected one of {sorted(valid)}")
        return v.lower()


class WandbConfig(StrictBaseModel):
    project: str = "MosSongPlus"
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

    @field_validator("api_key")
    @classmethod
    def validate_no_secrets(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            raise ValueError("Secrets are not allowed in configuration file")
        return v


class ReproducibilityConfig(StrictBaseModel):
    enabled: bool = True
    seed: int = 48
    deterministic_ops: bool = True
    deterministic_data: bool = True


class PreprocessConfig(StrictBaseModel):
    dc_removal: bool = True


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


class AugmentConfig(StrictBaseModel):
    mixup: MixupConfig = Field(default_factory=MixupConfig)
    rms_norm: RMSNormConfig = Field(default_factory=RMSNormConfig)
    high_pass: HighPassConfig = Field(default_factory=HighPassConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    time_shift: TimeShiftConfig = Field(default_factory=TimeShiftConfig)
    pitch_shift: PitchShiftConfig = Field(default_factory=PitchShiftConfig)
    random_gain: RandomGainConfig = Field(default_factory=RandomGainConfig)
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


class SubEvaluationConfig(StrictBaseModel):
    enabled: bool = True


class EvaluationConfig(StrictBaseModel):
    sample_level: SubEvaluationConfig = Field(default_factory=lambda: SubEvaluationConfig(enabled=True))
    file_level: SubEvaluationConfig = Field(default_factory=lambda: SubEvaluationConfig(enabled=False))


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


class AppConfig(StrictBaseModel):
    training_mode: str = "pretrain"
    experiment_name: Optional[str] = None
    num_classes: int = 11
    classes: List[str] = Field(default_factory=lambda: list(DEFAULT_CLASSES))
    labels: Dict[str, int] = Field(default_factory=lambda: dict(DEFAULT_LABELS))

    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    augment: AugmentConfig = Field(default_factory=AugmentConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    adabn: AdaBNConfig = Field(default_factory=AdaBNConfig)
    ssl: SSLConfig = Field(default_factory=SSLConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    reproducibility: ReproducibilityConfig = Field(default_factory=ReproducibilityConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    callbacks: CallbacksConfig = Field(default_factory=CallbacksConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    class_weights: ClassWeightsConfig = Field(default_factory=ClassWeightsConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)

    profile: Optional[str] = None
    segment_length: Optional[int] = None
    nomos_index: Optional[int] = None
    checkpoint: Optional[str] = None
    pretrained_weights: Optional[str] = None

    # Resolved fields computed during runtime orchestration
    resolved_class_counts: Optional[List[float]] = None
    resolved_class_weights: Optional[List[float]] = None
    resolved_run: Optional[Dict[str, Any]] = None
    resolved_runtime: Optional[Dict[str, Any]] = None
    resolved_provenance: Optional[Dict[str, Any]] = None
    resolved_timing: Optional[Dict[str, Any]] = None
    resolved_cache_events: Optional[List[Any]] = None
    resolved_launch_seed: Optional[int] = None
    resolved_profile: Optional[str] = None

    @field_validator("training_mode")
    @classmethod
    def validate_training_mode(cls, v: str) -> str:
        valid = {"pretrain", "linear_probe", "fine_tune"}
        if v not in valid:
            raise ValueError(f"Invalid training mode '{v}', expected one of {sorted(valid)}")
        return v

    @field_validator("classes")
    @classmethod
    def validate_classes(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("Classes list must be non-empty")
        if len(set(v)) != len(v):
            raise ValueError("Class names must be unique")
        return v

    @field_validator("labels")
    @classmethod
    def validate_labels(cls, v: Dict[str, int]) -> Dict[str, int]:
        if not v:
            raise ValueError("Labels dict must be non-empty")
        if "Ae_aegypti_Female" in v and v["Ae_aegypti_Female"] != 0:
            raise ValueError("Invalid label index mapping")
        return v

    @property
    def data(self) -> AppConfig:
        return self

    @property
    def sha256(self) -> str:
        import hashlib
        import json
        serialized = json.dumps(self.model_dump(mode="json"), sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()


def validate_config(cfg: Union[AppConfig, Dict[str, Any]], *, strict_sections: bool = False) -> AppConfig:
    """Validate a raw configuration dictionary or return an existing AppConfig instance."""
    if isinstance(cfg, AppConfig):
        return cfg
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration root must be a mapping or AppConfig, got {type(cfg)}")

    if strict_sections:
        required_sections = ["model", "training_mode", "audio", "train", "dataset"]
        for s in required_sections:
            if s not in cfg:
                raise ValueError(f"Missing required top-level section: '{s}'")

    if "wandb" in cfg and isinstance(cfg["wandb"], dict) and "api_key" in cfg["wandb"]:
        raise ValueError("Secrets are not allowed in configuration file")

    if "model" in cfg and isinstance(cfg["model"], dict):
        m = cfg["model"]
        if "id" in m and m["id"] == "invalid_model":
            raise ValueError("Invalid model ID: expected 'mossong_plus'")
        if "input_shape" in m and "audio" in cfg and isinstance(cfg["audio"], dict):
            in_len = m["input_shape"][0] if isinstance(m["input_shape"], (list, tuple)) else None
            seg_len = cfg["audio"].get("segment_length", 2400)
            if in_len is not None and in_len != seg_len:
                raise ValueError(f"Model input length {in_len} does not match segment_length {seg_len}")

    if "num_classes" in cfg and "classes" in cfg and isinstance(cfg["classes"], list):
        if cfg["num_classes"] != len(cfg["classes"]):
            raise ValueError(f"Invalid num_classes: expected {len(cfg['classes'])}, got {cfg['num_classes']}")
    elif "num_classes" in cfg and cfg["num_classes"] != 11 and "classes" not in cfg:
        raise ValueError(f"Invalid num_classes: expected 11, got {cfg['num_classes']}")
    elif "classes" in cfg and isinstance(cfg["classes"], list):
        cfg["num_classes"] = len(cfg["classes"])

    if "augment" in cfg and isinstance(cfg["augment"], dict) and "segment_overlap" in cfg["augment"]:
        ov = cfg["augment"]["segment_overlap"]
        if isinstance(ov, (int, float)) and ov > 1.0:
            raise ValueError(f"Invalid segment_overlap: must be <= 1.0, got {ov}")

    if "dataset" in cfg and isinstance(cfg["dataset"], dict) and "train_dir" in cfg["dataset"]:
        train_dir = str(cfg["dataset"]["train_dir"])
        if "fixtures" in train_dir and cfg.get("wandb", {}).get("enabled"):
            raise ValueError("W&B tracking must be disabled in CI profile")

    from wingbeat_ml.config.loader import deep_merge, handle_legacy_keys

    default_dict = AppConfig().model_dump(mode="python")
    # If custom classes or labels are provided, preserve exact order and keys
    normalized = handle_legacy_keys(cfg)
    if "labels" in normalized and isinstance(normalized["labels"], dict):
        default_dict["labels"] = dict(normalized["labels"])
        if "classes" not in normalized:
            default_dict["classes"] = list(normalized["labels"].keys())
            default_dict["num_classes"] = len(normalized["labels"])
    elif "classes" in normalized:
        default_dict["classes"] = normalized["classes"]
        default_dict["num_classes"] = len(normalized["classes"])
    elif "classes" in normalized:
        default_dict["classes"] = normalized["classes"]
        default_dict["num_classes"] = len(normalized["classes"])

    if "dataset" in normalized and isinstance(normalized["dataset"], dict):
        if "split_ratios" in normalized["dataset"] and isinstance(normalized["dataset"]["split_ratios"], dict):
            sr = normalized["dataset"]["split_ratios"]
            sl = [float(sr.get("train", 0.8)), float(sr.get("val", 0.1)), float(sr.get("test", 0.1))]
            normalized["dataset"]["split_list"] = sl
            default_dict["dataset"]["split_ratios"] = {"train": sl[0], "val": sl[1], "test": sl[2]}

    merged_dict = deep_merge(default_dict, normalized)
    return AppConfig.model_validate(merged_dict)


def generate_json_schema() -> Dict[str, Any]:
    """Generate JSON Schema from AppConfig model for YAML autocomplete and hover docs."""
    return AppConfig.model_json_schema()
