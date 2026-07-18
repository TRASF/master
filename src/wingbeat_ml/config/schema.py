def validate_config(cfg: dict) -> None:
    # 1. Root structure check
    if not isinstance(cfg, dict):
        raise ValueError("Configuration root must be a YAML mapping (dictionary)")

    required_sections = ["model", "training_mode", "audio", "train", "dataset"]
    for section in required_sections:
        if section not in cfg:
            raise ValueError(f"Missing required top-level section: '{section}'")
        if section != "training_mode" and not isinstance(cfg[section], dict):
            raise ValueError(f"Top-level section '{section}' must be a dictionary")

    # 2. Experiment / Training Mode
    mode = cfg.get("training_mode")
    valid_modes = ["pretrain", "linear_probe", "fine_tune"]
    if mode not in valid_modes:
        raise ValueError(f"Invalid training mode: expected one of {valid_modes}, got {repr(mode)}")

    exp_name = cfg.get("experiment_name")
    if exp_name is not None:
        if not isinstance(exp_name, str) or not exp_name.strip():
            raise ValueError("experiment_name must be a non-empty string")

    # 3. Seed validation
    train_section = cfg.get("train", {})
    if isinstance(train_section, dict):
        seed = train_section.get("seed")
        if seed is not None:
            if not isinstance(seed, int) or seed < 0:
                raise ValueError(f"Invalid train.seed type: expected non-negative int, got {repr(seed)}")

    repro = cfg.get("reproducibility", {})
    if isinstance(repro, dict):
        repro_seed = repro.get("seed")
        if repro_seed is not None:
            if not isinstance(repro_seed, int) or repro_seed < 0:
                raise ValueError(f"Invalid reproducibility.seed type: expected non-negative int, got {repr(repro_seed)}")

    # 4. Data/Audio validation
    audio = cfg.get("audio", {})
    sample_rate = audio.get("sample_rate")
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError(f"Invalid sample_rate: must be a positive integer, got {repr(sample_rate)}")

    segment_length = audio.get("segment_length") or cfg.get("segment_length")
    if segment_length is not None:
        if not isinstance(segment_length, int) or segment_length <= 0:
            raise ValueError(f"Invalid segment_length: must be a positive integer, got {repr(segment_length)}")

    # Overlap range check
    augment = cfg.get("augment", {})
    overlap = augment.get("segment_overlap") if isinstance(augment, dict) else None
    if overlap is not None:
        if isinstance(overlap, dict):
            val = overlap.get("val")
            if val is not None:
                if not isinstance(val, (int, float)) or not (0.0 <= val <= 1.0):
                    raise ValueError(f"Invalid segment_overlap.val: must be float between 0 and 1.0, got {repr(val)}")
            train = overlap.get("train")
            if train is not None:
                if not isinstance(train, list) or not all(isinstance(x, (int, float)) and 0.0 <= x <= 1.0 for x in train):
                    raise ValueError(f"Invalid segment_overlap.train: must be list of floats between 0 and 1.0, got {repr(train)}")
        elif not isinstance(overlap, (int, float)) or not (0.0 <= overlap <= 1.0):
            raise ValueError(f"Invalid segment_overlap: must be float between 0 and 1.0, got {repr(overlap)}")

    # Class list validation
    classes = cfg.get("classes")
    labels = cfg.get("labels")
    if classes is not None:
        if not isinstance(classes, list) or len(classes) == 0:
            raise ValueError("Classes list must be a non-empty list")
        if len(set(classes)) != len(classes):
            raise ValueError("Class names must be unique")

    if labels is not None:
        if not isinstance(labels, dict) or len(labels) == 0:
            raise ValueError("Labels must be a non-empty dictionary")
        # Validate canonical indices
        expected_indices = {
            "Ae_aegypti_Female": 0, "Ae_aegypti_F": 0,
            "Ae_aegypti_Male": 1, "Ae_aegypti_M": 1,
            "Ae_albopictus_Female": 2, "Ae_albopictus_F": 2,
            "Ae_albopictus_Male": 3, "Ae_albopictus_M": 3,
            "An_dirus_Female": 4, "An_dirus_F": 4,
            "An_dirus_Male": 5, "An_dirus_M": 5,
            "An_minimus_Female": 6, "An_minimus_F": 6,
            "An_minimus_Male": 7, "An_minimus_M": 7,
            "Cx_quin_Female": 8, "Cx_quin_F": 8,
            "Cx_quin_Male": 9, "Cx_quin_M": 9,
            "No.mos": 10, "No.Mos": 10
        }
        for name, idx in labels.items():
            if name in expected_indices:
                if idx != expected_indices[name]:
                    raise ValueError(f"Invalid label index for '{name}': expected {expected_indices[name]}, got {idx}")

    num_classes = cfg.get("num_classes")
    if num_classes is not None:
        if not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")
        if num_classes != 11:
            raise ValueError(f"Invalid num_classes: expected 11, got {num_classes}")
        if classes is not None and len(classes) != num_classes:
            raise ValueError(f"num_classes ({num_classes}) does not match classes list length ({len(classes)})")
        if labels is not None and len(labels) != num_classes:
            raise ValueError(f"num_classes ({num_classes}) does not match labels mapping length ({len(labels)})")

    # 5. Model validation
    model_section = cfg.get("model", {})
    model_id = model_section.get("id")
    if model_id != "mossong_plus":
        raise ValueError(f"Invalid model ID: expected 'mossong_plus', got {repr(model_id)}")

    input_shape = model_section.get("input_shape")
    if input_shape is not None:
        if not isinstance(input_shape, list) or not all(isinstance(x, int) and x > 0 for x in input_shape):
            raise ValueError(f"Invalid model.input_shape: must be list of positive integers, got {repr(input_shape)}")
        if segment_length is not None and len(input_shape) > 0:
            if input_shape[0] != segment_length:
                raise ValueError(f"Model input length ({input_shape[0]}) does not match segment length ({segment_length})")

    model_num_classes = model_section.get("num_classes")
    if model_num_classes is not None:
        if num_classes is not None and model_num_classes != num_classes:
            raise ValueError(f"Model output class count ({model_num_classes}) does not match data class count ({num_classes})")

    # 6. Training validation
    epochs = train_section.get("epochs")
    if epochs is not None:
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError(f"Invalid train.epochs: must be positive integer, got {repr(epochs)}")

    batch_size = train_section.get("batch_size")
    if batch_size is not None:
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"Invalid train.batch_size: must be positive integer, got {repr(batch_size)}")

    opt_section = cfg.get("optimizer", {})
    if isinstance(opt_section, dict):
        lr = opt_section.get("learning_rate")
        if lr is not None:
            if not isinstance(lr, (int, float)) or lr <= 0:
                raise ValueError(f"Invalid learning_rate: must be positive float, got {repr(lr)}")

    if mode == "linear_probe":
        checkpoint = (
            cfg.get("checkpoint") or 
            cfg.get("backbone") or 
            cfg.get("pretrained_weights") or 
            cfg.get("model", {}).get("checkpoint") or 
            cfg.get("model", {}).get("backbone") or 
            cfg.get("model", {}).get("pretrained_weights") or 
            cfg.get("train", {}).get("checkpoint") or 
            cfg.get("train", {}).get("backbone") or 
            cfg.get("train", {}).get("pretrained_weights")
        )
        if not checkpoint:
            raise ValueError("Linear probing requires a checkpoint/backbone configuration field")

    # 7. Runtime Profile validation
    dataset_section = cfg.get("dataset", {})
    if isinstance(dataset_section, dict):
        train_dir = dataset_section.get("train_dir")
        if train_dir == "tests/fixtures/audio_11class":
            wandb_section = cfg.get("wandb", {})
            if isinstance(wandb_section, dict) and wandb_section.get("enabled") is not False:
                raise ValueError("W&B tracking must be disabled in CI profile")

    # 8. Secrets validation
    def check_secrets(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if any(x in k.lower() for x in ["api_key", "secret", "password", "token", "private"]):
                    raise ValueError(f"Secrets are not allowed in configuration file (found key {repr(k)})")
                check_secrets(v)
        elif isinstance(d, list):
            for item in d:
                check_secrets(item)

    check_secrets(cfg)
