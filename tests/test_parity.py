import os
import sys
import json
import unittest
import numpy as np
import tensorflow as tf

# Force CPU execution for tests to avoid GPU dependency
tf.config.set_visible_devices([], 'GPU')

# Disable W&B completely for tests
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_DISABLED"] = "true"

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.mos_config import load_config, normalize_config, apply_reproducibility_environment, resolve_experiment_paths
from src.framework.supervised.dataset import SupervisedDataset
from model.mossongplus import MosSongPlusModel

class TestMosSongPlusParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "parity/baselines"))

        # Load configs
        cls.defaults_path = "configs/defaults.yaml"
        cls.model_cfg_path = "configs/model.yaml"

        cls.defaults_raw = load_config(cls.defaults_path)
        cls.defaults_raw["dataset"]["train_dir"] = "tests/fixtures/audio_11class"
        cls.defaults_raw["dataset"]["val_dir"] = None
        cls.defaults_raw["dataset"]["test_dir"] = None
        cls.defaults_raw["dataset"]["split_list"] = [0.6, 0.2, 0.2]
        cls.defaults_raw["reproducibility"]["seed"] = 45
        cls.defaults_raw["reproducibility"]["deterministic_data"] = True
        cls.defaults_raw["reproducibility"]["deterministic_ops"] = True
        cls.defaults_raw["reproducibility"]["enabled"] = True
        cls.defaults_raw["wandb"] = {"enabled": False}

        # Disable noise overlay
        if "augment" not in cls.defaults_raw:
            cls.defaults_raw["augment"] = {}
        if "noise_overlay" not in cls.defaults_raw["augment"]:
            cls.defaults_raw["augment"]["noise_overlay"] = {}
        cls.defaults_raw["augment"]["noise_overlay"]["p"] = 0.0

        cls.cfg = normalize_config(cls.defaults_raw)
        cls.model_cfg = load_config(cls.model_cfg_path)

        apply_reproducibility_environment(cls.cfg["reproducibility"])
        tf.random.set_seed(45)
        np.random.seed(45)

        cls.ds_builder = SupervisedDataset(
            dataset_dir=cls.cfg["dataset"]["train_dir"],
            val_dir=cls.cfg["dataset"]["val_dir"],
            test_dir=cls.cfg["dataset"]["test_dir"],
            sample_rate=cls.cfg["audio"]["sample_rate"],
            segment_length=cls.cfg["audio"]["segment_length"],
            classes=cls.cfg["classes"],
            noise_dirs=None,
            augment_cfg=cls.cfg["augment"],
            seed=cls.cfg["reproducibility"]["seed"],
            deterministic=cls.cfg["reproducibility"]["deterministic_data"],
            nomos_index=cls.cfg["nomos_index"],
            labels_dict=cls.cfg["labels"]
        )

        # Build the dataset
        cls.train_ds, cls.val_ds, cls.test_ds = cls.ds_builder.build(
            split=cls.cfg["dataset"]["split_list"],
            batch_size=2,
            shuffle=False
        )

    def test_file_discovery(self):
        with open(os.path.join(self.baseline_dir, "file_discovery.json"), "r") as f:
            baseline = json.load(f)

        train_paths, train_labels = self.ds_builder.data_loader.gather_files()

        discovered_files = [os.path.basename(p) for p in sorted(train_paths)]
        discovered_labels = [int(l) for _, l in sorted(zip(train_paths, train_labels))]

        self.assertEqual(discovered_files, baseline["files"])
        self.assertEqual(discovered_labels, baseline["labels"])

    def test_label_map(self):
        with open(os.path.join(self.baseline_dir, "label_map.json"), "r") as f:
            baseline = json.load(f)

        self.assertEqual(self.cfg["classes"], baseline["classes"])
        self.assertEqual(self.cfg["labels"], baseline["class_to_idx"])
        self.assertEqual(self.cfg["num_classes"], baseline["num_classes"])

    def test_splits(self):
        with open(os.path.join(self.baseline_dir, "splits.json"), "r") as f:
            baseline = json.load(f)

        train_files = [os.path.basename(p) for p in self.ds_builder.train_paths]
        train_labels = [int(l) for l in self.ds_builder.train_labels]
        val_files = [os.path.basename(p) for p in self.ds_builder.val_paths]
        val_labels = [int(l) for l in self.ds_builder.val_labels]
        test_files = [os.path.basename(p) for p in self.ds_builder.test_paths]
        test_labels = [int(l) for l in self.ds_builder.test_labels]

        self.assertEqual(train_files, baseline["train_paths"])
        self.assertEqual(train_labels, baseline["train_labels"])
        self.assertEqual(val_files, baseline["val_paths"])
        self.assertEqual(val_labels, baseline["val_labels"])
        self.assertEqual(test_files, baseline["test_paths"])
        self.assertEqual(test_labels, baseline["test_labels"])

    def test_preprocessing(self):
        baseline = np.load(os.path.join(self.baseline_dir, "preprocessing_outputs.npz"))
        train_paths, train_labels = self.ds_builder.data_loader.gather_files()

        for path, label in zip(train_paths, train_labels):
            name = os.path.basename(path)
            raw_audio = self.ds_builder.data_loader.load_file(path)
            sliced_audio = raw_audio[:self.ds_builder.segment_length]
            if len(sliced_audio) < self.ds_builder.segment_length:
                sliced_audio = np.pad(sliced_audio, (0, self.ds_builder.segment_length - len(sliced_audio)))

            audio_tensor = tf.convert_to_tensor(sliced_audio, dtype=tf.float32)
            preprocessed_audio, _ = self.ds_builder.augmentor.apply_post_processing(
                audio_tensor, tf.constant(label, dtype=tf.int32), augment=False
            )

            # Tolerances for float audio preprocessing
            rtol = 1e-6
            atol = 1e-7

            np.testing.assert_allclose(sliced_audio, baseline[f"{name}_raw"], rtol=rtol, atol=atol)
            np.testing.assert_allclose(preprocessed_audio.numpy(), baseline[f"{name}_preprocessed"], rtol=rtol, atol=atol)

    def test_augmentation(self):
        baseline = np.load(os.path.join(self.baseline_dir, "augmentation_outputs.npz"))
        train_paths, train_labels = self.ds_builder.data_loader.gather_files()

        for i, (path, label) in enumerate(zip(train_paths[:3], train_labels[:3])):
            name = os.path.basename(path)
            raw_audio = self.ds_builder.data_loader.load_file(path)
            sliced_audio = raw_audio[:self.ds_builder.segment_length]
            if len(sliced_audio) < self.ds_builder.segment_length:
                sliced_audio = np.pad(sliced_audio, (0, self.ds_builder.segment_length - len(sliced_audio)))

            audio_tensor = tf.convert_to_tensor(sliced_audio, dtype=tf.float32)

            test_augment_cfg = {
                "preprocess": {"dc_removal": True},
                "rms_norm": {"target_rms": 0.05, "min_gain": 0.05, "max_gain": 15.0},
                "high_pass": {"p": 1.0, "fc": 150},
                "time_shift": {"p": 1.0, "rate": [-0.05, 0.05]},
                "random_gain": {"p": 1.0, "gain_db": [-3.0, 3.0]},
                "pitch_shift": {"p": 1.0, "semitones": [-0.2, 0.2]}
            }
            test_augmentor = self.ds_builder.augmentor.__class__(
                segment_length=self.cfg["audio"]["segment_length"],
                config=test_augment_cfg,
                seed=45,
                deterministic=True,
                nomos_index=self.ds_builder.nomos_index
            )

            hpf_audio = test_augmentor.apply_hpf(audio_tensor)
            time_shifted = test_augmentor.time_shift(audio_tensor, [-0.05, 0.05], tf.constant([45, 1], dtype=tf.int64))
            gained = test_augmentor.random_gain(audio_tensor, [-3.0, 3.0], tf.constant([45, 2], dtype=tf.int64))
            pitch_shifted = test_augmentor.pitch_shift(audio_tensor, [-0.2, 0.2], tf.constant([45, 3], dtype=tf.int64))

            rtol = 1e-5
            atol = 1e-6

            np.testing.assert_allclose(hpf_audio.numpy(), baseline[f"{name}_hpf"], rtol=rtol, atol=atol)
            np.testing.assert_allclose(time_shifted.numpy(), baseline[f"{name}_timeshift"], rtol=rtol, atol=atol)
            np.testing.assert_allclose(gained.numpy(), baseline[f"{name}_gain"], rtol=rtol, atol=atol)
            np.testing.assert_allclose(pitch_shifted.numpy(), baseline[f"{name}_pitchshift"], rtol=rtol, atol=atol)

    def test_model_structure(self):
        with open(os.path.join(self.baseline_dir, "model_structure.json"), "r") as f:
            baseline = json.load(f)

        import tensorflow.keras as keras
        keras.backend.clear_session()
        keras.utils.set_random_seed(45)
        model_builder = MosSongPlusModel(self.model_cfg, model_overrides=self.cfg.get("model"))
        model = model_builder.build(
            input_shape=(self.cfg["audio"]["segment_length"], 1),
            output_units=self.cfg["num_classes"],
            output_activation=self.cfg["model"]["output_activation"]
        )

        self.assertEqual(model.name, baseline["name"])
        self.assertEqual(len(model.layers), len(baseline["layers"]))

        for layer, base_layer in zip(model.layers, baseline["layers"]):
            self.assertEqual(layer.name, base_layer["name"])
            self.assertEqual(layer.__class__.__name__, base_layer["class"])
            self.assertEqual(layer.trainable, base_layer["trainable"])
            self.assertEqual([list(w.shape) for w in layer.weights], base_layer["weight_shapes"])

        total_params = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))
        self.assertEqual(total_params, baseline["total_params"])

    def test_predictions(self):
        baseline = np.load(os.path.join(self.baseline_dir, "initial_predictions.npz"))

        import tensorflow.keras as keras
        keras.backend.clear_session()
        keras.utils.set_random_seed(45)
        model_builder = MosSongPlusModel(self.model_cfg, model_overrides=self.cfg.get("model"))
        model = model_builder.build(
            input_shape=(self.cfg["audio"]["segment_length"], 1),
            output_units=self.cfg["num_classes"],
            output_activation=self.cfg["model"]["output_activation"]
        )

        dummy_input = np.ones((5, self.cfg["audio"]["segment_length"], 1), dtype=np.float32)
        initial_preds = model.predict(dummy_input)

        np.testing.assert_allclose(initial_preds, baseline["preds"], rtol=1e-5, atol=1e-6)

    def test_output_paths(self):
        with open(os.path.join(self.baseline_dir, "output_paths.json"), "r") as f:
            baseline = json.load(f)

        pretrain_paths = resolve_experiment_paths(self.cfg, "Pretrain_test_exp")
        self.assertEqual(pretrain_paths, baseline["pretrain"])

    def test_wandb_keys(self):
        with open(os.path.join(self.baseline_dir, "wandb_keys.json"), "r") as f:
            baseline = json.load(f)

        self.assertEqual(baseline["init_keys"], ["project", "config", "group", "tags", "job_type"])

if __name__ == "__main__":
    unittest.main()
