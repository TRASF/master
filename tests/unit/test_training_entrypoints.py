"""Contract tests for canonical and legacy training entrypoints."""

import importlib
import importlib.util
import inspect
import unittest


class TestTrainingEntrypoints(unittest.TestCase):
    def test_canonical_modules_exist(self):
        for name in (
            "wingbeat_ml.pipelines.pretrain",
            "wingbeat_ml.pipelines.linear_probe",
            "wingbeat_ml.pipelines.fine_tune",
        ):
            self.assertIsNotNone(
                importlib.util.find_spec(name),
                f"canonical {name.rsplit('.', 1)[-1]} entrypoint is missing",
            )

    def test_central_mode_selector(self):
        from wingbeat_ml.pipelines import get_training_entrypoint
        from wingbeat_ml.pipelines.fine_tune import train_finetune
        from wingbeat_ml.pipelines.linear_probe import train_linear_probe
        from wingbeat_ml.pipelines.pretrain import train_supervised
        from wingbeat_ml.pipelines.ssl_tf import run_tf_ssl_pipeline

        self.assertIs(
            get_training_entrypoint("pretrain"),
            train_supervised,
        )
        self.assertIs(
            get_training_entrypoint("linear-probe"),
            train_linear_probe,
        )
        self.assertIs(
            get_training_entrypoint("finetune"),
            train_finetune,
        )
        self.assertIs(
            get_training_entrypoint("ssl_tf"),
            run_tf_ssl_pipeline,
        )

        with self.assertRaises(ValueError):
            get_training_entrypoint("unknown")



    def test_canonical_modules_do_not_import_legacy_framework(self):
        modules = (
            importlib.import_module("wingbeat_ml.pipelines.pretrain"),
            importlib.import_module("wingbeat_ml.pipelines.linear_probe"),
            importlib.import_module("wingbeat_ml.pipelines.fine_tune"),
        )

        for module in modules:
            source = inspect.getsource(module)
            self.assertNotIn("configs.mos_config", source)
            self.assertNotIn("src.framework", source)

    def test_canonical_entrypoints_share_the_epoch_loop(self):
        for name in (
            "wingbeat_ml.pipelines.pretrain",
            "wingbeat_ml.pipelines.linear_probe",
            "wingbeat_ml.pipelines.fine_tune",
        ):
            module = importlib.import_module(name)
            source = inspect.getsource(module)
            self.assertIn("run_training(", source)
            self.assertNotIn("for epoch in range(epochs)", source)

    def test_canonical_entrypoints_use_pipeline_helpers(self):
        for name in (
            "wingbeat_ml.pipelines.pretrain",
            "wingbeat_ml.pipelines.linear_probe",
            "wingbeat_ml.pipelines.fine_tune",
        ):
            source = inspect.getsource(importlib.import_module(name))
            self.assertIn("prepare_training_run(", source)
            self.assertIn("build_supervised_components(", source)
            self.assertIn("wingbeat_ml.pipelines.helpers", source)
            self.assertNotIn("configure_training_runtime(", source)
            self.assertNotIn("initialize_training_run(", source)
            self.assertNotIn("set_memory_growth(", source)
            self.assertNotIn("tf.random.set_seed(", source)
            self.assertNotIn("wandb.init(", source)

    def test_component_assembly_is_not_repeated_in_entrypoints(self):
        for name in (
            "wingbeat_ml.pipelines.pretrain",
            "wingbeat_ml.pipelines.linear_probe",
            "wingbeat_ml.pipelines.fine_tune",
        ):
            source = inspect.getsource(importlib.import_module(name))
            self.assertNotIn("build_datasets(", source)
            self.assertNotIn("build_model(", source)
            self.assertNotIn("resolve_training_class_weights(", source)
            self.assertNotIn("SupervisedDataset(", source)
            self.assertNotIn("MosSongPlusModel(", source)




if __name__ == "__main__":
    unittest.main()
