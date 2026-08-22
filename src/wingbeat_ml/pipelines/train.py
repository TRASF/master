"""Compatibility wrapper for classification pipeline train."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("wingbeat_ml.classification.pipelines.train", run_name="__main__")
else:
    from wingbeat_ml.classification.pipelines.train import *
