"""Compatibility wrapper for classification pipeline evaluate."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("wingbeat_ml.classification.pipelines.evaluate", run_name="__main__")
else:
    from wingbeat_ml.classification.pipelines.evaluate import *
