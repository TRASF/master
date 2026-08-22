"""Compatibility wrapper for classification pipeline linear_probe."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("wingbeat_ml.classification.pipelines.linear_probe", run_name="__main__")
else:
    from wingbeat_ml.classification.pipelines.linear_probe import *
