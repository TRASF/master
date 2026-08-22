"""Compatibility wrapper for classification pipeline ssl_tf."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("wingbeat_ml.classification.pipelines.ssl_tf", run_name="__main__")
else:
    from wingbeat_ml.classification.pipelines.ssl_tf import *
