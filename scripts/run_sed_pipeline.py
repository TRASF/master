"""Compatibility entrypoint for wingbeat_ml.sed.application.run_sed_pipeline."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("wingbeat_ml.sed.application.run_sed_pipeline", run_name="__main__")
else:
    from wingbeat_ml.sed.application.run_sed_pipeline import *
