"""Compatibility entrypoint for wingbeat_ml.sed.application.predict_gold."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("wingbeat_ml.sed.application.predict_gold", run_name="__main__")
else:
    from wingbeat_ml.sed.application.predict_gold import *
