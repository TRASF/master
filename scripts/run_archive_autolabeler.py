"""Compatibility entrypoint for wingbeat_ml.sed.application.run_archive_autolabeler."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("wingbeat_ml.sed.application.run_archive_autolabeler", run_name="__main__")
else:
    from wingbeat_ml.sed.application.run_archive_autolabeler import *
