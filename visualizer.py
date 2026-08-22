"""Compatibility launcher for the host visualizer."""
import runpy

if __name__ == "__main__":
    runpy.run_path("apps/visualize.py", run_name="__main__")
