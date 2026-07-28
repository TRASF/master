"""Unit tests for SSL pipeline (FixMatch & FlexMatch)."""

import pytest
import torch
from wingbeat_ml.pipelines.ssl import run_ssl_pipeline, train_fixmatch, train_flexmatch


def test_fixmatch_pipeline_run():
    res = train_fixmatch(epochs=1, verbose=False)
    assert res["status"] == "success"
    assert res["method"] == "fixmatch"
    assert len(res["history"]) == 1
    assert "source_accuracy" in res["final_evaluation"]
    assert "target_accuracy" in res["final_evaluation"]


def test_flexmatch_pipeline_run():
    res = train_flexmatch(epochs=1, verbose=False)
    assert res["status"] == "success"
    assert res["method"] == "flexmatch"
    assert len(res["history"]) == 1
    assert "source_accuracy" in res["final_evaluation"]
    assert "target_accuracy" in res["final_evaluation"]
