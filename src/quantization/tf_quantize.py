"""Compatibility wrapper for canonical TFLite export modules."""

from wingbeat_ml.export.bundle import (
    export_input_quantization_header,
    export_tflite_to_c_header,
    write_esp32_readme,
)
from wingbeat_ml.export.tflite import (
    convert_dynamic_range_tflite,
    convert_float_tflite,
    convert_full_int8_tflite,
    convert_int16x8_tflite_experiment,
    convert_selective_quantized_tflite_for_analysis,
    dump_tflite_analyzer,
    ensure_dir,
    inspect_tflite_io,
    make_representative_dataset,
    run_quantization_debugger,
    save_tflite_model,
)
from wingbeat_ml.export.verify import (
    compare_model_pair_agreement,
    dequantize_input,
    evaluate_keras_input_qdq_model,
    evaluate_keras_model,
    evaluate_predictions,
    evaluate_tflite_model,
    predict_keras_dataset,
    predict_keras_with_input_qdq,
    predict_tflite_dataset,
)
from wingbeat_ml.pipelines.export import (
    main,
    run_basic_quantization_suite,
)

__all__ = [
    "compare_model_pair_agreement",
    "convert_dynamic_range_tflite",
    "convert_float_tflite",
    "convert_full_int8_tflite",
    "convert_int16x8_tflite_experiment",
    "convert_selective_quantized_tflite_for_analysis",
    "dequantize_input",
    "dump_tflite_analyzer",
    "ensure_dir",
    "evaluate_keras_input_qdq_model",
    "evaluate_keras_model",
    "evaluate_predictions",
    "evaluate_tflite_model",
    "export_input_quantization_header",
    "export_tflite_to_c_header",
    "inspect_tflite_io",
    "main",
    "make_representative_dataset",
    "predict_keras_dataset",
    "predict_keras_with_input_qdq",
    "predict_tflite_dataset",
    "run_basic_quantization_suite",
    "run_quantization_debugger",
    "save_tflite_model",
    "write_esp32_readme",
]


if __name__ == "__main__":
    main()
