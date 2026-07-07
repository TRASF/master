#pragma once

#include <stddef.h>
#include <stdint.h>
#include "esp_err.h"

struct ClassifierResult {
    int predicted_class;
    float confidence;

    int top_class[3];
    float top_score[3];

    float input_scale;
    int input_zero_point;
    float output_scale;
    int output_zero_point;

    int input_clip_count;
};

const char* ClassName(int class_index);

esp_err_t InitClassifier();

// Bounded deterministic self-test for OTA first-boot validation. It does not
// read live audio; it validates tensor dimensions, invokes the interpreter with
// a fixed zero input, and checks that all output values are finite.
esp_err_t RunClassifierBootSelfTest();

esp_err_t RunClassifier(
    const float* audio_window,
    size_t sample_count,
    ClassifierResult* result
);
