#pragma once

#include <stddef.h>

#include "audio_provider.h"
#include "esp_err.h"

struct ClassifierResult {
    int predicted_class;
    float confidence;

    int top_class[3];
    float top_score[3];

    bool used_dummy_classifier;

    int input_clip_count;
    float input_scale;
    int input_zero_point;

    float output_scale;
    int output_zero_point;
};

esp_err_t InitClassifier();

esp_err_t RunClassifier(
    const float* audio_window,
    size_t sample_count,
    const AudioStats* audio_stats,
    ClassifierResult* result
);

const char* ClassName(int class_index);
