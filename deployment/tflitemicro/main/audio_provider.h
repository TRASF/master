#pragma once

#include <stddef.h>
#include "esp_err.h"

struct AudioStats {
    // Raw signal immediately after I2S int24 -> float conversion
    float raw_min;
    float raw_max;
    float raw_mean;
    float raw_rms;
    float raw_peak;
    float raw_mean_abs_over_rms;

    // Signal after DC removal, before RMS normalization
    float dc_min;
    float dc_max;
    float dc_mean;
    float dc_rms;
    float dc_peak;

    // Final signal after optional RMS normalization
    float norm_min;
    float norm_max;
    float norm_mean;
    float norm_rms;
    float norm_peak;

    int norm_clip_count;

    bool signal_present;
    bool normalization_applied;
};

esp_err_t InitAudio();
esp_err_t GetAudioWindow(float* output, size_t sample_count, AudioStats* stats);
