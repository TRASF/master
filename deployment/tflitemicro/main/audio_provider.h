#pragma once

#include <stddef.h>
#include "esp_err.h"

esp_err_t InitAudio();
esp_err_t GetAudioWindow(float* output, size_t sample_count);
