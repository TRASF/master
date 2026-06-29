#include "audio_provider.h"
#include "config.h"

#include <math.h>
#include <string.h>

#include "driver/i2s_std.h"
#include "esp_check.h"
#include "esp_log.h"

static const char* TAG = "audio_provider";

static i2s_chan_handle_t rx_chan = nullptr;

// ============================================================
// Statistics helper
// ============================================================

static float ComputeStats(
    const float* x,
    size_t n,
    float* min_v,
    float* max_v,
    float* mean_v,
    float* peak_v
) {
    if (x == nullptr || n == 0) {
        *min_v = 0.0f;
        *max_v = 0.0f;
        *mean_v = 0.0f;
        *peak_v = 0.0f;
        return 0.0f;
    }

    float min_val = x[0];
    float max_val = x[0];
    float peak = fabsf(x[0]);

    double sum = 0.0;
    double sum_sq = 0.0;

    for (size_t i = 0; i < n; ++i) {
        float v = x[i];

        if (v < min_val) {
            min_val = v;
        }

        if (v > max_val) {
            max_val = v;
        }

        float av = fabsf(v);
        if (av > peak) {
            peak = av;
        }

        sum += v;
        sum_sq += (double)v * (double)v;
    }

    float mean = (float)(sum / (double)n);
    float rms = sqrtf((float)(sum_sq / (double)n));

    *min_v = min_val;
    *max_v = max_val;
    *mean_v = mean;
    *peak_v = peak;

    return rms;
}

// ============================================================
// DC removal
// ============================================================

static float RemoveDc(float* x, size_t n) {
    if (x == nullptr || n == 0) {
        return 0.0f;
    }

    double sum = 0.0;

    for (size_t i = 0; i < n; ++i) {
        sum += x[i];
    }

    float mean = (float)(sum / (double)n);

    for (size_t i = 0; i < n; ++i) {
        x[i] -= mean;
    }

    return mean;
}

// ============================================================
// RMS normalization
// ============================================================

static int RmsNormalize(float* x, size_t n, float target_rms) {
    if (x == nullptr || n == 0) {
        return 0;
    }

    const float eps = 1e-8f;

    double sum_sq = 0.0;

    for (size_t i = 0; i < n; ++i) {
        sum_sq += (double)x[i] * (double)x[i];
    }

    float rms = sqrtf((float)(sum_sq / (double)n));
    float gain = target_rms / (rms + eps);

    int clip_count = 0;

    for (size_t i = 0; i < n; ++i) {
        float v = x[i] * gain;

        if (v > 1.0f) {
            v = 1.0f;
            clip_count++;
        } else if (v < -1.0f) {
            v = -1.0f;
            clip_count++;
        }

        x[i] = v;
    }

    return clip_count;
}

// ============================================================
// I2S initialization
// ============================================================

esp_err_t InitAudio() {
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(
        I2S_NUM_AUTO,
        I2S_ROLE_MASTER
    );

    chan_cfg.dma_desc_num = 4;
    chan_cfg.dma_frame_num = 256;

    ESP_RETURN_ON_ERROR(
        i2s_new_channel(&chan_cfg, nullptr, &rx_chan),
        TAG,
        "i2s_new_channel failed"
    );

    i2s_std_config_t std_cfg = {};

    std_cfg.clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE_HZ);

    std_cfg.slot_cfg = I2S_STD_MSB_SLOT_DEFAULT_CONFIG(
        I2S_DATA_BIT_WIDTH_32BIT,
        I2S_SLOT_MODE_MONO
    );

#if USE_LEFT_CHANNEL
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;
#else
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_RIGHT;
#endif

    std_cfg.gpio_cfg.mclk = I2S_GPIO_UNUSED;
    std_cfg.gpio_cfg.bclk = I2S_BCLK_GPIO;
    std_cfg.gpio_cfg.ws = I2S_WS_GPIO;
    std_cfg.gpio_cfg.dout = I2S_GPIO_UNUSED;
    std_cfg.gpio_cfg.din = I2S_DIN_GPIO;

    std_cfg.gpio_cfg.invert_flags.mclk_inv = false;
    std_cfg.gpio_cfg.invert_flags.bclk_inv = false;
    std_cfg.gpio_cfg.invert_flags.ws_inv = false;

    ESP_RETURN_ON_ERROR(
        i2s_channel_init_std_mode(rx_chan, &std_cfg),
        TAG,
        "i2s_channel_init_std_mode failed"
    );

    ESP_RETURN_ON_ERROR(
        i2s_channel_enable(rx_chan),
        TAG,
        "i2s_channel_enable failed"
    );

    ESP_LOGI(
        TAG,
        "I2S initialized: sample_rate=%d Hz, window_ms=%d, samples/window=%d",
        SAMPLE_RATE_HZ,
        WINDOW_MS,
        AUDIO_SAMPLE_COUNT
    );

    return ESP_OK;
}

// ============================================================
// Get one 300 ms audio window
// ============================================================

esp_err_t GetAudioWindow(float* output, size_t sample_count, AudioStats* stats) {
    if (output == nullptr || stats == nullptr) {
        return ESP_ERR_INVALID_ARG;
    }

    if (sample_count != AUDIO_SAMPLE_COUNT) {
        ESP_LOGE(
            TAG,
            "Expected %d samples, got %u",
            AUDIO_SAMPLE_COUNT,
            (unsigned)sample_count
        );
        return ESP_ERR_INVALID_SIZE;
    }

    memset(stats, 0, sizeof(AudioStats));

    static int32_t raw_i2s[AUDIO_SAMPLE_COUNT];

    size_t bytes_read = 0;
    const size_t bytes_to_read = sizeof(raw_i2s);

    esp_err_t err = i2s_channel_read(
        rx_chan,
        raw_i2s,
        bytes_to_read,
        &bytes_read,
        portMAX_DELAY
    );

    if (err != ESP_OK) {
        ESP_LOGE(TAG, "i2s_channel_read failed: %s", esp_err_to_name(err));
        return err;
    }

    size_t samples_read = bytes_read / sizeof(int32_t);

    if (samples_read < AUDIO_SAMPLE_COUNT) {
        ESP_LOGW(
            TAG,
            "Short read: expected=%d samples, got=%u samples",
            AUDIO_SAMPLE_COUNT,
            (unsigned)samples_read
        );
        return ESP_ERR_INVALID_SIZE;
    }

    // ------------------------------------------------------------
    // Convert INMP441 I2S samples to float waveform.
    //
    // Common INMP441 case:
    // 24-bit signed sample is left-aligned inside a 32-bit word.
    // sample24 = raw32 >> 8
    //
    // float scale:
    // signed 24-bit full-scale = 2^23 = 8388608
    // ------------------------------------------------------------

    for (size_t i = 0; i < AUDIO_SAMPLE_COUNT; ++i) {
        int32_t raw32 = raw_i2s[i];
        int32_t sample24 = raw32 >> 8;

        output[i] = (float)sample24 / 8388608.0f;
    }

    // ------------------------------------------------------------
    // 1. Raw stats before DC removal
    // ------------------------------------------------------------

    stats->raw_rms = ComputeStats(
        output,
        AUDIO_SAMPLE_COUNT,
        &stats->raw_min,
        &stats->raw_max,
        &stats->raw_mean,
        &stats->raw_peak
    );

    if (stats->raw_rms > 1e-8f) {
        stats->raw_mean_abs_over_rms = fabsf(stats->raw_mean) / stats->raw_rms;
    } else {
        stats->raw_mean_abs_over_rms = 0.0f;
    }

    // ------------------------------------------------------------
    // 2. Remove DC offset
    // ------------------------------------------------------------

    RemoveDc(output, AUDIO_SAMPLE_COUNT);

    // ------------------------------------------------------------
    // 3. Stats after DC removal
    // ------------------------------------------------------------

    stats->dc_rms = ComputeStats(
        output,
        AUDIO_SAMPLE_COUNT,
        &stats->dc_min,
        &stats->dc_max,
        &stats->dc_mean,
        &stats->dc_peak
    );

    // ------------------------------------------------------------
    // 4. Gate using DC-removed RMS
    // ------------------------------------------------------------

    stats->signal_present = stats->dc_rms >= MIN_RAW_RMS_GATE;
    stats->normalization_applied = false;
    stats->norm_clip_count = 0;

    // ------------------------------------------------------------
    // 5. RMS normalize only if signal is strong enough
    // ------------------------------------------------------------

    if (stats->signal_present) {
        stats->norm_clip_count = RmsNormalize(
            output,
            AUDIO_SAMPLE_COUNT,
            TARGET_RMS
        );

        stats->normalization_applied = true;
    }

    // ------------------------------------------------------------
    // 6. Final stats
    // If signal_present == false, this is the DC-removed unnormalized signal.
    // ------------------------------------------------------------

    stats->norm_rms = ComputeStats(
        output,
        AUDIO_SAMPLE_COUNT,
        &stats->norm_min,
        &stats->norm_max,
        &stats->norm_mean,
        &stats->norm_peak
    );

    return ESP_OK;
}
