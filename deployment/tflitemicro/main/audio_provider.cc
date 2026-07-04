#include "audio_provider.h"
#include "config.h"

#include <math.h>
#include <string.h>

#include "driver/i2s_std.h"
#include "esp_check.h"
#include "esp_log.h"

static const char* TAG = "audio_provider";

static i2s_chan_handle_t rx_chan = NULL;

// ============================================================
// Bit Shifting Helper
// ============================================================
static int32_t sign_extend_24(uint32_t value) {
    value &= 0x00FFFFFFu;
    if ((value & 0x00800000u) != 0) {
        value |= 0xFF000000u;
    }
    return (int32_t)value;
}

// ============================================================
// DC removal
// ============================================================
static void RemoveDc(float* x, size_t n) {
    if (x == NULL || n == 0) return;

    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += x[i];
    }
    float mean = (float)(sum / (double)n);
    for (size_t i = 0; i < n; ++i) {
        x[i] -= mean;
    }
}

// ============================================================
// RMS normalization (Matched to TF Pipeline Config)
// ============================================================
static void RmsNormalize(float* x, size_t n, float current_rms, float target_rms) {
    if (x == NULL || n == 0) return;

    const float eps = 1e-8f;
    float gain = target_rms / (current_rms + eps);

    // Apply the min/max gain limits from your Python config.yaml
    if (gain < 0.05f) gain = 0.05f;
    if (gain > 15.0f) gain = 15.0f;

    for (size_t i = 0; i < n; ++i) {
        float v = x[i] * gain;

        // Hard clipping constraint to [-1.0, 1.0]
        if (v > 1.0f) v = 1.0f;
        else if (v < -1.0f) v = -1.0f;

        x[i] = v;
    }
}

// ============================================================
// I2S initialization
// ============================================================
esp_err_t InitAudio() {
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
    chan_cfg.dma_desc_num = 4;
    chan_cfg.dma_frame_num = 256;

    ESP_RETURN_ON_ERROR(i2s_new_channel(&chan_cfg, NULL, &rx_chan), TAG, "i2s_new_channel failed");

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE_HZ),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = I2S_BCLK_GPIO,
            .ws = I2S_WS_GPIO,
            .dout = I2S_GPIO_UNUSED,
            .din = I2S_DIN_GPIO,
            .invert_flags = {.mclk_inv = false, .bclk_inv = false, .ws_inv = false},
        },
    };
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;

    ESP_RETURN_ON_ERROR(i2s_channel_init_std_mode(rx_chan, &std_cfg), TAG, "i2s_channel_init_std_mode failed");
    ESP_RETURN_ON_ERROR(i2s_channel_enable(rx_chan), TAG, "i2s_channel_enable failed");

    ESP_LOGI(TAG, "I2S initialized (MONO): sample_rate=%d Hz, samples=%d", SAMPLE_RATE_HZ, AUDIO_SAMPLE_COUNT);
    return ESP_OK;
}

// ============================================================
// Get one audio window
// ============================================================
esp_err_t GetAudioWindow(float* output, size_t sample_count) {
    if (output == NULL) return ESP_ERR_INVALID_ARG;
    if (sample_count != AUDIO_SAMPLE_COUNT) return ESP_ERR_INVALID_SIZE;

    static int32_t raw_i2s[AUDIO_SAMPLE_COUNT];
    size_t bytes_read = 0;

    esp_err_t err = i2s_channel_read(rx_chan, raw_i2s, sizeof(raw_i2s), &bytes_read, portMAX_DELAY);
    if (err != ESP_OK) return err;

    // 1. Convert I2S to float waveform
    for (size_t i = 0; i < AUDIO_SAMPLE_COUNT; i++) {
        int32_t sample = sign_extend_24(raw_i2s[i] >> 8);
        output[i] = (float)sample / 8388608.0f;
    }

    // 2. Remove DC offset
    RemoveDc(output, AUDIO_SAMPLE_COUNT);

    // 3. Calculate current RMS for the noise gate
    double sum_sq = 0.0;
    for (size_t i = 0; i < AUDIO_SAMPLE_COUNT; i++) {
        sum_sq += (double)output[i] * (double)output[i];
    }
    float current_rms = sqrtf((float)(sum_sq / (double)AUDIO_SAMPLE_COUNT));

    // 4. Gate and Normalize (Now active and matching TF)
    if (current_rms >= MIN_RAW_RMS_GATE) {
        RmsNormalize(output, AUDIO_SAMPLE_COUNT, current_rms, TARGET_RMS);
    }

    return ESP_OK;
}
