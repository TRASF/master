#include "audio_provider.h"
#include "config.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "driver/i2s_std.h"
#include "esp_check.h"
#include "esp_log.h"

#if BIT_DEPTH != 16 && BIT_DEPTH != 24
#error "BIT_DEPTH must be either 16 or 24"
#endif

// INMP441 is a native 24-bit microphone transported in a 32-bit I2S slot.
// Keep the hardware/I2S read width at 32 bits. BIT_DEPTH selects how we
// convert that native 24-bit payload for the model/debug stream:
//   24: use the full signed 24-bit sample
//   16: down-convert the signed 24-bit sample to signed 16-bit precision
using I2sRawSample = int32_t;
#define I2S_CAPTURE_DATA_BIT_WIDTH I2S_DATA_BIT_WIDTH_32BIT

static const char* TAG = "audio_provider";
static i2s_chan_handle_t rx_chan = NULL;

// ============================================================
// Bit-depth conversion helpers
// ============================================================
static int32_t sign_extend_24(uint32_t value) {
    value &= 0x00FFFFFFu;
    if ((value & 0x00800000u) != 0) {
        value |= 0xFF000000u;
    }
    return (int32_t)value;
}

static int32_t I2sRawSampleToSigned24(I2sRawSample raw_sample) {
    // ESP-IDF reads the INMP441 24-bit payload from a 32-bit slot. The common
    // layout is signed 24-bit data in bits [31:8], with the low byte unused.
    return sign_extend_24(((uint32_t)raw_sample) >> 8);
}

static float I2sSampleToFloat(I2sRawSample raw_sample) {
    int32_t sample24 = I2sRawSampleToSigned24(raw_sample);
#if BIT_DEPTH == 16
    // Down-convert the native signed 24-bit sample to signed 16-bit precision,
    // then normalize. This keeps I2S capture correct for INMP441 while allowing
    // a selectable 16-bit processing/streaming mode.
    int16_t sample16 = (int16_t)(sample24 >> 8);
    float normalized = (float)sample16 / 32768.0f;
#else
    float normalized = (float)sample24 / 8388608.0f;
#endif
    return normalized / 0.03;
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

static float ComputeRms(const float* x, size_t n) {
    if (x == NULL || n == 0) return 0.0f;

    double sum_sq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum_sq += (double)x[i] * (double)x[i];
    }
    return sqrtf((float)(sum_sq / (double)n));
}

// ============================================================
// RMS normalization (matched to Python AudioAugmentor.rms_normalize)
// ============================================================
static void RmsNormalize(float* x, size_t n, float current_rms) {
    if (x == NULL || n == 0) return;

    const float eps = 1e-8f;
    float gain = TARGET_RMS / (current_rms + eps);

    if (gain < RMS_MIN_GAIN) gain = RMS_MIN_GAIN;
    if (gain > RMS_MAX_GAIN) gain = RMS_MAX_GAIN;

    for (size_t i = 0; i < n; ++i) {
        float v = x[i] * gain;
        if (v > 1.0f) v = 1.0f;
        else if (v < -1.0f) v = -1.0f;
        x[i] = v;
    }
}

static esp_err_t ReadRawSamples(float* output, size_t sample_count) {
    if (output == NULL) return ESP_ERR_INVALID_ARG;

    static I2sRawSample raw_i2s[AUDIO_SAMPLE_COUNT];
    if (sample_count > AUDIO_SAMPLE_COUNT) return ESP_ERR_INVALID_SIZE;

    size_t samples_read = 0;
    while (samples_read < sample_count) {
        const size_t remaining = sample_count - samples_read;
        size_t bytes_read = 0;

        esp_err_t err = i2s_channel_read(
            rx_chan,
            raw_i2s,
            remaining * sizeof(raw_i2s[0]),
            &bytes_read,
            portMAX_DELAY
        );
        if (err != ESP_OK) return err;

        const size_t got = bytes_read / sizeof(raw_i2s[0]);
        if (got == 0) return ESP_FAIL;

        for (size_t i = 0; i < got; ++i) {
            output[samples_read + i] = I2sSampleToFloat(raw_i2s[i]);
        }
        samples_read += got;
    }

    return ESP_OK;
}

static void ApplyTrainingPreprocess(float* output, size_t sample_count) {
    // Training/validation pipeline order:
    //   1. DC removal
    //   2. RMS normalization with configured gain clamp
    //   3. Clip to [-1, 1] inside RmsNormalize
    RemoveDc(output, sample_count);

    float current_rms = ComputeRms(output, sample_count);

#if ENABLE_RAW_RMS_GATE
    // Optional deployment safety gate. Disabled by default because training does
    // not gate quiet samples before RMS normalization.
    if (current_rms < MIN_RAW_RMS_GATE) {
        return;
    }
#endif

    RmsNormalize(output, sample_count, current_rms);
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
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_CAPTURE_DATA_BIT_WIDTH, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = I2S_BCLK_GPIO,
            .ws = I2S_WS_GPIO,
            .dout = I2S_GPIO_UNUSED,
            .din = I2S_DIN_GPIO,
            .invert_flags = {.mclk_inv = false, .bclk_inv = false, .ws_inv = false},
        },
    };

#if USE_LEFT_CHANNEL
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;
#else
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_RIGHT;
#endif

    ESP_RETURN_ON_ERROR(i2s_channel_init_std_mode(rx_chan, &std_cfg), TAG, "i2s_channel_init_std_mode failed");
    ESP_RETURN_ON_ERROR(i2s_channel_enable(rx_chan), TAG, "i2s_channel_enable failed");

    ESP_LOGI(
        TAG,
        "I2S initialized: sample_rate=%d Hz, bit_depth=%d, window=%d samples, hop=%d samples",
        SAMPLE_RATE_HZ,
        BIT_DEPTH,
        AUDIO_SAMPLE_COUNT,
        AUDIO_HOP_SAMPLE_COUNT
    );
    return ESP_OK;
}

// ============================================================
// Get one audio window, using validation-style overlapping frames.
// The rolling buffer stores raw audio; preprocessing is applied only to the
// returned copy so prior overlap samples are not normalized repeatedly.
// ============================================================
esp_err_t GetAudioWindow(float* output, size_t sample_count) {
    if (output == NULL) return ESP_ERR_INVALID_ARG;
    if (sample_count != AUDIO_SAMPLE_COUNT) return ESP_ERR_INVALID_SIZE;

    static float raw_window[AUDIO_SAMPLE_COUNT];
    static bool window_initialized = false;

    if (!window_initialized) {
        ESP_RETURN_ON_ERROR(ReadRawSamples(raw_window, AUDIO_SAMPLE_COUNT), TAG, "initial I2S read failed");
        window_initialized = true;
    } else {
        memmove(
            raw_window,
            raw_window + AUDIO_HOP_SAMPLE_COUNT,
            (AUDIO_SAMPLE_COUNT - AUDIO_HOP_SAMPLE_COUNT) * sizeof(raw_window[0])
        );
        ESP_RETURN_ON_ERROR(
            ReadRawSamples(raw_window + (AUDIO_SAMPLE_COUNT - AUDIO_HOP_SAMPLE_COUNT), AUDIO_HOP_SAMPLE_COUNT),
            TAG,
            "hop I2S read failed"
        );
    }

    memcpy(output, raw_window, AUDIO_SAMPLE_COUNT * sizeof(output[0]));
    ApplyTrainingPreprocess(output, AUDIO_SAMPLE_COUNT);

    return ESP_OK;
}
