#include <stdio.h>

#include "audio_provider.h"
#include "classifier.h"
#include "config.h"

#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char* TAG = "main";

extern "C" void app_main(void) {
    ESP_LOGI(TAG, "Mosquito raw-audio edge classifier booting");

    ESP_LOGI(TAG, "Sample rate: %d Hz", SAMPLE_RATE_HZ);
    ESP_LOGI(TAG, "Window: %d ms", WINDOW_MS);
    ESP_LOGI(TAG, "Samples/window: %d", AUDIO_SAMPLE_COUNT);
    ESP_LOGI(TAG, "Target RMS: %.6f", TARGET_RMS);
    ESP_LOGI(TAG, "Min RMS gate after DC removal: %.6f", MIN_RAW_RMS_GATE);
    ESP_LOGI(TAG, "Dummy classifier: %d", USE_DUMMY_CLASSIFIER);

    ESP_ERROR_CHECK(InitAudio());
    ESP_ERROR_CHECK(InitClassifier());

    static float audio_window[AUDIO_SAMPLE_COUNT];

    while (true) {
        AudioStats stats = {};
        ClassifierResult result = {};

        int64_t t0 = esp_timer_get_time();

        esp_err_t err = GetAudioWindow(
            audio_window,
            AUDIO_SAMPLE_COUNT,
            &stats
        );

        int64_t t1 = esp_timer_get_time();

        if (err != ESP_OK) {
            ESP_LOGE(TAG, "GetAudioWindow failed: %s", esp_err_to_name(err));
            vTaskDelay(pdMS_TO_TICKS(500));
            continue;
        }

        err = RunClassifier(
            audio_window,
            AUDIO_SAMPLE_COUNT,
            &stats,
            &result
        );

        int64_t t2 = esp_timer_get_time();

        if (err != ESP_OK) {
            ESP_LOGE(TAG, "RunClassifier failed: %s", esp_err_to_name(err));
            vTaskDelay(pdMS_TO_TICKS(500));
            continue;
        }

        ESP_LOGI(
            TAG,
            "raw: min=%+.6f max=%+.6f mean=%+.6f rms=%.6f peak=%.6f mean/rms=%.3f | "
            "dc: mean=%+.6f rms=%.6f peak=%.6f | "
            "norm: mean=%+.6f rms=%.6f peak=%.6f clips=%d present=%d normalized=%d | "
            "qinput: scale=%.9f zp=%d qclips=%d | "
            "pred=%d:%s score=%.3f top3=[%d:%s %.3f, %d:%s %.3f, %d:%s %.3f] | "
            "audio=%lld us infer=%lld us",

            stats.raw_min,
            stats.raw_max,
            stats.raw_mean,
            stats.raw_rms,
            stats.raw_peak,
            stats.raw_mean_abs_over_rms,

            stats.dc_mean,
            stats.dc_rms,
            stats.dc_peak,

            stats.norm_mean,
            stats.norm_rms,
            stats.norm_peak,
            stats.norm_clip_count,
            stats.signal_present ? 1 : 0,
            stats.normalization_applied ? 1 : 0,

            result.input_scale,
            result.input_zero_point,
            result.input_clip_count,

            result.predicted_class,
            ClassName(result.predicted_class),
            result.confidence,

            result.top_class[0],
            ClassName(result.top_class[0]),
            result.top_score[0],

            result.top_class[1],
            ClassName(result.top_class[1]),
            result.top_score[1],

            result.top_class[2],
            ClassName(result.top_class[2]),
            result.top_score[2],

            (long long)(t1 - t0),
            (long long)(t2 - t1)
        );

        vTaskDelay(pdMS_TO_TICKS(50));
    }
}
