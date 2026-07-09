#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "driver/uart.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_timer.h"

#include "audio_provider.h"
#include "classification.h"
#include "config.h"
#include "ota_update.h"
#include "wifi_helper.h"
#include "nvs_flash.h"

static const char *TAG = "MAIN_APP";

#pragma pack(push, 1)
struct TelemetryPayload {
    uint32_t seq;
    uint32_t audio_timestamp_us;
    uint8_t predicted_class;
    float confidence;
    uint32_t inference_time_us;
    uint32_t class_age_ms;
    uint32_t classifier_seq;
    int16_t audio[AUDIO_SAMPLE_COUNT];
};
#pragma pack(pop)

struct ClassifierJob {
    uint32_t seq;
    float audio[AUDIO_SAMPLE_COUNT];
};

struct LatestClassifierState {
    uint8_t predicted_class;
    float confidence;
    uint32_t inference_time_us;
    uint32_t result_timestamp_us;
    uint32_t classifier_seq;
};

static SemaphoreHandle_t g_result_mutex;
static QueueHandle_t g_classifier_queue;
static LatestClassifierState g_latest_result = {
    .predicted_class = 10,
    .confidence = 0.0f,
    .inference_time_us = 0,
    .result_timestamp_us = 0,
    .classifier_seq = 0,
};

// COBS encode a payload; the caller supplies enough output space.
static size_t cobs_encode(const uint8_t *input, size_t length, uint8_t *output) {
    size_t read_index = 0, write_index = 1, code_index = 0;
    uint8_t code = 1;

    while (read_index < length) {
        if (input[read_index] == 0) {
            output[code_index] = code;
            code = 1;
            code_index = write_index++;
            read_index++;
        } else {
            output[write_index++] = input[read_index++];
            code++;
            if (code == 0xFF) {
                output[code_index] = code;
                code = 1;
                code_index = write_index++;
            }
        }
    }
    output[code_index] = code;
    return write_index;
}

static void audio_telemetry_task(void *arg) {
    ESP_LOGI(TAG, "Starting audio telemetry task");

    float *audio_window = (float *)malloc(AUDIO_SAMPLE_COUNT * sizeof(float));
    ClassifierJob *job = (ClassifierJob *)malloc(sizeof(ClassifierJob));
#if STREAM_TO_PYTHON
    TelemetryPayload *payload = (TelemetryPayload *)malloc(sizeof(TelemetryPayload));
    const size_t max_cobs_len = sizeof(TelemetryPayload) + sizeof(TelemetryPayload) / 254 + 2;
    uint8_t *cobs_buffer = (uint8_t *)malloc(max_cobs_len);
#else
    TelemetryPayload *payload = NULL;
    uint8_t *cobs_buffer = NULL;
#endif

    if (!audio_window || !job
#if STREAM_TO_PYTHON
        || !payload || !cobs_buffer
#endif
    ) {
        ESP_LOGE(TAG, "Audio telemetry allocation failed");
        free(audio_window);
        free(job);
        free(payload);
        free(cobs_buffer);
        vTaskDelete(NULL);
        return;
    }

    uint32_t seq = 0;
    while (true) {
        // GetAudioWindow advances by AUDIO_HOP_SAMPLE_COUNT. Inference never runs
        // on this high-priority path, so packet cadence follows the audio clock.
        if (GetAudioWindow(audio_window, AUDIO_SAMPLE_COUNT) != ESP_OK) {
            continue;
        }

        const uint32_t now_us = (uint32_t)esp_timer_get_time();
        job->seq = seq;
        memcpy(job->audio, audio_window, sizeof(job->audio));
        // Queue length is one: replace a stale pending window without blocking.
        xQueueOverwrite(g_classifier_queue, job);

#if STREAM_TO_PYTHON
        LatestClassifierState snapshot;
        xSemaphoreTake(g_result_mutex, portMAX_DELAY);
        snapshot = g_latest_result;
        xSemaphoreGive(g_result_mutex);

        payload->seq = seq;
        payload->audio_timestamp_us = now_us;
        payload->predicted_class = snapshot.predicted_class;
        payload->confidence = snapshot.confidence;
        payload->inference_time_us = snapshot.inference_time_us;
        payload->classifier_seq = snapshot.classifier_seq;
        payload->class_age_ms = snapshot.result_timestamp_us == 0
            ? UINT32_MAX
            : (now_us - snapshot.result_timestamp_us) / 1000U;

        for (size_t i = 0; i < AUDIO_SAMPLE_COUNT; ++i) {
            float sample = audio_window[i];
            if (sample > 1.0f) sample = 1.0f;
            if (sample < -1.0f) sample = -1.0f;
            payload->audio[i] = (int16_t)(sample * 32767.0f);
        }

        const size_t cobs_len = cobs_encode(
            (const uint8_t *)payload, sizeof(TelemetryPayload), cobs_buffer);
        const uint8_t zero = 0;
        uart_write_bytes(UART_NUM_0, &zero, 1);
        uart_write_bytes(UART_NUM_0, cobs_buffer, cobs_len);
        uart_write_bytes(UART_NUM_0, &zero, 1);
#endif
        ++seq;
    }
}

static void classifier_task(void *arg) {
    ESP_LOGI(TAG, "Starting opportunistic classifier task");
    ClassifierJob *job = (ClassifierJob *)malloc(sizeof(ClassifierJob));
    if (!job) {
        ESP_LOGE(TAG, "Classifier job allocation failed");
        vTaskDelete(NULL);
        return;
    }

    while (true) {
        if (xQueueReceive(g_classifier_queue, job, portMAX_DELAY) != pdTRUE) {
            continue;
        }

        ClassifierResult result = {};
        const uint64_t start_us = esp_timer_get_time();
        const esp_err_t err = RunClassifier(job->audio, AUDIO_SAMPLE_COUNT, &result);
        const uint64_t end_us = esp_timer_get_time();
        if (err != ESP_OK) {
            continue;
        }

        LatestClassifierState state = {
            .predicted_class = (uint8_t)result.predicted_class,
            .confidence = result.confidence,
            .inference_time_us = (uint32_t)(end_us - start_us),
            .result_timestamp_us = (uint32_t)end_us,
            .classifier_seq = job->seq,
        };
        xSemaphoreTake(g_result_mutex, portMAX_DELAY);
        g_latest_result = state;
        xSemaphoreGive(g_result_mutex);

#if !STREAM_TO_PYTHON
        ESP_LOGI(TAG, "Pred: [%s] | Conf: %.3f | Inference: %.2f ms",
                 ClassName(result.predicted_class), result.confidence,
                 state.inference_time_us / 1000.0f);
#endif
    }
}

extern "C" void app_main(void) {
#if STREAM_TO_PYTHON
    esp_log_level_set("*", ESP_LOG_NONE);
#endif

    uart_config_t uart_config = {
        .baud_rate = UART_BAUD_RATE,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .rx_flow_ctrl_thresh = 0,
        .source_clk = UART_SCLK_DEFAULT,
    };
    ESP_ERROR_CHECK(uart_driver_install(UART_NUM_0, 4096, 8192, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_param_config(UART_NUM_0, &uart_config));

    esp_err_t nvs_err = nvs_flash_init();
    if (nvs_err == ESP_ERR_NVS_NO_FREE_PAGES || nvs_err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        nvs_err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(nvs_err);

    esp_err_t init_err = InitAudio();
    if (init_err != ESP_OK) {
        ESP_LOGE(TAG, "InitAudio failed during startup self-test: %s", esp_err_to_name(init_err));
        OtaMarkAppInvalidAndRollbackIfPending();
        ESP_ERROR_CHECK(init_err);
    }

    init_err = InitClassifier();
    if (init_err != ESP_OK) {
        ESP_LOGE(TAG, "InitClassifier failed during startup self-test: %s", esp_err_to_name(init_err));
        OtaMarkAppInvalidAndRollbackIfPending();
        ESP_ERROR_CHECK(init_err);
    }

    if (OtaRollbackIsPendingVerify()) {
        init_err = RunClassifierBootSelfTest();
        if (init_err != ESP_OK) {
            ESP_LOGE(TAG, "Classifier boot self-test failed: %s", esp_err_to_name(init_err));
            OtaMarkAppInvalidAndRollbackIfPending();
            ESP_ERROR_CHECK(init_err);
        }
    }
    ESP_ERROR_CHECK(OtaMarkAppValidAfterSuccessfulBoot());

#if !STREAM_TO_PYTHON
    if (connect_wifi(WIFI_SSID, WIFI_PASSWORD) == ESP_OK) {
        ESP_LOGI(TAG, "Wi-Fi connected. Starting background OTA checking task.");
        StartOtaBackgroundChecking();
    } else {
        ESP_LOGW(TAG, "Wi-Fi connection failed. Running offline without OTA updates.");
    }
#endif

    g_result_mutex = xSemaphoreCreateMutex();
    g_classifier_queue = xQueueCreate(1, sizeof(ClassifierJob));
    ESP_ERROR_CHECK(g_result_mutex == NULL ? ESP_ERR_NO_MEM : ESP_OK);
    ESP_ERROR_CHECK(g_classifier_queue == NULL ? ESP_ERR_NO_MEM : ESP_OK);

    BaseType_t task_result = xTaskCreatePinnedToCore(
        audio_telemetry_task, "audio_telemetry_task", 8192, NULL, 7, NULL, 0);
    ESP_ERROR_CHECK(task_result == pdPASS ? ESP_OK : ESP_ERR_NO_MEM);
    task_result = xTaskCreatePinnedToCore(
        classifier_task, "classifier_task", 16384, NULL, 3, NULL, 1);
    ESP_ERROR_CHECK(task_result == pdPASS ? ESP_OK : ESP_ERR_NO_MEM);
}
