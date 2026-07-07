#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/uart.h"
#include "esp_log.h"
#include "esp_system.h"

#include "audio_provider.h"
#include "classification.h"
#include "config.h"
#include "ota_update.h"
#include "wifi_helper.h"
#include "nvs_flash.h"

static const char *TAG = "MAIN_APP";


// Tightly pack the ML results and 16-bit audio into a single payload struct
#pragma pack(push, 1)
struct TelemetryPayload {
    uint8_t predicted_class;
    float confidence;
    int16_t audio[AUDIO_SAMPLE_COUNT];
};
#pragma pack(pop)

// -----------------------------------------------------------------------------
// COBS ENCODING ALGORITHM
// -----------------------------------------------------------------------------
size_t cobs_encode(const uint8_t *input, size_t length, uint8_t *output) {
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

// -----------------------------------------------------------------------------
// CAPTURE & INFERENCE TASK
// -----------------------------------------------------------------------------
void main_loop_task(void *arg) {
    ESP_LOGI(TAG, "Starting main ML loop...");

    float *audio_window = (float *)malloc(AUDIO_SAMPLE_COUNT * sizeof(float));

    struct TelemetryPayload *payload = (struct TelemetryPayload *)malloc(sizeof(struct TelemetryPayload));
    size_t max_cobs_len = sizeof(struct TelemetryPayload) + (sizeof(struct TelemetryPayload) / 254) + 2;
    uint8_t *cobs_buffer = (uint8_t *)malloc(max_cobs_len);

    if (!audio_window || !payload || !cobs_buffer) {
        ESP_LOGE(TAG, "Failed to allocate memory!");
        vTaskDelete(NULL);
        return;
    }

    while (1) {
        if (GetAudioWindow(audio_window, AUDIO_SAMPLE_COUNT) == ESP_OK) {
            ClassifierResult result = {};

            if (RunClassifier(audio_window, AUDIO_SAMPLE_COUNT, &result) == ESP_OK) {
#if STREAM_TO_PYTHON
                payload->predicted_class = (uint8_t)result.predicted_class;
                payload->confidence = result.confidence;

                // Convert float audio back to 16-bit PCM for the visualizer
                for (size_t i = 0; i < AUDIO_SAMPLE_COUNT; i++) {
                    // Multiply by 32767 to scale [-1.0, 1.0] to int16 range
                    payload->audio[i] = (int16_t)(audio_window[i] * 32767.0f);
                }

                // Encode COBS and send over UART with 0x00 delimiters
                size_t cobs_len = cobs_encode((const uint8_t *)payload, sizeof(struct TelemetryPayload), cobs_buffer);

                const uint8_t zero = 0x00;
                uart_write_bytes(UART_NUM_0, &zero, 1);
                uart_write_bytes(UART_NUM_0, cobs_buffer, cobs_len);
                uart_write_bytes(UART_NUM_0, &zero, 1);
#else
                ESP_LOGI(TAG, "Pred: [%s] | Conf: %.3f", ClassName(result.predicted_class), result.confidence);
#endif
            }
        }
    }
}

extern "C" void app_main(void) {
#if STREAM_TO_PYTHON
    esp_log_level_set("*", ESP_LOG_NONE);
#endif

    uart_config_t uart_config = {
        .baud_rate = UART_BAUD_RATE,
        .data_bits = UART_DATA_8_BITS,
        .parity    = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .rx_flow_ctrl_thresh = 0,
        .source_clk = UART_SCLK_DEFAULT,
    };

    uart_driver_install(UART_NUM_0, 4096, 8192, 0, NULL, 0);
    uart_param_config(UART_NUM_0, &uart_config);

    // Initialize NVS
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
    // Connect to Wi-Fi and start OTA checking in background
    if (connect_wifi(WIFI_SSID, WIFI_PASSWORD) == ESP_OK) {
        ESP_LOGI(TAG, "Wi-Fi connected. Starting background OTA checking task.");
        StartOtaBackgroundChecking();
    } else {
        ESP_LOGW(TAG, "Wi-Fi connection failed. Running offline without OTA updates.");
    }
#endif

    xTaskCreatePinnedToCore(main_loop_task, "main_loop_task", 8192, NULL, 5, NULL, 1);
}
