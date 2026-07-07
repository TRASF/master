#include "ota_update.h"

#include <string.h>
#include "esp_err.h"
#include "esp_https_ota.h"
#include "esp_http_client.h"
#include "esp_log.h"
#include "esp_ota_ops.h"
#include "esp_system.h"
#include "esp_partition.h"
#include "nvs.h"
#include "nvs_flash.h"
#include "cJSON.h"
#include "config.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char* TAG = "ota_update";

// -----------------------------------------------------------------------------
// FIRMWARE OTA IMPLEMENTATION
// -----------------------------------------------------------------------------
esp_err_t RunFirmwareOta(
    const char* url,
    const char* server_cert_pem,
    const char* server_common_name
) {
    if (url == nullptr || url[0] == '\0') {
        ESP_LOGE(TAG, "Firmware OTA URL is empty");
        return ESP_ERR_INVALID_ARG;
    }

    const bool is_https = (strncmp(url, "https://", 8) == 0);
    if (!is_https) {
        ESP_LOGE(TAG, "Firmware OTA URL must use https://");
        return ESP_ERR_INVALID_ARG;
    }

    if (server_cert_pem == nullptr || server_cert_pem[0] == '\0') {
        ESP_LOGE(TAG, "Refusing HTTPS OTA without a trusted server certificate");
        return ESP_ERR_INVALID_ARG;
    }

    if (server_common_name == nullptr || server_common_name[0] == '\0') {
        ESP_LOGE(TAG, "Refusing HTTPS OTA without an expected server identity");
        return ESP_ERR_INVALID_ARG;
    }

    ESP_LOGI(TAG, "Starting firmware OTA from %s; expected TLS identity: %s", url, server_common_name);

    esp_http_client_config_t http_config = {};
    http_config.url = url;
    http_config.cert_pem = server_cert_pem;
    http_config.common_name = server_common_name;
    http_config.skip_cert_common_name_check = false;
    http_config.timeout_ms = 15000;
    http_config.keep_alive_enable = true;

    esp_https_ota_config_t ota_config = {};
    ota_config.http_config = &http_config;

    esp_err_t err = esp_https_ota(&ota_config);
    if (err == ESP_OK) {
        ESP_LOGI(TAG, "Firmware OTA succeeded; restarting");
        esp_restart();
    }

    ESP_LOGE(TAG, "Firmware OTA failed: %s", esp_err_to_name(err));
    return err;
}

static esp_err_t GetRunningOtaState(esp_ota_img_states_t* ota_state) {
    if (ota_state == nullptr) {
        return ESP_ERR_INVALID_ARG;
    }

    const esp_partition_t* running = esp_ota_get_running_partition();
    if (running == nullptr) {
        return ESP_FAIL;
    }

    return esp_ota_get_state_partition(running, ota_state);
}

bool OtaRollbackIsPendingVerify(void) {
    esp_ota_img_states_t ota_state = ESP_OTA_IMG_UNDEFINED;
    esp_err_t err = GetRunningOtaState(&ota_state);
    if (err != ESP_OK) {
        return false;
    }

    return ota_state == ESP_OTA_IMG_PENDING_VERIFY;
}

esp_err_t OtaMarkAppValidAfterSuccessfulBoot(void) {
    esp_ota_img_states_t ota_state = ESP_OTA_IMG_UNDEFINED;
    esp_err_t err = GetRunningOtaState(&ota_state);
    if (err != ESP_OK) {
        return err;
    }

    if (ota_state != ESP_OTA_IMG_PENDING_VERIFY) {
        return ESP_OK;
    }

    ESP_LOGI(TAG, "OTA image self-test passed; marking app valid");
    err = esp_ota_mark_app_valid_cancel_rollback();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to mark OTA app valid: %s", esp_err_to_name(err));
    }
    return err;
}

void OtaMarkAppInvalidAndRollbackIfPending(void) {
    if (!OtaRollbackIsPendingVerify()) {
        return;
    }

    ESP_LOGE(TAG, "OTA image self-test failed; marking app invalid and rolling back");
    esp_err_t err = esp_ota_mark_app_invalid_rollback_and_reboot();
    ESP_LOGE(TAG, "Rollback request failed: %s", esp_err_to_name(err));
}

// -----------------------------------------------------------------------------
// NVS ACTIVE/PENDING MODEL CONFIGURATION
// -----------------------------------------------------------------------------
esp_err_t GetActiveModelConfig(int* active_partition_idx, int* version) {
    nvs_handle_t handle;
    esp_err_t err = nvs_open("ota_model", NVS_READONLY, &handle);
    if (err != ESP_OK) {
        *active_partition_idx = -1; // Fallback to embedded g_model
        *version = 0;
        return err;
    }

    int32_t act = -1;
    int32_t ver = 0;
    nvs_get_i32(handle, "active_idx", &act);
    nvs_get_i32(handle, "version", &ver);
    nvs_close(handle);

    *active_partition_idx = (int)act;
    *version = (int)ver;
    return ESP_OK;
}

esp_err_t SetActiveModelConfig(int active_partition_idx, int version) {
    nvs_handle_t handle;
    esp_err_t err = nvs_open("ota_model", NVS_READWRITE, &handle);
    if (err != ESP_OK) {
        return err;
    }

    nvs_set_i32(handle, "active_idx", active_partition_idx);
    nvs_set_i32(handle, "version", version);
    err = nvs_commit(handle);
    nvs_close(handle);
    return err;
}

esp_err_t GetPendingModelConfig(int* pending_partition_idx, int* version) {
    nvs_handle_t handle;
    esp_err_t err = nvs_open("ota_model", NVS_READONLY, &handle);
    if (err != ESP_OK) {
        *pending_partition_idx = -1;
        *version = 0;
        return err;
    }

    int32_t pend = -1;
    int32_t ver = 0;
    nvs_get_i32(handle, "pending_idx", &pend);
    nvs_get_i32(handle, "pending_ver", &ver);
    nvs_close(handle);

    *pending_partition_idx = (int)pend;
    *version = (int)ver;
    return ESP_OK;
}

esp_err_t SetPendingModelConfig(int pending_partition_idx, int version) {
    nvs_handle_t handle;
    esp_err_t err = nvs_open("ota_model", NVS_READWRITE, &handle);
    if (err != ESP_OK) {
        return err;
    }

    nvs_set_i32(handle, "pending_idx", pending_partition_idx);
    nvs_set_i32(handle, "pending_ver", version);
    err = nvs_commit(handle);
    nvs_close(handle);
    return err;
}

esp_err_t ClearPendingModelConfig(void) {
    nvs_handle_t handle;
    esp_err_t err = nvs_open("ota_model", NVS_READWRITE, &handle);
    if (err != ESP_OK) {
        return err;
    }

    nvs_erase_key(handle, "pending_idx");
    nvs_erase_key(handle, "pending_ver");
    err = nvs_commit(handle);
    nvs_close(handle);
    return err;
}

// -----------------------------------------------------------------------------
// MODEL DOWNLOAD IMPLEMENTATION
// -----------------------------------------------------------------------------
static esp_err_t DownloadModel(
    const char* url,
    const esp_partition_t* partition,
    const char* cert_pem,
    const char* common_name
) {
    if (url == nullptr || partition == nullptr) {
        return ESP_ERR_INVALID_ARG;
    }

    ESP_LOGI(TAG, "Downloading model from %s to partition %s...", url, partition->label);

    esp_http_client_config_t config = {};
    config.url = url;
    config.cert_pem = cert_pem;
    config.common_name = common_name;
    config.skip_cert_common_name_check = false;
    config.timeout_ms = 15000;
    config.keep_alive_enable = true;

    esp_http_client_handle_t client = esp_http_client_init(&config);
    if (!client) {
        ESP_LOGE(TAG, "Failed to initialize HTTP client");
        return ESP_FAIL;
    }

    esp_err_t err = esp_http_client_open(client, 0);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to open HTTP connection: %s", esp_err_to_name(err));
        esp_http_client_cleanup(client);
        return err;
    }

    int content_length = esp_http_client_fetch_headers(client);
    if (content_length <= 0) {
        ESP_LOGE(TAG, "Failed to fetch headers or invalid content length: %d", content_length);
        esp_http_client_cleanup(client);
        return ESP_FAIL;
    }

    if (content_length > partition->size) {
        ESP_LOGE(TAG, "Model size (%d) exceeds partition size (%d)", content_length, partition->size);
        esp_http_client_cleanup(client);
        return ESP_ERR_INVALID_SIZE;
    }

    // Erase model partition
    ESP_LOGI(TAG, "Erasing partition %s (%d bytes)...", partition->label, partition->size);
    err = esp_partition_erase_range(partition, 0, partition->size);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to erase partition: %s", esp_err_to_name(err));
        esp_http_client_cleanup(client);
        return err;
    }

    char* buffer = (char*)malloc(4096);
    if (!buffer) {
        ESP_LOGE(TAG, "Failed to allocate download buffer");
        esp_http_client_cleanup(client);
        return ESP_ERR_NO_MEM;
    }

    int write_offset = 0;
    int read_len;
    while ((read_len = esp_http_client_read(client, buffer, 4096)) > 0) {
        err = esp_partition_write(partition, write_offset, buffer, read_len);
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "Failed to write partition: %s", esp_err_to_name(err));
            free(buffer);
            esp_http_client_cleanup(client);
            return err;
        }
        write_offset += read_len;
    }

    free(buffer);
    esp_http_client_cleanup(client);

    if (write_offset != content_length) {
        ESP_LOGE(TAG, "Downloaded model size %d mismatch expected size %d", write_offset, content_length);
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "Successfully downloaded and wrote model (%d bytes)", write_offset);
    return ESP_OK;
}

// -----------------------------------------------------------------------------
// HTTP GET FETCH MANIFEST & PARSE MANIFEST
// -----------------------------------------------------------------------------
static esp_err_t FetchManifest(
    const char* url,
    const char* cert_pem,
    const char* common_name,
    char* buffer,
    size_t buffer_max
) {
    esp_http_client_config_t config = {};
    config.url = url;
    config.cert_pem = cert_pem;
    config.common_name = common_name;
    config.skip_cert_common_name_check = false;
    config.timeout_ms = 15000;

    esp_http_client_handle_t client = esp_http_client_init(&config);
    if (!client) {
        return ESP_FAIL;
    }

    esp_err_t err = esp_http_client_open(client, 0);
    if (err != ESP_OK) {
        esp_http_client_cleanup(client);
        return err;
    }

    int content_length = esp_http_client_fetch_headers(client);
    if (content_length >= (int)buffer_max) {
        ESP_LOGE(TAG, "Manifest size %d exceeds buffer size %d", content_length, buffer_max);
        esp_http_client_cleanup(client);
        return ESP_ERR_INVALID_SIZE;
    }

    int read_len = esp_http_client_read(client, buffer, buffer_max - 1);
    if (read_len >= 0) {
        buffer[read_len] = '\0';
        err = ESP_OK;
    } else {
        err = ESP_FAIL;
    }

    esp_http_client_cleanup(client);
    return err;
}

static esp_err_t ParseManifest(
    const char* json_str,
    int* fw_ver,
    char* fw_url,
    size_t fw_url_max,
    int* model_ver,
    char* model_url,
    size_t model_url_max
) {
    cJSON* root = cJSON_Parse(json_str);
    if (!root) {
        ESP_LOGE(TAG, "Manifest JSON parsing failed");
        return ESP_FAIL;
    }

    cJSON* fw_node = cJSON_GetObjectItem(root, "firmware");
    if (fw_node) {
        cJSON* ver_node = cJSON_GetObjectItem(fw_node, "version");
        cJSON* url_node = cJSON_GetObjectItem(fw_node, "url");
        if (ver_node && url_node) {
            *fw_ver = ver_node->valueint;
            strncpy(fw_url, url_node->valuestring, fw_url_max - 1);
            fw_url[fw_url_max - 1] = '\0';
        }
    }

    cJSON* model_node = cJSON_GetObjectItem(root, "model");
    if (model_node) {
        cJSON* ver_node = cJSON_GetObjectItem(model_node, "version");
        cJSON* url_node = cJSON_GetObjectItem(model_node, "url");
        if (ver_node && url_node) {
            *model_ver = ver_node->valueint;
            strncpy(model_url, url_node->valuestring, model_url_max - 1);
            model_url[model_url_max - 1] = '\0';
        }
    }

    cJSON_Delete(root);
    return ESP_OK;
}

// -----------------------------------------------------------------------------
// BACKGROUND POLLING LOOP
// -----------------------------------------------------------------------------
static void ota_polling_task(void* pvParameters) {
    ESP_LOGI(TAG, "Starting periodic OTA polling task...");

    char* manifest_buf = (char*)malloc(2048);
    if (!manifest_buf) {
        ESP_LOGE(TAG, "Failed to allocate manifest download buffer");
        vTaskDelete(NULL);
        return;
    }

    while (1) {
        ESP_LOGI(TAG, "Fetching OTA manifest from %s...", OTA_MANIFEST_URL);
        esp_err_t err = FetchManifest(OTA_MANIFEST_URL, OTA_SERVER_CERT_PEM, OTA_SERVER_COMMON_NAME, manifest_buf, 2048);
        if (err == ESP_OK) {
            int remote_fw_ver = 0;
            char remote_fw_url[256] = {0};
            int remote_model_ver = 0;
            char remote_model_url[256] = {0};

            err = ParseManifest(manifest_buf, &remote_fw_ver, remote_fw_url, sizeof(remote_fw_url),
                                &remote_model_ver, remote_model_url, sizeof(remote_model_url));
            if (err == ESP_OK) {
                // 1. Process Firmware Update
                if (remote_fw_ver > FIRMWARE_VERSION && strlen(remote_fw_url) > 0) {
                    ESP_LOGI(TAG, "New firmware version %d available (current: %d). Starting firmware OTA...",
                             remote_fw_ver, FIRMWARE_VERSION);
                    free(manifest_buf);
                    RunFirmwareOta(remote_fw_url, OTA_SERVER_CERT_PEM, OTA_SERVER_COMMON_NAME);
                    // If OTA fails or returns, we re-allocate manifest buffer and continue
                    manifest_buf = (char*)malloc(2048);
                    if (!manifest_buf) {
                        vTaskDelete(NULL);
                        return;
                    }
                }

                // 2. Process Model Update
                int active_model_idx = -1;
                int active_model_ver = 0;
                GetActiveModelConfig(&active_model_idx, &active_model_ver);

                if (remote_model_ver > active_model_ver && strlen(remote_model_url) > 0) {
                    ESP_LOGI(TAG, "New model version %d available (current: %d). Starting model OTA...",
                             remote_model_ver, active_model_ver);

                    int inactive_idx = (active_model_idx == 0) ? 1 : 0;
                    const esp_partition_t* inactive_partition = esp_partition_find_first(
                        (esp_partition_type_t)0x40, (esp_partition_subtype_t)inactive_idx,
                        (inactive_idx == 0) ? "model_0" : "model_1");

                    if (inactive_partition) {
                        err = DownloadModel(remote_model_url, inactive_partition, OTA_SERVER_CERT_PEM, OTA_SERVER_COMMON_NAME);
                        if (err == ESP_OK) {
                            ESP_LOGI(TAG, "Model download successful. Setting model_pending to partition %d (version %d) and rebooting for verification self-test...",
                                     inactive_idx, remote_model_ver);
                            SetPendingModelConfig(inactive_idx, remote_model_ver);
                            esp_restart();
                        } else {
                            ESP_LOGE(TAG, "Model download failed");
                        }
                    } else {
                        ESP_LOGE(TAG, "Model partition slots not found in partition table");
                    }
                }
            }
        } else {
            ESP_LOGE(TAG, "Failed to fetch manifest: %s", esp_err_to_name(err));
        }

        // Poll every 5 minutes
        vTaskDelay(pdMS_TO_TICKS(5 * 60 * 1000));
    }
}

void StartOtaBackgroundChecking(void) {
    // Priority 2, pinned to Core 0 (where networking/Wi-Fi stack runs)
    xTaskCreatePinnedToCore(ota_polling_task, "ota_polling_task", 8192, NULL, 2, NULL, 0);
}
