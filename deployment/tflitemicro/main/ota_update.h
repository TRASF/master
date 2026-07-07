#pragma once

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

// Run a firmware/code OTA update from an HTTPS URL. Network connectivity must
// already be established by the caller. `server_cert_pem` must contain the
// trusted root/pinned certificate. `server_common_name` is the expected server
// certificate identity; pass a fixed identity such as "termux-ota.local" when
// connecting to a variable DHCP gateway IP.
esp_err_t RunFirmwareOta(
    const char* url,
    const char* server_cert_pem,
    const char* server_common_name
);

// Return true when the running app is an OTA image awaiting first-boot
// validation by the rollback-capable bootloader.
bool OtaRollbackIsPendingVerify(void);

// Mark the running app valid after startup diagnostics/self-test succeed.
esp_err_t OtaMarkAppValidAfterSuccessfulBoot(void);

// If the running app is pending verification, mark it invalid and reboot into
// the previous valid OTA image. If rollback is not pending, this returns.
void OtaMarkAppInvalidAndRollbackIfPending(void);

// Retrieve active model partition index (0 for model_0, 1 for model_1, -1 for embedded g_model)
// and version from NVS.
esp_err_t GetActiveModelConfig(int* active_partition_idx, int* version);

// Save active model partition index and version to NVS.
esp_err_t SetActiveModelConfig(int active_partition_idx, int version);

// Retrieve pending model partition index and version from NVS.
esp_err_t GetPendingModelConfig(int* pending_partition_idx, int* version);

// Save pending model partition index and version to NVS.
esp_err_t SetPendingModelConfig(int pending_partition_idx, int version);

// Clear pending model partition config in NVS.
esp_err_t ClearPendingModelConfig(void);

// Start the background FreeRTOS task to periodically query the server and update firmware or model.
void StartOtaBackgroundChecking(void);

#ifdef __cplusplus
}
#endif
