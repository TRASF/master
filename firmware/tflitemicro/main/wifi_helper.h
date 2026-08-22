#pragma once

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initialize network interfaces and connect to the specified Wi-Fi network.
// Blocks until connection is established or retries are exhausted.
esp_err_t connect_wifi(const char* ssid, const char* password);

#ifdef __cplusplus
}
#endif
