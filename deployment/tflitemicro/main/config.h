#pragma once

#include "freertos/FreeRTOS.h"
#include "driver/gpio.h"

// ===== Audio config: must match configs/defaults.yaml =====
#define SAMPLE_RATE_HZ          8000
#define WINDOW_MS               300
#define AUDIO_SAMPLE_COUNT      ((SAMPLE_RATE_HZ * WINDOW_MS) / 1000)  // 2400 samples
// Select output/conversion precision for the INMP441 sample path.
// Supported values: 16 or 24.
// Important: INMP441 is a native 24-bit microphone transported in a 32-bit I2S
// slot, so audio_provider.cc always reads 32-bit slots from I2S. BIT_DEPTH only
// selects whether that native 24-bit payload is down-converted to 16-bit
// precision or kept at full 24-bit precision before float normalization.
#define BIT_DEPTH               24

// Training/validation uses augment.segment_overlap.val = 0.5, i.e. a 50% hop.
#define AUDIO_OVERLAP_PERCENT   50
#define AUDIO_HOP_SAMPLE_COUNT  (AUDIO_SAMPLE_COUNT * (100 - AUDIO_OVERLAP_PERCENT) / 100)

// Must match augment.rms_norm in configs/defaults.yaml.
#define TARGET_RMS              0.05f
#define RMS_MIN_GAIN            0.05f
#define RMS_MAX_GAIN            15.0f

// Training does not gate quiet samples before RMS normalization, so keep this
// disabled by default. Set to 1 only as a deployment-specific safety choice.
#define ENABLE_RAW_RMS_GATE     0
#define MIN_RAW_RMS_GATE        0.0005f

// ===== INMP441 I2S pins =====
// Change these to your actual wiring.
#define I2S_BCLK_GPIO           GPIO_NUM_4   // SCK / BCLK
#define I2S_WS_GPIO             GPIO_NUM_5   // WS / LRCLK
#define I2S_DIN_GPIO            GPIO_NUM_6   // SD / DOUT from INMP441

// INMP441 L/R pin:
// L/R = GND usually means left channel.
// L/R = VDD usually means right channel.
#define USE_LEFT_CHANNEL        1

#define STREAM_TO_PYTHON        1
#define UART_BAUD_RATE          2000000
#define ENABLE_SERIAL_MONITOR   0

#define NUM_CLASSES             11
extern const char* kClassNames[NUM_CLASSES];

// ===== OTA and Network Config =====
#define FIRMWARE_VERSION        1

// Wi-Fi Credentials for OTA updates (replace with actual SSID/Password)
#define WIFI_SSID               "WIFI_SSID"
#define WIFI_PASSWORD           "WIFI_PASSWORD"

// Remote HTTPS OTA Server Config
#define OTA_MANIFEST_URL        "https://termux-ota.local/ota_manifest.json"
#define OTA_SERVER_COMMON_NAME  "termux-ota.local"

// Server / CA Certificate PEM (place-holder certificate)
#define OTA_SERVER_CERT_PEM \
"-----BEGIN CERTIFICATE-----\n" \
"MIIBozCCAUqgAwIBAgIQCgEBADANBgkqhkiG9w0BAQsFADAAMB4XDTE2MDEAMDFY\n" \
"MFoXDTI2MDEAMDFYMDAwADAAMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKC\n" \
"AQEA0Gj5bW0K3wK+vO0q6f9hLd2oV2oV2oV2oV2oV2oV2oV2oV2oV2oV2oV2oV2o\n" \
"V2oV2oV2oV2oV2oV2oV2oV2oV2oV2oV2oV2oV2oV2oV2oV2oV2oV2oV2oV2oV2oV\n" \
"2oV2oV2oV2oV2oV2oV2oV2oV2oV2oV2oV2oV2oV2oV2oV2oV2oV2oV2oV2oV2oV2\n" \
"oV2oV2oV2oV2oV2oV2oV2oV2oV2oV2oV2oV2oV2oV2oV2oV2oV2oV2oV2oV2oV2o\n" \
"V2oV2oV2oV2oV2oV2oV2oV2oV2IDAQABo0IwQDAOBgNVHQ8BAf8EBAMCAQYwDwYD\n" \
"VR0TAQH/BAUwAwEB/zAdBgNVHQ4EFgQU5vOQ2w1n/9G2v3v933v933v933wwDQYJ\n" \
"KoZIhvcNAQELBQADggEBAE4U2r2w1n/9G2v3v933v933v933v933v933v933v933\n" \
"v933v933v933v933v933v933v933v933v933v933v933v933v933v933v933v933\n" \
"v933v933v933v933v933v933v933v933v933v933v933v933v933v933v933v933\n" \
"v933v933v933v933v933v933v933v933v933v933v933v933v933v933v933v933\n" \
"v933v933v933v933v933v933v933v933v933v933v933v933\n" \
"-----END CERTIFICATE-----\n"
