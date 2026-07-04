#pragma once

#include "freertos/FreeRTOS.h"
#include "driver/gpio.h"

// ===== Audio config =====
#define SAMPLE_RATE_HZ      8000
#define WINDOW_MS           300
#define AUDIO_SAMPLE_COUNT  ((SAMPLE_RATE_HZ * WINDOW_MS) / 1000)

// Must match training RMS normalization.
// Change this if your Python training code used another target RMS.
#define TARGET_RMS          0.05f

// Gate before RMS normalization.
// If the raw input is too quiet, do not normalize it into fake signal.
#define MIN_RAW_RMS_GATE    0.0005f

// ===== INMP441 I2S pins =====
// Change these to your actual wiring.
#define I2S_BCLK_GPIO       GPIO_NUM_4   // SCK / BCLK
#define I2S_WS_GPIO         GPIO_NUM_5   // WS / LRCLK
#define I2S_DIN_GPIO        GPIO_NUM_6   // SD / DOUT from INMP441

// INMP441 L/R pin:
// L/R = GND usually means left channel.
// L/R = VDD usually means right channel.
#define USE_LEFT_CHANNEL    1

#define UART_BAUD_RATE      2000000
#define ENABLE_SERIAL_MONITOR 1

#define NUM_CLASSES         11
extern const char* kClassNames[NUM_CLASSES];
