idf_component_register(
    SRCS
        "main.cc"
        "audio_provider.cc"
        "classification.cc"
    INCLUDE_DIRS
        "."
    PRIV_REQUIRES
        esp_driver_i2s
        esp_timer
        esp_driver_gpio
        esp-tflite-micro
        esp_driver_uart
)

# Enable ESP-NN optimizations when TFLite Micro inference is added.
target_compile_options(${COMPONENT_LIB} PRIVATE -DESP_NN)
