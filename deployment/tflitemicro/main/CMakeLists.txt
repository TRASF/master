idf_component_register(
    SRCS
        "main.cc"
        "audio_provider.cc"
    INCLUDE_DIRS
        "."
    PRIV_REQUIRES
        esp_driver_i2s
        esp_timer
        esp_driver_gpio
        esp-tflite-micro
)

# Enable ESP-NN optimizations later when TFLite Micro inference is added.
# target_compile_options(${COMPONENT_LIB} PRIVATE -DESP_NN)
