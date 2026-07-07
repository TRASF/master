idf_component_register(
    SRCS
        "main.cc"
        "audio_provider.cc"
        "classification.cc"
        "ota_update.cc"
        "wifi_helper.cc"
    INCLUDE_DIRS
        "."
    PRIV_REQUIRES
        esp_driver_i2s
        esp_timer
        esp_driver_gpio
        esp-tflite-micro
        esp_driver_uart
        esp_https_ota
        esp_http_client
        app_update
        nvs_flash
        esp_wifi
        esp_event
        esp_netif
        json
)

# target_compile_options(${COMPONENT_LIB} PRIVATE -DESP_NN)
