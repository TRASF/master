#include "classification.h"
#include "config.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_partition.h"
#include "ota_update.h"

// TFLite Micro headers
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model.h"

static const char* TAG = "classifier";
static esp_partition_mmap_handle_t s_model_map_handle;
static const void* s_model_mapped_ptr = nullptr;

#ifndef NUM_CLASSES
#define NUM_CLASSES 11
#endif

const char* kClassNames[NUM_CLASSES] = {
    "Ae_aegypti_Female",
    "Ae_aegypti_Male",
    // "Ae_albopictus_Female",
    // "Ae_albopictus_Male",
    // "An_dirus_Female",
    // "An_dirus_Male",
    // "An_minimus_Female",
    // "An_minimus_Male",
    // "Cx_quin_Female",
    // "Cx_quin_Male",
    "No.Mos"
};

// ============================================================
// Real TFLite Micro Global Variables & Arena
// ============================================================
#include "esp_attr.h"

namespace {
    const tflite::Model* s_model = nullptr;
    tflite::MicroInterpreter* s_interpreter = nullptr;
    TfLiteTensor* s_input = nullptr;
    TfLiteTensor* s_output = nullptr;

    // The current baseline model needs about 268 KB during AllocateTensors().
    // Keep margin for allocator alignment and future small model changes.
    constexpr int kTensorArenaSize = 320 * 1024;
    uint8_t* s_tensor_arena = nullptr;

    uint8_t* AllocateTensorArena() {
        if (s_tensor_arena != nullptr) {
            return s_tensor_arena;
        }

#ifdef CONFIG_SPIRAM
        s_tensor_arena = static_cast<uint8_t*>(
            heap_caps_aligned_alloc(16, kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
        if (s_tensor_arena != nullptr) {
            ESP_LOGI(TAG, "Allocated tensor arena in PSRAM: %d bytes", kTensorArenaSize);
            return s_tensor_arena;
        }
        ESP_LOGW(TAG, "PSRAM tensor arena allocation failed; trying internal heap");
#endif

        s_tensor_arena = static_cast<uint8_t*>(
            heap_caps_aligned_alloc(16, kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
        if (s_tensor_arena != nullptr) {
            ESP_LOGI(TAG, "Allocated tensor arena in internal heap: %d bytes", kTensorArenaSize);
            return s_tensor_arena;
        }

        ESP_LOGE(TAG, "Failed to allocate tensor arena (%d bytes). Largest free block: internal=%u, psram=%u",
                 kTensorArenaSize,
                 heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
                 heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
        return nullptr;
    }
}

const char* ClassName(int class_index) {
    if (class_index < 0 || class_index >= NUM_CLASSES) {
        return "Unknown";
    }
    return kClassNames[class_index];
}

static int QuantizeInput(
    const float* audio_window,
    size_t sample_count,
    TfLiteTensor* input_tensor
) {
    int clip_count = 0;
    if (input_tensor->type == kTfLiteInt8) {
        int8_t* input_data = input_tensor->data.int8;
        float scale = input_tensor->params.scale;
        int zero_point = input_tensor->params.zero_point;
        float inv_scale = (scale != 0.0f) ? (1.0f / scale) : 1.0f;

        for (size_t i = 0; i < sample_count; ++i) {
            float x = audio_window[i];
            int q = (int)roundf((x * inv_scale) + (float)zero_point);
            if (q > 127) {
                q = 127;
                clip_count++;
            } else if (q < -128) {
                q = -128;
                clip_count++;
            }
            input_data[i] = (int8_t)q;
        }
    } else if (input_tensor->type == kTfLiteFloat32) {
        float* input_data = input_tensor->data.f;
        for (size_t i = 0; i < sample_count; ++i) {
            input_data[i] = audio_window[i];
        }
    } else if (input_tensor->type == kTfLiteUInt8) {
        uint8_t* input_data = input_tensor->data.uint8;
        float scale = input_tensor->params.scale;
        int zero_point = input_tensor->params.zero_point;
        float inv_scale = (scale != 0.0f) ? (1.0f / scale) : 1.0f;

        for (size_t i = 0; i < sample_count; ++i) {
            float x = audio_window[i];
            int q = (int)roundf((x * inv_scale) + (float)zero_point);
            if (q > 255) {
                q = 255;
                clip_count++;
            } else if (q < 0) {
                q = 0;
                clip_count++;
            }
            input_data[i] = (uint8_t)q;
        }
    } else {
        ESP_LOGE(TAG, "Unsupported input tensor type: %d", input_tensor->type);
    }
    return clip_count;
}

static float DequantizeOutput(const TfLiteTensor* output_tensor, int class_index) {
    if (output_tensor->type == kTfLiteInt8) {
        int8_t q = output_tensor->data.int8[class_index];
        float scale = output_tensor->params.scale;
        int zero_point = output_tensor->params.zero_point;
        return ((float)((int)q - zero_point)) * scale;
    } else if (output_tensor->type == kTfLiteFloat32) {
        return output_tensor->data.f[class_index];
    } else if (output_tensor->type == kTfLiteUInt8) {
        uint8_t q = output_tensor->data.uint8[class_index];
        float scale = output_tensor->params.scale;
        int zero_point = output_tensor->params.zero_point;
        return ((float)((int)q - zero_point)) * scale;
    } else {
        return 0.0f;
    }
}

static int TensorElementCount(const TfLiteTensor* tensor) {
    if (tensor == nullptr || tensor->dims == nullptr) {
        return 0;
    }

    int count = 1;
    for (int i = 0; i < tensor->dims->size; ++i) {
        int dim = tensor->dims->data[i];
        if (dim <= 0) {
            return 0;
        }
        count *= dim;
    }
    return count;
}

static esp_err_t FillInputWithDeterministicZero(TfLiteTensor* input_tensor) {
    if (input_tensor == nullptr) {
        return ESP_ERR_INVALID_ARG;
    }

    const int input_elements = TensorElementCount(input_tensor);
    if (input_elements != AUDIO_SAMPLE_COUNT) {
        ESP_LOGE(TAG, "Boot self-test input element mismatch: expected %d, got %d",
                 AUDIO_SAMPLE_COUNT, input_elements);
        return ESP_ERR_INVALID_SIZE;
    }

    if (input_tensor->type == kTfLiteInt8) {
        int q = input_tensor->params.zero_point;
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        for (int i = 0; i < input_elements; ++i) {
            input_tensor->data.int8[i] = (int8_t)q;
        }
    } else if (input_tensor->type == kTfLiteUInt8) {
        int q = input_tensor->params.zero_point;
        if (q > 255) q = 255;
        if (q < 0) q = 0;
        for (int i = 0; i < input_elements; ++i) {
            input_tensor->data.uint8[i] = (uint8_t)q;
        }
    } else if (input_tensor->type == kTfLiteFloat32) {
        for (int i = 0; i < input_elements; ++i) {
            input_tensor->data.f[i] = 0.0f;
        }
    } else {
        ESP_LOGE(TAG, "Boot self-test unsupported input tensor type: %d", input_tensor->type);
        return ESP_ERR_NOT_SUPPORTED;
    }

    return ESP_OK;
}

static esp_err_t ValidateSelfTestOutput(const TfLiteTensor* output_tensor, int expected_elements) {
    if (output_tensor == nullptr) {
        return ESP_ERR_INVALID_ARG;
    }

    switch (output_tensor->type) {
        case kTfLiteFloat32:
            for (int i = 0; i < expected_elements; ++i) {
                if (!isfinite(output_tensor->data.f[i])) {
                    ESP_LOGE(TAG, "Boot self-test non-finite float output at index %d", i);
                    return ESP_FAIL;
                }
            }
            return ESP_OK;

        case kTfLiteInt8:
            if (!isfinite(output_tensor->params.scale)) {
                ESP_LOGE(TAG, "Boot self-test invalid int8 output scale");
                return ESP_FAIL;
            }
            for (int i = 0; i < expected_elements; ++i) {
                float score = DequantizeOutput(output_tensor, i);
                if (!isfinite(score)) {
                    ESP_LOGE(TAG, "Boot self-test non-finite dequantized int8 output at index %d", i);
                    return ESP_FAIL;
                }
            }
            return ESP_OK;

        case kTfLiteUInt8:
            if (!isfinite(output_tensor->params.scale)) {
                ESP_LOGE(TAG, "Boot self-test invalid uint8 output scale");
                return ESP_FAIL;
            }
            for (int i = 0; i < expected_elements; ++i) {
                float score = DequantizeOutput(output_tensor, i);
                if (!isfinite(score)) {
                    ESP_LOGE(TAG, "Boot self-test non-finite dequantized uint8 output at index %d", i);
                    return ESP_FAIL;
                }
            }
            return ESP_OK;

        default:
            ESP_LOGE(TAG, "Boot self-test unsupported output tensor type: %d", output_tensor->type);
            return ESP_ERR_NOT_SUPPORTED;
    }
}

static void FillTop3FromScores(
    const float* scores,
    int num_scores,
    ClassifierResult* result
) {
    for (int i = 0; i < 3; ++i) {
        result->top_class[i] = -1;
        result->top_score[i] = -1.0e30f;
    }

    for (int c = 0; c < num_scores; ++c) {
        float s = scores[c];
        for (int k = 0; k < 3; ++k) {
            if (s > result->top_score[k]) {
                for (int j = 2; j > k; --j) {
                    result->top_score[j] = result->top_score[j - 1];
                    result->top_class[j] = result->top_class[j - 1];
                }
                result->top_score[k] = s;
                result->top_class[k] = c;
                break;
            }
        }
    }

    result->predicted_class = result->top_class[0];
    result->confidence = result->top_score[0];
}

esp_err_t InitClassifier() {
    ESP_LOGI(TAG, "Using tensor arena of %d bytes", kTensorArenaSize);

    uint8_t* tensor_arena = AllocateTensorArena();
    if (tensor_arena == nullptr) {
        return ESP_ERR_NO_MEM;
    }

    int active_idx = -1;
    int active_ver = 0;
    int pending_idx = -1;
    int pending_ver = 0;

    GetActiveModelConfig(&active_idx, &active_ver);
    GetPendingModelConfig(&pending_idx, &pending_ver);

    const esp_partition_t* partition_to_load = nullptr;
    bool loading_pending = false;

    if (pending_idx != -1) {
        loading_pending = true;
        partition_to_load = esp_partition_find_first((esp_partition_type_t)0x40, (esp_partition_subtype_t)pending_idx,
                                                     (pending_idx == 0) ? "model_0" : "model_1");
        ESP_LOGI(TAG, "Booting with pending model partition %s (version %d) for verification...",
                 partition_to_load ? partition_to_load->label : "NULL", pending_ver);
    } else if (active_idx != -1) {
        partition_to_load = esp_partition_find_first((esp_partition_type_t)0x40, (esp_partition_subtype_t)active_idx,
                                                     (active_idx == 0) ? "model_0" : "model_1");
        ESP_LOGI(TAG, "Booting with active model partition %s (version %d)",
                 partition_to_load ? partition_to_load->label : "NULL", active_ver);
    }

    esp_err_t map_err = ESP_FAIL;
    if (partition_to_load != nullptr) {
        map_err = esp_partition_mmap(partition_to_load, 0, partition_to_load->size,
                                     ESP_PARTITION_MMAP_DATA, &s_model_mapped_ptr, &s_model_map_handle);
        if (map_err == ESP_OK) {
            s_model = tflite::GetModel(s_model_mapped_ptr);
        } else {
            ESP_LOGE(TAG, "Failed to map model partition: %s", esp_err_to_name(map_err));
        }
    }

    if (map_err != ESP_OK) {
        ESP_LOGI(TAG, "Using embedded baseline model");
        s_model = tflite::GetModel(g_model);
        s_model_mapped_ptr = nullptr;
    }

    if (s_model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema version %lu is not supported (expected %d).",
                 s_model->version(), TFLITE_SCHEMA_VERSION);
        if (loading_pending) {
            ESP_LOGE(TAG, "Pending model version unsupported. Clearing pending and rolling back.");
            ClearPendingModelConfig();
            esp_restart();
        }
        return ESP_FAIL;
    }

    static tflite::MicroMutableOpResolver<9> resolver;
    static bool resolver_init = false;
    if (!resolver_init) {
        if (resolver.AddExpandDims() != kTfLiteOk ||
            resolver.AddConv2D() != kTfLiteOk ||
            resolver.AddReshape() != kTfLiteOk ||
            resolver.AddMaxPool2D() != kTfLiteOk ||
            resolver.AddShape() != kTfLiteOk ||
            resolver.AddStridedSlice() != kTfLiteOk ||
            resolver.AddPack() != kTfLiteOk ||
            resolver.AddFullyConnected() != kTfLiteOk ||
            resolver.AddHardSwish() != kTfLiteOk) {
            ESP_LOGE(TAG, "Failed to register operators.");
            return ESP_FAIL;
        }
        resolver_init = true;
    }

    static tflite::MicroInterpreter static_interpreter(
        s_model, resolver, tensor_arena, kTensorArenaSize);
    s_interpreter = &static_interpreter;

    TfLiteStatus allocate_status = s_interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed.");
        if (loading_pending) {
            ESP_LOGE(TAG, "Pending model AllocateTensors failed. Clearing pending and rolling back.");
            ClearPendingModelConfig();
            esp_restart();
        }
        return ESP_FAIL;
    }

    s_input = s_interpreter->input(0);
    s_output = s_interpreter->output(0);

    ESP_LOGI(TAG, "Real TFLite model loaded successfully.");
    ESP_LOGI(TAG, "Input tensor: type=%d, scale=%.10f, zero_point=%ld, size=%u bytes",
             s_input->type,
             (double)s_input->params.scale,
             (long)s_input->params.zero_point,
             (unsigned)s_input->bytes);
    ESP_LOGI(TAG, "Output tensor: type=%d, scale=%.10f, zero_point=%ld, size=%u bytes",
             s_output->type,
             (double)s_output->params.scale,
             (long)s_output->params.zero_point,
             (unsigned)s_output->bytes);

    if (loading_pending) {
        esp_err_t test_err = RunClassifierBootSelfTest();
        if (test_err == ESP_OK) {
            ESP_LOGI(TAG, "Pending model verification passed! Setting as active model.");
            SetActiveModelConfig(pending_idx, pending_ver);
            ClearPendingModelConfig();
        } else {
            ESP_LOGE(TAG, "Pending model verification failed. Clearing pending and rolling back.");
            ClearPendingModelConfig();
            esp_restart();
        }
    }

    return ESP_OK;
}

esp_err_t RunClassifierBootSelfTest() {
    if (s_model == nullptr || s_interpreter == nullptr || s_input == nullptr || s_output == nullptr) {
        ESP_LOGE(TAG, "Boot self-test failed: classifier not initialized");
        return ESP_ERR_INVALID_STATE;
    }

    if (s_model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Boot self-test schema mismatch: got %lu expected %d",
                 s_model->version(), TFLITE_SCHEMA_VERSION);
        return ESP_FAIL;
    }

    const int input_elements = TensorElementCount(s_input);
    const int output_elements = TensorElementCount(s_output);
    if (input_elements != AUDIO_SAMPLE_COUNT) {
        ESP_LOGE(TAG, "Boot self-test input element mismatch: expected %d, got %d",
                 AUDIO_SAMPLE_COUNT, input_elements);
        return ESP_ERR_INVALID_SIZE;
    }
    if (output_elements != NUM_CLASSES) {
        ESP_LOGE(TAG, "Boot self-test output class mismatch: expected %d, got %d",
                 NUM_CLASSES, output_elements);
        return ESP_ERR_INVALID_SIZE;
    }

    esp_err_t err = FillInputWithDeterministicZero(s_input);
    if (err != ESP_OK) {
        return err;
    }

    if (s_interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Boot self-test Invoke failed");
        return ESP_FAIL;
    }

    err = ValidateSelfTestOutput(s_output, NUM_CLASSES);
    if (err != ESP_OK) {
        return err;
    }

    ESP_LOGI(TAG, "Classifier boot self-test passed");
    return ESP_OK;
}

esp_err_t RunClassifier(
    const float* audio_window,
    size_t sample_count,
    ClassifierResult* result
) {
    if (audio_window == nullptr || result == nullptr) {
        return ESP_ERR_INVALID_ARG;
    }

    memset(result, 0, sizeof(ClassifierResult));

    if (sample_count != AUDIO_SAMPLE_COUNT) {
        ESP_LOGE(TAG, "Expected %d input samples, got %u", AUDIO_SAMPLE_COUNT, (unsigned)sample_count);
        return ESP_ERR_INVALID_SIZE;
    }

    if (s_interpreter == nullptr || s_input == nullptr || s_output == nullptr) {
        ESP_LOGE(TAG, "Real classifier not initialized!");
        return ESP_ERR_INVALID_STATE;
    }

    result->input_scale = s_input->params.scale;
    result->input_zero_point = s_input->params.zero_point;
    result->output_scale = s_output->params.scale;
    result->output_zero_point = s_output->params.zero_point;

    // 1. Quantize Float Audio to Int8
    result->input_clip_count = QuantizeInput(audio_window, sample_count, s_input);

    // 2. Run Inference
    TfLiteStatus invoke_status = s_interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        ESP_LOGE(TAG, "Interpreter Invoke failed");
        return ESP_FAIL;
    }

    // 3. Dequantize Int8 to Float Scores
    float output_scores[NUM_CLASSES];
    for (int c = 0; c < NUM_CLASSES; ++c) {
        output_scores[c] = DequantizeOutput(s_output, c);
    }

    // 4. Find Top 3 Predictions
    FillTop3FromScores(output_scores, NUM_CLASSES, result);

    return ESP_OK;
}
