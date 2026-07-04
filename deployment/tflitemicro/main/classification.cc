#include "classification.h"
#include "config.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "esp_log.h"
#include "esp_heap_caps.h"

// TFLite Micro headers
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model.h"

static const char* TAG = "classifier";

#ifndef NUM_CLASSES
#define NUM_CLASSES 11
#endif

const char* kClassNames[NUM_CLASSES] = {
    "Ae_aegypti_Female",
    "Ae_aegypti_Male",
    "Ae_albopictus_Female",
    "Ae_albopictus_Male",
    "An_dirus_Female",
    "An_dirus_Male",
    "An_minimus_Female",
    "An_minimus_Male",
    "Cx_quin_Female",
    "Cx_quin_Male",
    "No_Mos"
};

// ============================================================
// Real TFLite Micro Global Variables & Arena
// ============================================================
namespace {
    const tflite::Model* s_model = nullptr;
    tflite::MicroInterpreter* s_interpreter = nullptr;
    TfLiteTensor* s_input = nullptr;
    TfLiteTensor* s_output = nullptr;

    // 280KB arena to accommodate the model's requested 268000 bytes.
    constexpr int kTensorArenaSize = 280 * 1024;
    uint8_t* s_tensor_arena = nullptr;
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
    if (s_tensor_arena == nullptr) {
        s_tensor_arena = (uint8_t*)heap_caps_aligned_alloc(16, kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        if (s_tensor_arena == nullptr) {
            ESP_LOGE(TAG, "Failed to allocate %d bytes for tensor arena", kTensorArenaSize);
            return ESP_ERR_NO_MEM;
        }
        ESP_LOGI(TAG, "Allocated %d bytes dynamically for tensor arena (16-byte aligned)", kTensorArenaSize);
    }

    s_model = tflite::GetModel(g_model);
    if (s_model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema version %lu is not supported (expected %d).",
                 s_model->version(), TFLITE_SCHEMA_VERSION);
        return ESP_FAIL;
    }

    static tflite::MicroMutableOpResolver<9> resolver;
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

    static tflite::MicroInterpreter static_interpreter(
        s_model, resolver, s_tensor_arena, kTensorArenaSize);
    s_interpreter = &static_interpreter;

    TfLiteStatus allocate_status = s_interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed. Try increasing kTensorArenaSize.");
        return ESP_FAIL;
    }

    s_input = s_interpreter->input(0);
    s_output = s_interpreter->output(0);

    ESP_LOGI(TAG, "Real TFLite model loaded successfully.");
    ESP_LOGI(TAG, "Input tensor: type=%d, scale=%.6f, zero_point=%d, size=%d bytes",
             s_input->type, s_input->params.scale, s_input->params.zero_point, s_input->bytes);
    ESP_LOGI(TAG, "Output tensor: type=%d, scale=%.6f, zero_point=%d, size=%d bytes",
             s_output->type, s_output->params.scale, s_output->params.zero_point, s_output->bytes);

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
