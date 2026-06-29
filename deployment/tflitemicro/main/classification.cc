#include "classifier.h"
#include "config.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "esp_log.h"

static const char* TAG = "classifier";

#ifndef NUM_CLASSES
#define NUM_CLASSES 11
#endif

// ============================================================
// Placeholder quantization parameters.
// Later these must come from:
//
// TfLiteTensor* input = interpreter->input(0);
// input->params.scale
// input->params.zero_point
//
// TfLiteTensor* output = interpreter->output(0);
// output->params.scale
// output->params.zero_point
// ============================================================

// Good placeholder for centered RMS-normalized raw waveform.
// Approx range: [-0.25, +0.25]
// scale = 0.5 / 255 = 0.0019607843
static constexpr float kDummyInputScale = 0.0019607843f;
static constexpr int kDummyInputZeroPoint = 0;

// Placeholder output is treated as logit-like scores, not probabilities.
static constexpr float kDummyOutputScale = 0.02f;
static constexpr int kDummyOutputZeroPoint = 0;

// Fake int8 tensors.
// These mimic future TFLite tensors but do not require a real model yet.
static int8_t s_input_tensor[AUDIO_SAMPLE_COUNT];
static int8_t s_output_tensor[NUM_CLASSES];

static const char* kClassNames[NUM_CLASSES] = {
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

const char* ClassName(int class_index) {
    if (class_index < 0 || class_index >= NUM_CLASSES) {
        return "Unknown";
    }

    return kClassNames[class_index];
}

static int ClampInt(int x, int lo, int hi) {
    if (x < lo) {
        return lo;
    }

    if (x > hi) {
        return hi;
    }

    return x;
}

static int QuantizeInputInt8(
    const float* audio_window,
    size_t sample_count,
    int8_t* input_data,
    float scale,
    int zero_point
) {
    int clip_count = 0;

    for (size_t i = 0; i < sample_count; ++i) {
        float x = audio_window[i];

        int q = (int)roundf((x / scale) + (float)zero_point);

        if (q > 127) {
            q = 127;
            clip_count++;
        } else if (q < -128) {
            q = -128;
            clip_count++;
        }

        input_data[i] = (int8_t)q;
    }

    return clip_count;
}

static float DequantizeInt8(
    int8_t q,
    float scale,
    int zero_point
) {
    return ((float)((int)q - zero_point)) * scale;
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

// ============================================================
// Placeholder invoke.
// This mimics output from a real model.
//
// Important:
// This is not a classifier. It only creates fake output so that
// the rest of the inference path can be tested.
// ============================================================
static esp_err_t PlaceholderInvoke(
    const AudioStats* audio_stats,
    int8_t* output_data,
    size_t output_count
) {
    if (audio_stats == nullptr || output_data == nullptr) {
        return ESP_ERR_INVALID_ARG;
    }

    if (output_count != NUM_CLASSES) {
        return ESP_ERR_INVALID_SIZE;
    }

    // Low default logits.
    for (int c = 0; c < NUM_CLASSES; ++c) {
        output_data[c] = -30;
    }

    if (!audio_stats->signal_present) {
        // If RMS gate says no signal, placeholder predicts No_Mos.
        output_data[10] = 80;
    } else {
        // If signal exists, placeholder says "some mosquito-like signal".
        // It does not attempt species/sex classification.
        //
        // All mosquito classes receive similar fake scores.
        // Top-1 will be arbitrary among mosquito classes.
        for (int c = 0; c < 10; ++c) {
            output_data[c] = 20;
        }

        output_data[10] = -10;
    }

    return ESP_OK;
}

esp_err_t InitClassifier() {
    memset(s_input_tensor, 0, sizeof(s_input_tensor));
    memset(s_output_tensor, 0, sizeof(s_output_tensor));

    ESP_LOGW(TAG, "Using placeholder classifier. No real TFLite model is loaded.");
    ESP_LOGI(
        TAG,
        "Fake input tensor: dtype=int8 scale=%.9f zero_point=%d shape=[1,%d,1]",
        kDummyInputScale,
        kDummyInputZeroPoint,
        AUDIO_SAMPLE_COUNT
    );

    ESP_LOGI(
        TAG,
        "Fake output tensor: dtype=int8 scale=%.9f zero_point=%d shape=[1,%d]",
        kDummyOutputScale,
        kDummyOutputZeroPoint,
        NUM_CLASSES
    );

    return ESP_OK;
}

esp_err_t RunClassifier(
    const float* audio_window,
    size_t sample_count,
    const AudioStats* audio_stats,
    ClassifierResult* result
) {
    if (
        audio_window == nullptr ||
        audio_stats == nullptr ||
        result == nullptr
    ) {
        return ESP_ERR_INVALID_ARG;
    }

    if (sample_count != AUDIO_SAMPLE_COUNT) {
        ESP_LOGE(
            TAG,
            "Expected %d input samples, got %u",
            AUDIO_SAMPLE_COUNT,
            (unsigned)sample_count
        );
        return ESP_ERR_INVALID_SIZE;
    }

    memset(result, 0, sizeof(ClassifierResult));

    result->used_dummy_classifier = true;
    result->input_scale = kDummyInputScale;
    result->input_zero_point = kDummyInputZeroPoint;
    result->output_scale = kDummyOutputScale;
    result->output_zero_point = kDummyOutputZeroPoint;

    // 1. Real future step:
    //    Copy preprocessed float waveform into int8 TFLite input tensor.
    result->input_clip_count = QuantizeInputInt8(
        audio_window,
        sample_count,
        s_input_tensor,
        kDummyInputScale,
        kDummyInputZeroPoint
    );

    // 2. Real future step:
    //    interpreter->Invoke()
    //
    // Placeholder for now.
    esp_err_t err = PlaceholderInvoke(
        audio_stats,
        s_output_tensor,
        NUM_CLASSES
    );

    if (err != ESP_OK) {
        return err;
    }

    // 3. Real future step:
    //    Read output tensor and dequantize.
    float output_scores[NUM_CLASSES];

    for (int c = 0; c < NUM_CLASSES; ++c) {
        output_scores[c] = DequantizeInt8(
            s_output_tensor[c],
            kDummyOutputScale,
            kDummyOutputZeroPoint
        );
    }

    // 4. Real future step:
    //    Top-k class selection.
    FillTop3FromScores(
        output_scores,
        NUM_CLASSES,
        result
    );

    return ESP_OK;
}
