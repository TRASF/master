import os
import numpy as np
import soundfile as sf

def generate_fixtures():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "fixtures/audio_11class"))
    os.makedirs(base_dir, exist_ok=True)

    # 11 Classes exactly matching defaults.yaml
    classes = [
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
        "No.mos"
    ]

    sr = 8000
    duration = 1.0  # 1 second
    num_samples = int(sr * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    for i, cls in enumerate(classes):
        cls_dir = os.path.join(base_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)

        filename = f"{cls.lower()}_sample.wav"
        filepath = os.path.join(cls_dir, filename)

        if cls == "No.mos":
            # Deterministic noise (silence / near silence)
            data = np.zeros(num_samples, dtype=np.float32)
        else:
            # Sine wave with frequency depending on index
            freq = 200 + i * 50
            data = np.sin(2 * np.pi * freq * t)

        # Scale to fit standard PCM 16 range nicely
        data = (data * 0.5).astype(np.float32)

        # Write standard 16-bit PCM WAV
        sf.write(filepath, data, sr, subtype='PCM_16')
        print(f"Created standard PCM_16 file: {filepath}")

    # Add one non-8kHz sample (e.g. 16 kHz) in Ae_aegypti_Female to test resampling
    resample_dir = os.path.join(base_dir, "Ae_aegypti_Female")
    resample_path = os.path.join(resample_dir, "ae_aegypti_female_16khz.wav")
    t_16k = np.linspace(0, duration, int(16000 * duration), endpoint=False)
    data_16k = (np.sin(2 * np.pi * 400 * t_16k) * 0.5).astype(np.float32)
    sf.write(resample_path, data_16k, 16000, subtype='PCM_16')
    print(f"Created non-8kHz file: {resample_path}")

    # Add one alternate bit depth sample (e.g. Float32) in Ae_albopictus_Female
    float_dir = os.path.join(base_dir, "Ae_albopictus_Female")
    float_path = os.path.join(float_dir, "ae_albopictus_female_float32.wav")
    data_float = (np.sin(2 * np.pi * 350 * t) * 0.5).astype(np.float32)
    sf.write(float_path, data_float, sr, subtype='FLOAT')
    print(f"Created float32 WAV file: {float_path}")

if __name__ == "__main__":
    generate_fixtures()
