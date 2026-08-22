"""Long-recording WAV/PCM inference and event decoding."""

from __future__ import annotations

import re
import wave
from pathlib import Path

import numpy as np
import torch
import torchaudio.transforms as T

from wingbeat_ml.data.synthetic import decode_pcm_bytes
from wingbeat_ml.sed.inference.decoder import DecodedEvent, decode_events
from wingbeat_ml.sed.training.train import FullSEDTeacher


def parse_pcm_filename_meta(filename: str) -> tuple[int, int, int]:
    """Parse sample rate, sample width (bytes), and channels from PCM filename."""
    sr = 44100
    sw = 3
    ch = 1

    m_sr = re.search(r"(?:^|_)sr(\d+)(?=_|\.|$)", filename)
    if m_sr:
        sr = int(m_sr.group(1))

    m_b = re.search(r"(?:^|_)b(\d+)(?=_|\.|$)", filename)
    if m_b:
        sw = int(m_b.group(1)) // 8

    m_c = re.search(r"(?:^|_)c(\d+)(?=_|\.|$)", filename)
    if m_c:
        ch = int(m_c.group(1))

    return sr, sw, ch


def load_long_audio_pcm(
    path: str | Path,
    channel_index: int | None = None,
    target_sample_rate: int = 16_000,
) -> tuple[np.ndarray, int]:
    """Load one physical microphone channel as canonical 16 kHz mono audio."""
    p = Path(path)
    if p.suffix.lower() == ".pcm":
        sr, sw, n_ch = parse_pcm_filename_meta(p.name)
        with p.open("rb") as audio_file:
            signal = decode_pcm_bytes(audio_file.read(), sw)
    else:
        try:
            with wave.open(str(p), "rb") as wav_file:
                sr = wav_file.getframerate()
                n_ch = wav_file.getnchannels()
                sw = wav_file.getsampwidth()
                signal = decode_pcm_bytes(wav_file.readframes(wav_file.getnframes()), sw)
        except wave.Error:
            sr, sw, n_ch = parse_pcm_filename_meta(p.name)
            with p.open("rb") as audio_file:
                signal = decode_pcm_bytes(audio_file.read(), sw)

    if n_ch > 1 and len(signal) >= n_ch:
        signal = signal[: len(signal) - (len(signal) % n_ch)].reshape(-1, n_ch)
        if channel_index is not None:
            if channel_index < 1 or channel_index > n_ch:
                raise ValueError(f"Channel {channel_index} does not exist in {p}")
            signal = signal[:, channel_index - 1]
        else:
            signal = signal[:, int(np.argmax(np.sqrt(np.mean(signal**2, axis=0))))]

    signal_tensor = torch.from_numpy(signal.copy()).float()
    if sr != target_sample_rate:
        signal_tensor = T.Resample(
            sr,
            target_sample_rate,
            resampling_method="sinc_interp_hann",
        )(signal_tensor)
    return signal_tensor.numpy(), target_sample_rate


def _legacy_model_config(state: dict[str, torch.Tensor]) -> dict[str, int | float]:
    gru_weight = state.get("head.gru.weight_ih_l0")
    if gru_weight is None:
        raise ValueError("SED checkpoint is missing head.gru.weight_ih_l0")
    layer_numbers = [
        int(match.group(1))
        for key in state
        if (match := re.fullmatch(r"head\.gru\.weight_ih_l(\d+)", key))
    ]
    input_norm = state.get("head.input_norm.weight")
    input_dim = int(input_norm.shape[0]) if input_norm is not None else 768
    return {
        "n_atst_blocks": max(1, input_dim // 768),
        "conv_dim": int(gru_weight.shape[1]),
        "hidden_dim": int(gru_weight.shape[0] // 3),
        "gru_layers": max(layer_numbers, default=1) + 1,
        "dropout": 0.2,
    }


def load_sed_teacher_model(
    model_checkpoint: str | Path | None = None,
    atst_checkpoint: str | Path | None = None,
    device: str = "cpu",
) -> FullSEDTeacher:
    """Load one strictly validated teacher model for reuse across recordings."""
    state = None
    model_config: dict[str, int | float | str] = {
        "n_atst_blocks": 1,
        "conv_dim": 256,
        "hidden_dim": 256,
        "gru_layers": 2,
        "dropout": 0.2,
        "head_type": "gru",
        "transformer_heads": 4,
        "transformer_layers": 2,
    }

    if model_checkpoint:
        model_path = Path(model_checkpoint)
        if not model_path.is_file():
            raise FileNotFoundError(f"Trained SED checkpoint not found: {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
            model_config.update(checkpoint.get("model_config", {}))
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
            model_config.update(checkpoint.get("model_config", {}))
        elif isinstance(checkpoint, dict):
            state = checkpoint
        else:
            raise ValueError(f"Unsupported SED checkpoint format: {model_path}")
        if "model_config" not in checkpoint:
            model_config.update(_legacy_model_config(state))

    model = FullSEDTeacher(
        atst_checkpoint=atst_checkpoint or "checkpoints/atstframe_base.ckpt",
        n_atst_blocks=int(model_config["n_atst_blocks"]),
        conv_dim=int(model_config["conv_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        gru_layers=int(model_config["gru_layers"]),
        dropout=float(model_config["dropout"]),
        freeze_encoder=True,
        head_type=str(model_config.get("head_type", "gru")),
        transformer_heads=int(model_config.get("transformer_heads", 4)),
        transformer_layers=int(model_config.get("transformer_layers", 2)),
    )
    if state is not None:
        model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def infer_long_recording(
    wav_path: str | Path,
    model: FullSEDTeacher | None = None,
    model_checkpoint: str | Path | None = None,
    atst_checkpoint: str | Path | None = None,
    verifier_checkpoint: str | Path | None = None,
    chunk_len_s: float = 10.0,
    hop_len_s: float = 5.0,
    frame_rate_hz: float = 25.0,
    batch_size: int = 32,
    device: str = "cpu",
    channel_index: int | None = None,
    high_threshold: float = 0.8,
    low_threshold: float = 0.4,
    min_duration_s: float = 0.1,
    max_merge_gap_s: float = 0.2,
    auto_accept_threshold: float | None = None,  # Keep None / disabled until calibrated LCB >= 0.995
) -> list[DecodedEvent]:
    """Run batched 4 s sliding-window SED proposal generation + optional Stage 2 verification."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    signal, sr = load_long_audio_pcm(wav_path, channel_index=channel_index, target_sample_rate=16_000)
    total_frames = int(np.ceil(len(signal) / float(sr) * frame_rate_hz))
    if total_frames == 0:
        return []

    # Calculate recording-level bounded gain once for entire signal
    from wingbeat_ml.sed.data.dataset import compute_recording_bounded_gain
    rec_gain = compute_recording_bounded_gain(signal, target_rms=0.05)
    signal = signal * rec_gain

    chunk_samples = int(chunk_len_s * sr)
    hop_samples = int(hop_len_s * sr)
    chunk_frames = int(chunk_len_s * frame_rate_hz)
    if chunk_samples < 1 or hop_samples < 1 or chunk_frames < 1:
        raise ValueError("chunk and hop lengths must be positive")

    head = model or load_sed_teacher_model(
        model_checkpoint=model_checkpoint,
        atst_checkpoint=atst_checkpoint,
        device=device,
    )
    probability_sum = np.zeros(total_frames, dtype=np.float32)
    weight_sum = np.zeros(total_frames, dtype=np.float32)
    window = np.maximum(
        np.hanning(chunk_frames + 2)[1:-1].astype(np.float32),
        1e-3,
    )
    starts = range(0, len(signal), hop_samples)
    device_type = torch.device(device).type

    with torch.inference_mode():
        for batch_start in range(0, len(starts), batch_size):
            batch_starts = list(starts[batch_start : batch_start + batch_size])
            chunks = []
            valid_sample_counts = []
            for start_sample in batch_starts:
                chunk = signal[start_sample : start_sample + chunk_samples]
                valid_sample_counts.append(len(chunk))
                if len(chunk) < chunk_samples:
                    chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
                chunks.append(chunk.astype(np.float32, copy=False))

            batch_tensor = torch.from_numpy(np.stack(chunks)).to(device, non_blocking=True)
            with torch.autocast(
                device_type=device_type,
                dtype=torch.float16,
                enabled=device_type == "cuda",
            ):
                logits = head(batch_tensor)
            batch_probabilities = torch.sigmoid(logits).squeeze(-1).float().cpu().numpy()

            for start_sample, valid_samples, probabilities in zip(
                batch_starts, valid_sample_counts, batch_probabilities
            ):
                start_frame = int(start_sample / float(sr) * frame_rate_hz)
                valid_audio_frames = int(np.ceil(valid_samples / float(sr) * frame_rate_hz))
                valid_frames = min(
                    len(probabilities),
                    chunk_frames,
                    valid_audio_frames,
                    total_frames - start_frame,
                )
                if valid_frames <= 0:
                    continue
                end_frame = start_frame + valid_frames
                weights = window[:valid_frames]
                probability_sum[start_frame:end_frame] += probabilities[:valid_frames] * weights
                weight_sum[start_frame:end_frame] += weights

    continuous_probabilities = np.divide(
        probability_sum,
        weight_sum,
        out=np.zeros_like(probability_sum),
        where=weight_sum > 0,
    )
    proposals = decode_events(
        continuous_probabilities,
        frame_rate_hz=frame_rate_hz,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
        min_duration_s=min_duration_s,
        max_merge_gap_s=max_merge_gap_s,
    )

    # Optional Stage 2 Verifier evaluation
    if verifier_checkpoint and Path(verifier_checkpoint).is_file():
        from wingbeat_ml.sed.models.verifier import Stage2Verifier
        verifier = Stage2Verifier(
            atst_checkpoint=atst_checkpoint or "checkpoints/atstframe_base.ckpt",
            input_sample_rate=sr,
        ).to(device)
        verifier_state = torch.load(verifier_checkpoint, map_location=device, weights_only=False)
        if isinstance(verifier_state, dict) and "model_state_dict" in verifier_state:
            verifier.load_state_dict(verifier_state["model_state_dict"])
        elif isinstance(verifier_state, dict):
            verifier.load_state_dict(verifier_state)

        for prop in proposals:
            s_sample = int(prop.start_s * sr)
            e_sample = int(prop.end_s * sr)
            candidate_audio = torch.from_numpy(signal[s_sample:e_sample]).float()
            v_score = verifier.verify_candidate(candidate_audio, sample_rate=sr)
            prop.mean_score = prop.confidence  # Preserve stage 1 score
            prop.confidence = v_score           # Verifier decision score

    return proposals

    return proposals


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        events = infer_long_recording(sys.argv[1])
        print(f"Decoded {len(events)} mosquito events from {sys.argv[1]}")
