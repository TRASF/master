"""Deterministic Frame Event Decoder.

Converts frame probability series p(t) into timestamped event intervals [start_s, end_s]
using dual-threshold hysteresis, gap merging, and minimum duration filtering.
Completely decoupled from model architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class DecodedEvent:
    start_s: float
    end_s: float
    confidence: float
    mean_score: float | None = None
    p90_score: float | None = None
    top25_mean_score: float | None = None


def decode_events(
    probabilities: np.ndarray,
    frame_rate_hz: float = 50.0,
    high_threshold: float = 0.8,
    low_threshold: float = 0.4,
    min_duration_s: float = 0.1,
    max_merge_gap_s: float = 0.2,
    offset_s: float = 0.0,
) -> list[DecodedEvent]:
    """Decode per-frame probabilities into discrete temporal event intervals.

    Args:
        probabilities: 1D array of frame probabilities p_t in range [0, 1].
        frame_rate_hz: Temporal resolution in frames per second.
        high_threshold: Threshold to start a new event detection (p >= high).
        low_threshold: Threshold to sustain an ongoing event detection (p >= low).
        min_duration_s: Minimum duration required to accept an event.
        max_merge_gap_s: Maximum gap duration between nearby events to merge them.
        offset_s: Time offset in seconds to add to timestamps.

    Returns:
        List of DecodedEvent dataclasses.
    """
    probs = np.asarray(probabilities, dtype=np.float32).ravel()
    n_frames = len(probs)
    if n_frames == 0:
        return []

    frame_dur = 1.0 / frame_rate_hz
    raw_events: list[tuple[int, int, float]] = []

    in_event = False
    start_f = 0
    max_prob = 0.0

    # Hysteresis thresholding
    for i in range(n_frames):
        p = float(probs[i])
        if not in_event:
            if p >= high_threshold:
                in_event = True
                start_f = i
                max_prob = p
        else:
            if p >= low_threshold:
                max_prob = max(max_prob, p)
            else:
                in_event = False
                raw_events.append((start_f, i, max_prob))

    if in_event:
        raw_events.append((start_f, n_frames, max_prob))

    if not raw_events:
        return []

    # Merge nearby events separated by gaps <= max_merge_gap_s
    max_gap_frames = int(max_merge_gap_s * frame_rate_hz)
    merged_events: list[tuple[int, int, float]] = []

    curr_start, curr_end, curr_max = raw_events[0]
    for next_start, next_end, next_max in raw_events[1:]:
        if next_start - curr_end <= max_gap_frames:
            curr_end = next_end
            curr_max = max(curr_max, next_max)
        else:
            merged_events.append((curr_start, curr_end, curr_max))
            curr_start, curr_end, curr_max = next_start, next_end, next_max
    merged_events.append((curr_start, curr_end, curr_max))

    # Filter out events shorter than min_duration_s and format output
    min_frames = int(np.ceil(min_duration_s * frame_rate_hz))
    final_events: list[DecodedEvent] = []

    for s_f, e_f, conf in merged_events:
        if (e_f - s_f) >= min_frames:
            start_sec = offset_s + (s_f * frame_dur)
            end_sec = offset_s + (e_f * frame_dur)
            event_probs = probs[s_f:e_f]
            top_count = max(1, int(np.ceil(len(event_probs) * 0.25)))
            final_events.append(
                DecodedEvent(
                    start_s=start_sec,
                    end_s=end_sec,
                    confidence=conf,
                    mean_score=float(np.mean(event_probs)),
                    p90_score=float(np.quantile(event_probs, 0.9)),
                    top25_mean_score=float(np.mean(np.partition(event_probs, -top_count)[-top_count:])),
                )
            )

    return final_events
