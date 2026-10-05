"""Timestamp-based temporal windows and clip decoding for R3D-18."""
from dataclasses import dataclass
import math
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class WindowLabel:
    start_seconds: float
    end_seconds: float
    label: str
    target: int | None
    temporal_known: bool


def window_starts(duration, window_seconds, stride_seconds):
    if duration <= 0 or window_seconds <= 0 or stride_seconds <= 0:
        raise ValueError("duration, window and stride must be positive")
    if duration <= window_seconds:
        return [0.0]
    count = int(math.floor((duration - window_seconds) / stride_seconds)) + 1
    return [i * stride_seconds for i in range(count)]


def label_window(start, end, *, clip_label, event_start=None, event_end=None,
                 boundary_guard=0.5, positive_overlap_fraction=0.5):
    """Crash only when overlap is sufficient; uncertain event boundaries are ignored."""
    if event_start is None or event_end is None:
        if clip_label in {"crash", "normal"}:
            return WindowLabel(start, end, clip_label, int(clip_label == "crash"), False)
        return WindowLabel(start, end, "unknown", None, False)
    if not start < end or not event_start < event_end:
        raise ValueError("Window and event intervals must be increasing")
    overlap = max(0.0, min(end, event_end) - max(start, event_start))
    if overlap / (end - start) >= positive_overlap_fraction:
        return WindowLabel(start, end, "crash", 1, True)
    if end <= event_start - boundary_guard:
        return WindowLabel(start, end, "normal", 0, True)
    if start >= event_end + boundary_guard:
        return WindowLabel(start, end, "aftermath", None, True)
    return WindowLabel(start, end, "uncertain_boundary", None, True)


def sample_timestamps(window_start, frame_count=16, sampling_fps=8.0):
    if frame_count < 1 or sampling_fps <= 0:
        raise ValueError("frame_count and sampling_fps must be positive")
    return [float(window_start) + i / sampling_fps for i in range(frame_count)]


def letterbox_rgb(frame_bgr, width=112, height=112, fill=0):
    """Resize full scene to fit; pad without cropping roadway context."""
    h, w = frame_bgr.shape[:2]
    scale = min(width / w, height / h)
    resized_w, resized_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(frame_bgr, (resized_w, resized_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.full((height, width, 3), fill, dtype=np.uint8)
    x, y = (width - resized_w) // 2, (height - resized_h) // 2
    canvas[y:y + resized_h, x:x + resized_w] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def decode_window(path, start_seconds, *, frame_count=16, sampling_fps=8.0,
                  width=112, height=112, max_missing_fraction=0.2,
                  end_seconds=None, uniform_over_span=False):
    """Seek at explicit timestamps, preserve actual timestamps, pad only short tails."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    frames, actual, missing_indices, valid_indices = [], [], [], []
    if uniform_over_span:
        if end_seconds is None or end_seconds <= start_seconds:
            cap.release()
            raise ValueError("uniform sampling requires an increasing end_seconds")
        targets = np.linspace(start_seconds, end_seconds, frame_count, endpoint=False).tolist()
    else:
        targets = sample_timestamps(start_seconds, frame_count, sampling_fps)
    for index, target_time in enumerate(targets):
        cap.set(cv2.CAP_PROP_POS_MSEC, target_time * 1000)
        ok, frame = cap.read()
        if not ok or frame is None:
            missing_indices.append(index)
            continue
        actual_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if actual_time <= 0 and source_fps > 0:
            actual_time = max(0.0, cap.get(cv2.CAP_PROP_POS_FRAMES) / source_fps - 1 / source_fps)
        frames.append(letterbox_rgb(frame, width, height))
        actual.append(actual_time)
        valid_indices.append(index)
    cap.release()
    missing = len(missing_indices)
    if missing / frame_count > max_missing_fraction or not frames:
        return None, {"skipped": True, "missing_frames": missing, "requested_frames": frame_count,
                      "reason": "too_many_missing_frames"}
    if missing and min(missing_indices) < max(valid_indices):
        return None, {"skipped": True, "missing_frames": missing, "requested_frames": frame_count,
                      "reason": "missing_interior_frame_breaks_temporal_cadence"}
    real_frame_count = len(frames)
    padded = real_frame_count < frame_count
    while len(frames) < frame_count:
        frames.append(frames[-1].copy())
        actual.append(actual[-1])
    # T,C,H,W RGB uint8; padding count is explicit for reproducibility.
    while len(actual) < frame_count:
        actual.append(actual[-1])
    return np.stack(frames, axis=0), {"skipped": False, "timestamps_seconds": actual,
                                     "missing_frames": missing, "padded_tail_frames": frame_count - real_frame_count if padded else 0,
                                     "source_fps": source_fps}

