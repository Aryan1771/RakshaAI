"""OpenCV source abstraction, bounded timestamp ring buffer and two-path inference."""
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
import uuid
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np
import torch

from .temporal import letterbox_rgb


@dataclass
class VideoFrame:
    image_bgr: np.ndarray
    timestamp_seconds: float
    sequence: int


class VideoSource(ABC):
    @abstractmethod
    def read(self):
        """Return VideoFrame or None when ended/temporarily unavailable."""

    @abstractmethod
    def close(self):
        pass


class OpenCVSource(VideoSource):
    """File, RTSP URL or webcam source with bounded stream reconnect attempts."""
    def __init__(self, source, reconnect_attempts=5, reconnect_wait_seconds=1.0):
        text = str(source)
        is_webcam = isinstance(source, int) or text.isdecimal()
        self.source = int(source) if is_webcam else source
        self.is_file = not is_webcam and not text.lower().startswith("rtsp://")
        self.reconnect_attempts = int(reconnect_attempts)
        self.reconnect_wait = float(reconnect_wait_seconds)
        self.reconnects = 0
        self.lost_frames = 0
        self.sequence = 0
        self.started = time.monotonic()
        self._open()

    def _open(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open VideoSource: {self.source}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0)
        self.failures = 0

    def read(self):
        while True:
            ok, frame = self.cap.read()
            if ok and frame is not None and frame.size:
                if self.is_file:
                    milliseconds = float(self.cap.get(cv2.CAP_PROP_POS_MSEC))
                    timestamp = milliseconds / 1000 if milliseconds > 0 else (self.sequence / self.fps if self.fps > 0 else time.monotonic() - self.started)
                else:
                    timestamp = time.monotonic() - self.started
                result = VideoFrame(frame, timestamp, self.sequence)
                self.sequence += 1
                self.failures = 0
                return result
            if self.is_file:
                return None
            self.lost_frames += 1
            self.failures += 1
            if self.failures >= self.reconnect_attempts:
                return None
            self.cap.release()
            time.sleep(self.reconnect_wait)
            self.reconnects += 1
            try:
                self._open()
            except ValueError:
                continue

    def close(self):
        if getattr(self, "cap", None):
            self.cap.release()


class TimestampRingBuffer:
    def __init__(self, max_seconds, max_frames):
        self.max_seconds = float(max_seconds)
        self.frames = deque(maxlen=int(max_frames))

    def append(self, frame):
        if self.frames and frame.timestamp_seconds <= self.frames[-1].timestamp_seconds:
            return False
        self.frames.append(frame)
        cutoff = frame.timestamp_seconds - self.max_seconds
        while self.frames and self.frames[0].timestamp_seconds < cutoff:
            self.frames.popleft()
        return True

    def sample(self, end_seconds, count, fps, tolerance=None):
        if not self.frames:
            return None
        tolerance = tolerance if tolerance is not None else .5 / fps
        targets = [end_seconds - (count - 1 - i) / fps for i in range(count)]
        available = list(self.frames)
        chosen = []
        for target in targets:
            nearest = min(available, key=lambda f: abs(f.timestamp_seconds - target))
            if abs(nearest.timestamp_seconds - target) > tolerance:
                return None
            chosen.append(nearest)
        if any(a.sequence == b.sequence for a, b in zip(chosen, chosen[1:])):
            return None
        return chosen


class PersistenceGate:
    """Persistence with signal-based rearming; no global time cooldown."""
    def __init__(self, positive_threshold=.5, reset_threshold=.2, persist_count=2, history=3, reset_count=2):
        self.threshold, self.reset_threshold = positive_threshold, reset_threshold
        self.persist_count, self.reset_count = int(persist_count), int(reset_count)
        self.scores, self.low_count, self.active = deque(maxlen=int(history)), 0, False

    def update(self, score):
        self.scores.append(float(score))
        if score < self.reset_threshold:
            self.low_count += 1
        else:
            self.low_count = 0
        if self.active and self.low_count >= self.reset_count:
            self.active = False
            self.scores.clear()
            self.low_count = 0
            return False
        fires = sum(value >= self.threshold for value in self.scores) >= self.persist_count
        if fires and not self.active:
            self.active = True
            return True
        return False


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _safe_source(value):
    text = str(value)
    if "://" not in text:
        return text
    parts = urlsplit(text)
    host = parts.hostname or ""
    if parts.port:
        host += f":{parts.port}"
    return urlunsplit((parts.scheme, host, "", "", ""))


def run_inference(source, *, yolo_model=None, r3d_model=None, r3d_state=None,
                  config=None, camera_id=None, output_dir="models/runs/phase2/predictions",
                  show=False, max_frames=None):
    cfg = config or {}
    video_cfg = cfg.get("video", {})
    fps = float(video_cfg.get("sampling_fps", 8.0))
    frames_per_clip = int(video_cfg.get("clip_frames", 16))
    temporal_span = float(video_cfg.get("temporal_span_seconds", frames_per_clip / fps))
    width, height = int(video_cfg.get("width", 112)), int(video_cfg.get("height", 112))
    device = next(r3d_model.parameters()).device if r3d_model is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm = (r3d_state or {}).get("model_info", {})
    mean = torch.tensor(norm.get("normalization_mean", [.43216, .394666, .37645]), device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor(norm.get("normalization_std", [.22803, .22145, .216989]), device=device).view(1, 3, 1, 1, 1)
    source_obj = OpenCVSource(source, reconnect_attempts=int(cfg.get("stream", {}).get("reconnect_attempts", 5)))
    ring = TimestampRingBuffer(max_seconds=max(temporal_span * 2, 4), max_frames=max(128, frames_per_clip * 4))
    persistence = PersistenceGate(**cfg.get("stream", {}).get("persistence", {}))
    stride = float(video_cfg.get("window_stride_seconds", 1.0))
    next_sample_at, last_window_at = -1.0, -float("inf")
    incidents = []
    timings = {"detector_seconds": [], "classifier_seconds": [], "pipeline_frame_seconds": []}
    start_clock, first_ts, last_ts, frames_seen = time.perf_counter(), None, None, 0
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prediction_path, incident_path = output / "predictions.jsonl", output / "incidents.jsonl"
    warmup_count = int(cfg.get("stream", {}).get("warmup_iterations", 3))
    warmup_started = time.perf_counter()
    if warmup_count > 0 and (yolo_model is not None or r3d_model is not None):
        if yolo_model is not None:
            dummy = np.zeros((int(cfg.get("yolo", {}).get("imgsz", 640)),) * 2 + (3,), dtype=np.uint8)
            for _ in range(warmup_count):
                yolo_model.predict(dummy, verbose=False, conf=float(cfg.get("yolo", {}).get("inference_confidence", .15)),
                                   device=0 if device.type == "cuda" else "cpu")
                _sync(device)
        if r3d_model is not None:
            dummy = torch.zeros((1, 3, frames_per_clip, height, width), device=device)
            for _ in range(warmup_count):
                with torch.inference_mode(), torch.amp.autocast(device_type=device.type,
                        enabled=device.type == "cuda" and bool(cfg.get("r3d", {}).get("amp", True))):
                    r3d_model(dummy)
                _sync(device)
    warmup_seconds = time.perf_counter() - warmup_started
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        with prediction_path.open("w", encoding="utf-8") as prediction_file:
            while True:
                frame_started = time.perf_counter()
                item = source_obj.read()
                if item is None:
                    break
                frames_seen += 1
                ts = item.timestamp_seconds
                first_ts = ts if first_ts is None else first_ts
                last_ts = ts
                output_frame = item.image_bgr
                detection_count = None
                # Path A: detection and overlays are independent of the scene-classification path.
                if yolo_model is not None:
                    det_start = time.perf_counter()
                    det = yolo_model.predict(item.image_bgr, verbose=False,
                                             conf=float(cfg.get("yolo", {}).get("inference_confidence", .15)),
                                             device=0 if device.type == "cuda" else "cpu")[0]
                    _sync(device)
                    timings["detector_seconds"].append(time.perf_counter() - det_start)
                    detection_count = len(det.boxes) if det.boxes is not None else 0
                    if show:
                        output_frame = det.plot()
                # Path B: the full-scene classifier proceeds even if YOLO returns zero boxes.
                score = None
                if ts >= next_sample_at:
                    ring.append(item)
                    next_sample_at = ts + 1.0 / fps if next_sample_at < 0 else next_sample_at
                    while next_sample_at <= ts:
                        next_sample_at += 1.0 / fps
                frame_pred = {"camera_id": camera_id, "frame_timestamp_seconds": ts,
                              "source_sequence": item.sequence, "detected_road_users": detection_count,
                              "detections": [], "crash_probability": None, "event_candidate": False}
                if yolo_model is not None and det.boxes is not None:
                    xyxy = det.boxes.xyxy.detach().cpu().tolist()
                    cls = det.boxes.cls.detach().cpu().tolist()
                    conf = det.boxes.conf.detach().cpu().tolist()
                    frame_pred["detections"] = [{"xyxy": box, "class_id": int(c), "confidence": float(p)}
                                                  for box, c, p in zip(xyxy, cls, conf)]
                if r3d_model is not None and ts - last_window_at >= stride:
                    chosen = ring.sample(ts, frames_per_clip, fps)
                    if chosen:
                        frames = np.stack([letterbox_rgb(x.image_bgr, width, height) for x in chosen], axis=0)
                        tensor = torch.from_numpy(frames).to(device).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
                        tensor = (tensor - mean) / std
                        classifier_start = time.perf_counter()
                        with torch.inference_mode(), torch.amp.autocast(device_type=device.type,
                                enabled=device.type == "cuda" and bool(cfg.get("r3d", {}).get("amp", True))):
                            logits = r3d_model(tensor)
                            score = float(torch.softmax(logits.float(), dim=1)[0, 1].item())
                        _sync(device)
                        classifier_elapsed = time.perf_counter() - classifier_start
                        timings["classifier_seconds"].append(classifier_elapsed)
                        last_window_at = ts
                        frame_pred.update({"decision_timestamp_seconds": ts + classifier_elapsed,
                                "window_start_seconds": chosen[0].timestamp_seconds,
                                "window_end_seconds": chosen[-1].timestamp_seconds,
                                "crash_probability": score,
                                "classifier_ran_without_detections": detection_count == 0 if detection_count is not None else None,
                                "processing_latency_seconds": classifier_elapsed})
                        if persistence.update(score):
                            event = {"event_id": str(uuid.uuid4()), "event_type": "road_accident_candidate",
                                     "camera_id": camera_id, "decision_timestamp_seconds": ts + classifier_elapsed,
                                     "first_available_window_start_seconds": chosen[0].timestamp_seconds,
                                     "crash_probability": score, "dispatch": False,
                                     "model": str((r3d_state or {}).get("model_info", {}).get("architecture", "R3D-18"))}
                            incidents.append(event)
                            frame_pred["event_candidate"] = True
                prediction_file.write(json.dumps(frame_pred) + "\n")
                prediction_file.flush()
                timings["pipeline_frame_seconds"].append(time.perf_counter() - frame_started)
                if show:
                    cv2.imshow("RakshaAI local inference", output_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if max_frames and frames_seen >= max_frames:
                    break
    finally:
        source_obj.close()
        if show:
            cv2.destroyAllWindows()
    with incident_path.open("w", encoding="utf-8") as stream:
        for row in incidents:
            stream.write(json.dumps(row) + "\n")
    elapsed = time.perf_counter() - start_clock
    status = "source_ended" if source_obj.is_file else "stream_unavailable_or_terminated"
    if max_frames and frames_seen >= max_frames:
        status = "max_frames_reached"
    result = {"source": _safe_source(source), "camera_id": camera_id, "status": status,
              "frames_seen": frames_seen, "lost_frames": source_obj.lost_frames,
              "reconnects": source_obj.reconnects, "duration_seconds": max(0, (last_ts or 0) - (first_ts or 0)),
              "wall_seconds": elapsed, "decode_fps_wall": frames_seen / elapsed if elapsed else None,
              "detector_ran": yolo_model is not None, "classifier_ran": r3d_model is not None,
              "persistence_config": cfg.get("stream", {}).get("persistence", {}),
              "classifier_windows": len(timings["classifier_seconds"]), "local_incident_candidates": len(incidents),
              "timing_ms_mean": {key: (1000 * float(np.mean(values)) if values else None) for key, values in timings.items()},
              "warmup_iterations_per_model": warmup_count, "warmup_seconds_excluded_from_runtime": warmup_seconds,
              "inference_resolution": {"source_frame": [item.image_bgr.shape[1], item.image_bgr.shape[0]] if frames_seen else None,
                                       "r3d_input": [width, height],
                                       "yolo_imgsz": int(cfg.get("yolo", {}).get("imgsz", 640)) if yolo_model is not None else None},
              "cuda_peak_allocated_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
              "visualization_enabled": bool(show), "dispatch": False,
              "predictions_path": str(prediction_path.resolve()), "incidents_path": str(incident_path.resolve())}
    (output / "stream_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

