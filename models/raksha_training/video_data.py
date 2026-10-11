"""Crash-safe, timestamped R3D window manifest and Dataset."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .temporal import decode_window, label_window, window_starts


def load_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL row {line_no}: {exc}") from exc
    return rows


def eligible_review_row(row):
    return bool(row.get("country", "").casefold() == "india"
                and row.get("permission_status") in {"permitted", "research_basis"}
                and row.get("review_status") == "confirmed"
                and row.get("decision", "include") == "include"
                and row.get("annotation_origin", row.get("origin")) == "human"
                and row.get("incident_id") and row.get("source_video_id")
                and row.get("clip_path") and Path(row["clip_path"]).is_file())


def build_window_manifest(rows, *, window_seconds=2.0, stride_seconds=1.0,
                          boundary_guard=0.5, positive_overlap_fraction=0.5,
                          require_eligible=True):
    """Create window labels from reviewed temporal segments, not headlines."""
    output, rejected = [], Counter()
    for row in rows:
        if require_eligible and not eligible_review_row(row):
            rejected["review_or_provenance_gate"] += 1
            continue
        path = row.get("clip_path")
        if not path or not Path(path).is_file():
            rejected["missing_video"] += 1
            continue
        duration = float(row.get("duration_seconds") or row.get("video", {}).get("duration", 0))
        if duration <= 0:
            rejected["unknown_duration"] += 1
            continue
        segments = row.get("segments") or []
        has_event_bounds = row.get("event_start_seconds") is not None and row.get("event_end_seconds") is not None
        if not has_event_bounds and not segments:
            label = row.get("clip_label")
            if label not in {"crash", "normal"}:
                rejected["no_clip_or_temporal_label"] += 1
                continue
            # Clip classification remains supported, but its onset and delay are unknown.
            output.append({**row, "clip_path": str(Path(path).resolve()), "window_start_seconds": 0.0,
                           "window_end_seconds": duration, "target": int(label == "crash"),
                           "label": label, "temporal_known": False, "sampling_mode": "uniform_over_full_clip"})
            continue
        starts = window_starts(duration, window_seconds, stride_seconds)
        for start in starts:
            end = min(duration, start + window_seconds)
            segment = next((s for s in segments if s.get("target") in (0, 1)
                            and start >= float(s["start"]) + (boundary_guard if s["label"] == "normal" else 0)
                            and end <= float(s["end"]) - (boundary_guard if s["label"] == "normal" else 0)), None)
            if segment:
                label = segment["label"]
                target = int(segment["target"])
                temporal_known = True
            else:
                event_start = row.get("event_start_seconds")
                event_end = row.get("event_end_seconds")
                if event_start is not None and event_end is not None:
                    labeled = label_window(start, end, clip_label=row.get("clip_label"),
                                           event_start=float(event_start), event_end=float(event_end),
                                           boundary_guard=boundary_guard,
                                           positive_overlap_fraction=positive_overlap_fraction)
                    label, target, temporal_known = labeled.label, labeled.target, labeled.temporal_known
                else:
                    label, target, temporal_known = "unknown", None, False
            if target not in (0, 1):
                rejected[label] += 1
                continue
            output.append({**row, "clip_path": str(Path(path).resolve()),
                           "window_start_seconds": start, "window_end_seconds": end,
                           "target": target, "label": label, "temporal_known": temporal_known,
                           "sampling_mode": "timestamp_window"})
    return output, dict(rejected)


class VideoWindowDataset(Dataset):
    def __init__(self, manifest_rows, *, split, config, mean, std, augment=False):
        if split not in {"train", "validation", "test", "official_test"}:
            raise ValueError("split must be explicit")
        self.rows, self.split, self.config = list(manifest_rows), split, config
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1, 1)
        self.augment = bool(augment and split == "train")
        self.cache_root = Path(config.get("cache_root", "models/runs/cache")) / split
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        start = float(row["window_start_seconds"])
        span = float(row["window_end_seconds"] - start)
        cfg = self.config
        key_raw = f"{row['clip_path']}|{Path(row['clip_path']).stat().st_size}|{start}|{span}|{cfg}|{self.split}"
        key = hashlib.sha256(key_raw.encode()).hexdigest()
        cached = self.cache_root / f"{key}.npz"
        if cached.is_file():
            with np.load(cached, allow_pickle=False) as data:
                frames = data["frames"]
                timestamps = data["timestamps_seconds"]
                padded_tail_frames = int(data["padded_tail_frames"])
                source_fps = float(data["source_fps"])
        else:
            if row.get("sampling_mode") == "uniform_over_full_clip":
                frames, meta = decode_window(row["clip_path"], 0,
                    frame_count=int(cfg["clip_frames"]), sampling_fps=float(cfg["sampling_fps"]),
                    width=int(cfg["width"]), height=int(cfg["height"]),
                    max_missing_fraction=float(cfg.get("max_missing_fraction", .2)),
                    end_seconds=float(row["window_end_seconds"]), uniform_over_span=True)
            else:
                frames, meta = decode_window(row["clip_path"], start,
                    frame_count=int(cfg["clip_frames"]), sampling_fps=float(cfg["sampling_fps"]),
                    width=int(cfg["width"]), height=int(cfg["height"]),
                    max_missing_fraction=float(cfg.get("max_missing_fraction", .2)))
            if frames is None or meta["skipped"]:
                raise RuntimeError(f"Window decode failed ({meta['reason']}): {row['clip_path']} @ {start}s")
            timestamps = np.asarray(meta["timestamps_seconds"], dtype=np.float64)
            padded_tail_frames = int(meta["padded_tail_frames"])
            source_fps = float(meta["source_fps"] or 0)
            np.savez_compressed(cached, frames=frames, timestamps_seconds=timestamps,
                                padded_tail_frames=padded_tail_frames, source_fps=source_fps)
        # One spatial transform for all frames. Horizontal flip stays disabled by default.
        if self.augment and np.random.random() < float(cfg.get("horizontal_flip_probability", 0)):
            frames = frames[:, :, ::-1].copy()
        # RGB uint8 T,H,W,C -> float C,T,H,W; ImageNet-like Kinetics stats from weight enum.
        tensor = torch.from_numpy(frames.copy()).permute(3, 0, 1, 2).float() / 255.0
        tensor = (tensor - self.mean) / self.std
        return {"video": tensor, "label": torch.tensor(int(row["target"]), dtype=torch.long),
                "record_id": str(row.get("record_id", row.get("source_video_id", ""))),
                "incident_id": str(row.get("incident_id", "")),
                "window_start_seconds": start, "window_end_seconds": float(row["window_end_seconds"]),
                "temporal_known": bool(row.get("temporal_known", False)),
                "sample_timestamps_seconds": torch.from_numpy(np.asarray(timestamps, dtype=np.float64)),
                "padded_tail_frames": padded_tail_frames, "source_fps": source_fps,
                "camera_id": str(row.get("camera_id") or "")}

