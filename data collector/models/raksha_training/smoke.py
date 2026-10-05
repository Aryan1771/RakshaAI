"""Synthetic-only software smoke test. Outputs are quarantined and never training data."""
from datetime import datetime, timezone
import json
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .audit import _ffprobe
from .metrics import classification_metrics
from .models import build_r3d18
from .temporal import decode_window
from .video_data import VideoWindowDataset, build_window_manifest
from .yolo import train_yolo
from .stream import run_inference


def _synthetic_video(path, seed, crash_pattern):
    rng = np.random.default_rng(seed)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8, (96, 64))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create the synthetic smoke video")
    for index in range(16):
        frame = np.full((64, 96, 3), (50, 90, 55), dtype=np.uint8)
        cv2.rectangle(frame, (0, 38), (95, 63), (70, 70, 70), -1)
        cv2.line(frame, (0, 50), (95, 50), (220, 220, 220), 1)
        x = 5 + index * 4
        cv2.rectangle(frame, (x, 28), (x + 14, 37), (20, 20, 230), -1)
        if crash_pattern and 7 <= index <= 10:
            cv2.circle(frame, (48, 34), 10, (0, 0, 255), 2)
        noise = rng.integers(-2, 3, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        writer.write(frame)
    writer.release()


def run_smoke(output_dir, seed=42, run_yolo=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    out = Path(output_dir).resolve()
    root = out / "synthetic_only"
    videos = root / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    (root / "SMOKE_ONLY.txt").write_text(
        "Synthetic computer-generated fixtures for software validation only. Not real road footage, not accident evidence, and never a training dataset.\n",
        encoding="utf-8")
    rows = []
    for split in ("train", "validation"):
        for label, target in (("normal", 0), ("crash", 1)):
            for i in range(2):
                path = videos / f"{split}_{label}_{i}.mp4"
                _synthetic_video(path, seed + len(rows), target == 1)
                probe = _ffprobe(path)
                if not probe.get("ok") and probe.get("error") == "ffprobe_not_found":
                    cap = cv2.VideoCapture(str(path))
                    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
                    count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    if cap.isOpened() and fps > 0 and count > 0:
                        probe = {"ok": True, "duration_seconds": count / fps,
                                 "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                 "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), "fps": fps,
                                 "inspection_backend": "OpenCV fallback; ffprobe unavailable"}
                    cap.release()
                if not probe.get("ok"):
                    raise RuntimeError(f"Smoke video is unreadable: {path} {probe}")
                rows.append({"clip_path": str(path), "duration_seconds": probe["duration_seconds"],
                             "clip_label": label, "split": split,
                             "record_id": f"synthetic-{split}-{label}-{i}",
                             "incident_id": f"synthetic-{split}-{label}-{i}",
                             "source_video_id": f"synthetic-{split}-{label}-{i}",
                             "country": "India", "permission_status": "permitted",
                             "review_status": "confirmed", "annotation_origin": "human",
                             "annotation_review_status": "confirmed"})
    (root / "synthetic_manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    decoder, decode_meta = decode_window(rows[0]["clip_path"], 0, frame_count=16, sampling_fps=8,
                                          width=64, height=64)
    if decoder is None or decoder.shape != (16, 64, 64, 3):
        raise RuntimeError(f"Synthetic timestamp decoder smoke failed: {decode_meta}")
    smoke_video_config = {"clip_frames": 16, "sampling_fps": 8, "temporal_span_seconds": 2,
                          "window_stride_seconds": 1, "width": 64, "height": 64,
                          "cache_root": str(root / "cache"), "horizontal_flip_probability": 0}
    datasets = {}
    for split in ("train", "validation"):
        split_rows = [r for r in rows if r["split"] == split]
        windows, rejected = build_window_manifest(split_rows, require_eligible=False)
        if rejected or len(windows) != 4:
            raise RuntimeError(f"Synthetic manifest smoke failed: {rejected} {len(windows)}")
        datasets[split] = VideoWindowDataset(windows, split=split, config=smoke_video_config,
            mean=(.43216, .394666, .37645), std=(.22803, .22145, .216989))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_info = build_r3d18(2, None)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    loader = DataLoader(datasets["train"], batch_size=1, shuffle=False, num_workers=0)
    train_loss = None
    for batch in loader:
        video, label = batch["video"].to(device), batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(video), label)
        loss.backward()
        optimizer.step()
        train_loss = float(loss.item())
        break
    model.eval()
    truth, probs = [], []
    with torch.inference_mode():
        for batch in DataLoader(datasets["validation"], batch_size=1, shuffle=False, num_workers=0):
            video = batch["video"].to(device)
            logits = model(video)
            truth.append(int(batch["label"].item()))
            probs.append(float(torch.softmax(logits, 1)[0, 1].item()))
    metrics = classification_metrics(truth, probs, .5)
    checkpoint_dir = root / "r3d"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / "synthetic_smoke.pt"
    torch.save({"model": model.state_dict(), "model_info": model_info, "synthetic_only": True}, checkpoint)

    yolo_result = None
    stream_result = None
    if run_yolo:
        yolo_root = root / "yolo_dataset"
        names = ["vehicle"]
        for split in ("train", "val"):
            for i in range(2):
                image = np.full((96, 96, 3), 80 + i * 20, dtype=np.uint8)
                cv2.rectangle(image, (20 + i, 35), (64 + i, 66), (0, 0, 255), -1)
                image_path = yolo_root / f"images/{split}/{split}_{i}.jpg"
                label_path = yolo_root / f"labels/{split}/{split}_{i}.txt"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(image_path), image)
                label_path.write_text("0 0.45 0.526 0.46 0.323\n", encoding="utf-8")
        yaml_path = yolo_root / "smoke.yaml"
        yaml_path.write_text(f"path: {yolo_root.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: vehicle\n", encoding="utf-8")
        yolo_config = {"seed": seed, "yolo": {"model": "yolov8n.yaml", "pretrained_source": "random initialization; synthetic smoke only",
            "imgsz": 64, "batch": 1, "epochs": 1, "patience": 1, "optimizer": "SGD", "lr0": .001,
            "workers": 0, "amp": False, "seed": seed, "deterministic": True,
            "device": 0 if torch.cuda.is_available() else "cpu", "selection_metric": "metrics/mAP50-95(B)"}}
        yolo_result = train_yolo(yolo_config, yaml_path, output_dir=root / "yolo_run")
        from ultralytics import YOLO
        yolo_detector = YOLO(yolo_result["best_checkpoint"])
        model.eval()
        stream_config = {"seed": seed,
            "video": {**smoke_video_config, "temporal_span_seconds": 2.0, "window_stride_seconds": 1.0,
                      "sampling_fps": 8.0, "clip_frames": 16},
            "yolo": {"imgsz": 64, "inference_confidence": .15},
            "r3d": {"amp": False},
            "stream": {"warmup_iterations": 0,
                       "persistence": {"positive_threshold": .5, "reset_threshold": .2,
                                       "persist_count": 1, "history": 1, "reset_count": 2}}}
        stream_result = run_inference(rows[0]["clip_path"], yolo_model=yolo_detector,
            r3d_model=model, r3d_state={"model_info": model_info}, config=stream_config,
            output_dir=root / "stream_inference", max_frames=16)
        if stream_result["classifier_windows"] < 1:
            raise RuntimeError("Synthetic combined-stream smoke did not execute an R3D classifier window")
    report = {"status": "passed", "synthetic_only": True, "uses_real_road_data": False,
              "does_not_validate_accident_detection_quality": True,
              "created_at": datetime.now(timezone.utc).isoformat(), "device": str(device),
              "torch_version": torch.__version__, "r3d_train_step_loss": train_loss,
              "r3d_smoke_validation_metrics_not_real_world_scores": metrics,
              "video_decode": decode_meta, "r3d_checkpoint": str(checkpoint),
              "yolo_training": yolo_result, "combined_stream_inference": stream_result}
    (root / "smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
