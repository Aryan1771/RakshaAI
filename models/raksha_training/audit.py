"""Conservative source/video audit; unknown provenance and labels stay unknown."""
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import platform
import sys
from importlib import metadata as package_metadata

from .splits import grouped_split

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts"}


def _package_version(name):
    try:
        return package_metadata.version(name)
    except package_metadata.PackageNotFoundError:
        return None


def environment_report(repo_root):
    """Capture reproducibility-relevant hardware/runtime facts without requiring torch."""
    root = Path(repo_root).resolve()
    result = {"platform": platform.platform(), "python": sys.version,
              "logical_cpu_count": os.cpu_count(), "disk_free_bytes": shutil.disk_usage(root).free,
              "gpu": None, "torch": None, "torchvision": None,
              "package_versions": {name: _package_version(name)
                                   for name in ("numpy", "opencv-python", "ultralytics", "scikit-learn", "tensorboard")}}
    try:
        total = __import__("ctypes").c_ulonglong()
        class MemoryStatus(__import__("ctypes").Structure):
            _fields_ = [("length", __import__("ctypes").c_ulong), ("memory_load", __import__("ctypes").c_ulong),
                        ("total_phys", __import__("ctypes").c_ulonglong), ("avail_phys", __import__("ctypes").c_ulonglong),
                        ("total_page_file", __import__("ctypes").c_ulonglong), ("avail_page_file", __import__("ctypes").c_ulonglong),
                        ("total_virtual", __import__("ctypes").c_ulonglong), ("avail_virtual", __import__("ctypes").c_ulonglong),
                        ("avail_extended_virtual", __import__("ctypes").c_ulonglong)]
        state = MemoryStatus()
        state.length = __import__("ctypes").sizeof(state)
        if __import__("ctypes").windll.kernel32.GlobalMemoryStatusEx(__import__("ctypes").byref(state)):
            result["memory_total_bytes"], result["memory_available_bytes"] = state.total_phys, state.avail_phys
    except Exception:
        result["memory_total_bytes"] = result["memory_available_bytes"] = None
    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            output = subprocess.run([executable, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                                    capture_output=True, text=True, timeout=10, check=True).stdout.strip()
            result["gpu"] = output.splitlines()
            result["gpu_compute_capability"] = subprocess.run([executable, "--query-gpu=compute_cap", "--format=csv,noheader"],
                                    capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        except Exception as exc:
            result["gpu_error"] = type(exc).__name__
    try:
        import torch
        import torchvision
        result["torch"], result["torchvision"] = torch.__version__, torchvision.__version__
        result["torch_cuda_runtime"] = torch.version.cuda
        result["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            result["cuda_device"] = torch.cuda.get_device_name(0)
            result["cuda_vram_bytes"] = torch.cuda.get_device_properties(0).total_memory
    except Exception as exc:
        result["ml_stack_import_error"] = type(exc).__name__
    return result


def _ffprobe(path):
    executable = os.getenv("FFPROBE") or shutil.which("ffprobe")
    if not executable:
        try:
            import cv2
            cap = cv2.VideoCapture(str(path))
            opened = cap.isOpened()
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ok, _ = cap.read() if opened else (False, None)
            cap.release()
            if opened and ok and fps > 0 and count > 0 and width > 0 and height > 0:
                return {"ok": True, "duration_seconds": count / fps, "width": width, "height": height,
                        "fps": fps, "inspection_backend": "OpenCV fallback; ffprobe unavailable"}
            return {"ok": False, "error": "opencv_decode_failed", "inspection_backend": "OpenCV fallback; ffprobe unavailable"}
        except Exception as exc:
            return {"ok": False, "error": f"inspection_tools_unavailable:{type(exc).__name__}"}
    try:
        result = subprocess.run(
            [executable, "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height,avg_frame_rate",
             "-of", "json", str(path)], capture_output=True, text=True, timeout=30, check=True,
        )
        data = json.loads(result.stdout)
        video = next(s for s in data.get("streams", []) if s.get("codec_type") == "video")
        return {"ok": True, "duration_seconds": float(data["format"]["duration"]),
                "width": video.get("width"), "height": video.get("height"),
                "fps": video.get("avg_frame_rate"), "inspection_backend": "ffprobe"}
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__}


def _sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_phase1(root):
    db_path = root / "collection.sqlite3"
    if not db_path.is_file():
        return {}, {}, {"available": False, "records": 0, "annotations": 0, "duplicate_pairs": {}}
    from raksha_collector.storage import Store

    store = Store(db_path)
    records = {}
    for (payload,) in store.db.execute("SELECT payload FROM records"):
        row = json.loads(payload)
        records[row["id"]] = row
    annotations = {}
    for serial, payload in store.db.execute("SELECT serial,payload FROM annotations ORDER BY serial"):
        row = json.loads(payload)
        annotations[str(Path(row.get("local_path", "")).resolve())] = row
    pair_counts = Counter(f"{kind}:{status}" for _, _, kind, status in store.db.execute("SELECT a,b,kind,status FROM pairs"))
    store.db.close()
    return records, annotations, {"available": True, "records": len(records), "annotations": len(annotations),
                                  "duplicate_pairs": dict(pair_counts)}


def run_audit(repo_root, phase1_path, existing_clips_path, output_path, seed=42):
    repo_root = Path(repo_root).resolve()
    phase1 = (repo_root / phase1_path).resolve()
    existing = (repo_root / existing_clips_path).resolve()
    output = (repo_root / output_path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records, annotations, phase1_counts = _load_phase1(phase1)

    paths = []
    for path in (phase1 / "segments", phase1 / "annotated", existing):
        if path.is_dir():
            paths.extend(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
    paths = sorted(set(paths))
    record_by_path = {}
    for rid, record in records.items():
        local = record.get("download", {}).get("local_path")
        if local:
            record_by_path[str(Path(local).resolve())] = (rid, record)

    video_rows, hashes = [], defaultdict(list)
    for path in paths:
        resolved = str(path.resolve())
        ann = annotations.get(resolved)
        record_pair = record_by_path.get(resolved)
        rid, record = record_pair if record_pair else (None, {})
        probe = _ffprobe(path)
        digest = _sha256(path) if probe["ok"] else None
        if digest:
            hashes[digest].append(resolved)
        annotation_review = ann.get("annotation_review_status", "missing") if ann else "missing"
        reviewed_include = bool(record.get("human", {}).get("review_status") == "reviewed"
                                and record.get("human", {}).get("decision") == "include")
        permission = record.get("permission", {}).get("status", "unknown")
        country = record.get("human", {}).get("country", "unknown")
        eligible = bool(probe["ok"] and ann and ann.get("origin") == "human"
                        and annotation_review == "confirmed" and ann.get("training_eligible")
                        and reviewed_include and country.casefold() == "india"
                        and permission in {"permitted", "research_basis"}
                        and record.get("incident_id") and (record.get("source_video_id") or rid))
        row = {
            "path": resolved, "bytes": path.stat().st_size, **probe,
            "record_id": rid, "source_video_id": record.get("source_video_id"),
            "source_recording_id": rid, "incident_id": record.get("incident_id"),
            "country": country, "location_evidence_status": "present" if record.get("human", {}).get("location_evidence") else "missing",
            "permission_status": permission, "review_status": record.get("human", {}).get("review_status", "missing"),
            "footage_type": record.get("human", {}).get("footage_type", "unknown"),
            "annotation_review_status": annotation_review, "annotation_origin": ann.get("origin") if ann else None,
            "clip_label": ann.get("class_name") if ann else None,
            "event_start_seconds": ann.get("impact_start_source") if ann else None,
            "event_end_seconds": ann.get("impact_end_source") if ann else None,
            "training_eligible": eligible, "sha256": digest,
            "camera_id": None, "day_night": None, "weather": None, "occlusion": None,
        }
        video_rows.append(row)

    duplicate_groups = [items for items in hashes.values() if len(items) > 1]
    labeled = [r for r in video_rows if r["training_eligible"]]
    split = grouped_split(labeled, seed=seed)
    split_rows = [{"path": row["path"], "record_id": row["record_id"], "source_video_id": row["source_video_id"],
                   "source_recording_id": row["source_recording_id"], "incident_id": row["incident_id"],
                   "split": split["assignments"].get(i)} for i, row in enumerate(labeled)]
    (output / "video_audit.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in video_rows), encoding="utf-8")
    (output / "split_manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in split_rows), encoding="utf-8")

    usable = [r for r in video_rows if r["ok"]]
    inspection_unavailable = [r for r in video_rows if str(r.get("error", "")).startswith("inspection_tools_unavailable")]
    corrupt = [r for r in video_rows if not r["ok"] and r not in inspection_unavailable]
    known_normal = [r for r in video_rows if r["training_eligible"] and r["clip_label"] == "normal"]
    field_counts = {field: dict(Counter(str(r[field]) for r in video_rows))
                    for field in ("footage_type", "country", "permission_status", "review_status", "annotation_review_status")}
    report = {
        "video_files_found": len(paths), "usable_videos": len(usable), "corrupt_or_unreadable": len(corrupt),
        "inspection_unavailable_due_to_missing_media_tools": len(inspection_unavailable),
        "phase1": phase1_counts, "existing_clips_directory": str(existing),
        "existing_clips_directory_found": existing.is_dir(),
        "existing_clips_video_count": sum(1 for p in paths if existing in Path(p).parents),
        "candidate_source_record_count": len(records),
        "distinct_source_video_ids": len({r.get("source_video_id") for r in records.values() if r.get("source_video_id")}),
        "source_records_without_video_id": sum(1 for r in records.values() if not r.get("source_video_id")),
        "missing_labels": sum(1 for r in video_rows if not r["clip_label"]),
        "human_confirmed_training_eligible_videos": len(labeled),
        "split_group_count": split["group_count"], "split_video_counts": split["split_counts"],
        "unassigned_for_missing_incident_or_source_identity": len(split["unassigned"]),
        "independent_reviewed_incidents": len({r["incident_id"] for r in labeled if r["incident_id"]}),
        "normal_footage_hours_human_confirmed": sum(r["duration_seconds"] for r in known_normal) / 3600,
        "exact_duplicate_groups": duplicate_groups,
        "near_duplicate_review_status_counts": phase1_counts.get("duplicate_pairs", {}),
        "camera_day_night_weather_occlusion": field_counts,
        "camera_disjoint_evaluation_possible": any(r.get("camera_id") for r in video_rows),
        "footage_viewpoint_counts": dict(Counter(str(r.get("footage_type", "unknown")) for r in video_rows)),
        "lighting_day_night_unknown_counts": dict(Counter(str(r.get("day_night") or "unknown") for r in video_rows)),
        "weather_unknown_counts": dict(Counter(str(r.get("weather") or "unknown") for r in video_rows)),
        "occlusion_unknown_counts": dict(Counter(str(r.get("occlusion") or "unknown") for r in video_rows)),
        "warnings": [
            "No 3,000-clip directory was supplied or found under the repository; pass --existing-clips when its location is known.",
            "Assistant-generated labels, unreviewed records and publisher metadata are excluded from training.",
            "Camera ID, day/night, weather and occlusion are unknown unless explicitly reviewed.",
            "Zero eligible labels or insufficient class/event counts prevent a meaningful R3D fit and event recall estimate.",
        ],
        "audit_version": 1,
        "environment": environment_report(repo_root),
    }
    (output / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

