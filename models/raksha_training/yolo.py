"""YOLOv8 detection training and benchmark reporting."""
from datetime import datetime, timezone
from importlib import metadata as package_metadata
import json
import math
from pathlib import Path
import time


def _resolve_ultralytics_yaml(path):
    """Resolve dataset roots beside their YAML before Ultralytics applies its global datasets dir."""
    import yaml

    path = Path(path).resolve()
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = Path(document.get("path", "."))
    if not root.is_absolute():
        root = path.parent / root
    document["path"] = str(root.resolve())
    runtime_path = path.with_name(f"{path.stem}.runtime.yaml")
    runtime_path.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return runtime_path


def _dataset_provenance(data_yaml, config):
    path = Path(data_yaml).resolve()
    source_manifest = next((parent / "source_manifest.json" for parent in path.parents
                            if (parent / "source_manifest.json").is_file()), None)
    source = json.loads(source_manifest.read_text(encoding="utf-8")) if source_manifest else {}
    return {
        "dataset": source.get("dataset", "iisc-aim/UVH-26"),
        "source_url": source.get("source_url", "https://huggingface.co/datasets/iisc-aim/UVH-26"),
        "source_revision": source.get("revision"),
        "license": source.get("license"),
        "attribution": source.get("attribution"),
        "annotation_variant": config.get("uvh26", {}).get("variant", "MV"),
        "data_yaml": str(path),
        "note": "UVH-26 is used here for vehicle detection only; its images are not crash-video labels.",
    }


def train_yolo(config, data_yaml, *, output_dir, model_path=None, official_test_yaml=None):
    import torch
    from ultralytics import YOLO

    cfg = config["yolo"]
    seed = int(cfg.get("seed", config.get("seed", 42)))
    device = cfg.get("device", "auto")
    if device == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"
    model_source = model_path or cfg["model"]
    model = YOLO(model_source)
    data_yaml = _resolve_ultralytics_yaml(data_yaml)
    if official_test_yaml:
        official_test_yaml = _resolve_ultralytics_yaml(official_test_yaml)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_result = model.train(
        data=str(data_yaml), imgsz=int(cfg["imgsz"]), batch=int(cfg["batch"]),
        epochs=int(cfg["epochs"]), patience=int(cfg["patience"]), optimizer=cfg["optimizer"],
        lr0=float(cfg["lr0"]), weight_decay=float(cfg.get("weight_decay", 0.0005)),
        workers=int(cfg["workers"]), amp=bool(cfg["amp"] and torch.cuda.is_available()),
        fraction=float(cfg.get("fraction", 1.0)),
        seed=seed, deterministic=bool(cfg.get("deterministic", True)), device=device,
        project=str(out), name="yolov8", exist_ok=True,
        **cfg.get("augment", {}),
    )
    elapsed = time.perf_counter() - start
    training_peak_cuda_memory = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    save_dir = Path(getattr(train_result, "save_dir", out / "yolov8"))
    best = save_dir / "weights/best.pt"
    if not best.is_file():
        last = save_dir / "weights/last.pt"
        if not last.is_file():
            raise RuntimeError("Ultralytics training completed without a best.pt or last.pt checkpoint")
        best = last
    metadata = {
        "model": "YOLOv8-S" if "yolov8s" in model_source.lower() else model_source,
        "initialization": cfg.get("pretrained_source"), "initialization_checkpoint": model_source,
        "transfer_head": "Ultralytics initializes/adapts the class-specific detection head to the 14 UVH-26 classes; compatible COCO weights transfer.",
        "ultralytics_version": __import__("ultralytics").__version__, "torch_version": torch.__version__,
        "torchvision_version": package_metadata.version("torchvision"), "opencv_version": package_metadata.version("opencv-python"),
        "scikit_learn_version": package_metadata.version("scikit-learn"),
        "device": str(device), "config": cfg, "data_yaml": str(Path(data_yaml).resolve()),
        "dataset_provenance": _dataset_provenance(data_yaml, config),
        "best_checkpoint": str(best.resolve()), "train_seconds": elapsed,
        "selection_validation_metrics": yolo_result_metrics(train_result),
        "training_peak_cuda_memory_bytes": training_peak_cuda_memory,
        "small_object_metrics": {"status": "not_computed", "reason": "No size-stratified prediction evaluation was run; COCO ground-truth area support is recorded in the UVH audit."},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_criterion": cfg.get("selection_metric", "metrics/mAP50-95(B)"),
    }
    test_metrics = None
    if official_test_yaml:
        # This is one final evaluation on the untouched supplied validation split.
        final_model = YOLO(str(best))
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        result = final_model.val(data=str(official_test_yaml), split="val", plots=True, project=str(out), name="official_uvh26_benchmark", exist_ok=True)
        test_metrics = yolo_result_metrics(result)
        test_metrics["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        metadata["official_benchmark"] = {"data_yaml": str(Path(official_test_yaml).resolve()),
                                           "metrics": test_metrics, "evaluated_once_after_selection": True}
    (out / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if test_metrics is not None:
        (out / "official_benchmark_metrics.json").write_text(json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def yolo_result_metrics(result):
    box = result.box
    names = getattr(result, "names", {})
    per_class = {}
    class_indices = list(getattr(box, "ap_class_index", []))
    all_ap = getattr(box, "all_ap", None)
    precisions = getattr(box, "p", [])
    recalls = getattr(box, "r", [])
    for position, class_id in enumerate(class_indices):
        name = names.get(int(class_id), str(class_id)) if isinstance(names, dict) else str(class_id)
        ap_values = all_ap[position].tolist() if all_ap is not None and position < len(all_ap) else []
        per_class[name] = {"class_id": int(class_id),
                           "precision": _finite(precisions[position]) if position < len(precisions) else None,
                           "recall": _finite(recalls[position]) if position < len(recalls) else None,
                           "ap50": _finite(ap_values[0]) if ap_values else None,
                           "ap50_95": _finite(sum(ap_values) / len(ap_values)) if ap_values else None}
    return {"precision": _finite(getattr(box, "mp", None)),
            "recall": _finite(getattr(box, "mr", None)),
            "map50": _finite(getattr(box, "map50", None)),
            "map50_95": _finite(getattr(box, "map", None)),
            "per_class": per_class,
            "speed_ms_per_image": getattr(result, "speed", None)}


def _finite(value):
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None

