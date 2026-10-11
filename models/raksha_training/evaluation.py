"""Frozen-checkpoint R3D evaluation and validation-only operating-point sweeps."""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import classification_metrics, grouped_recall_bootstrap, threshold_sweep
from .models import build_r3d18
from .training import _evaluate
from .video_data import VideoWindowDataset, build_window_manifest, load_jsonl


def evaluate_r3d(checkpoint, manifest_path, *, split="test", threshold=0.5, output_dir,
                 config=None, final_test=False):
    if split not in {"validation", "test", "official_test"}:
        raise ValueError("split must be validation, test, or official_test")
    if split in {"test", "official_test"} and not final_test:
        raise ValueError("Untouched test splits may only be evaluated once after configuration freeze (final_test=True)")
    checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = config or checkpoint_state["config"]
    rows = [r for r in load_jsonl(manifest_path) if r.get("split") == split]
    windows, rejected = build_window_manifest(rows,
        window_seconds=float(config["video"]["temporal_span_seconds"]),
        stride_seconds=float(config["video"]["window_stride_seconds"]),
        boundary_guard=float(config["video"].get("boundary_guard_seconds", .5)),
        positive_overlap_fraction=float(config["video"].get("positive_overlap_fraction", .5)))
    if not windows:
        raise ValueError(f"No eligible windows in frozen {split} split; rejected {rejected}")
    model, info = build_r3d18(2, None)
    model.load_state_dict(checkpoint_state["model"])
    requested_device = config.get("r3d", {}).get("device", "auto")
    if requested_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for R3D evaluation, but this PyTorch installation cannot access CUDA")
    model.to(device)
    norm = checkpoint_state.get("model_info", info)
    dataset = VideoWindowDataset(windows, split=split, config=config["video"],
                                 mean=norm["normalization_mean"], std=norm["normalization_std"])
    loader = DataLoader(dataset, batch_size=int(config["r3d"]["batch_size"]), shuffle=False,
                        num_workers=int(config["r3d"]["workers"]), pin_memory=device.type == "cuda")
    criterion = torch.nn.CrossEntropyLoss()
    metrics, truth, probabilities, metadata = _evaluate(model, loader, device, criterion)
    metrics.update(classification_metrics(truth, probabilities, threshold))
    # The bootstrap unit is incident, not a repeated window.
    incident_predictions = {}
    for meta, y, p in zip(metadata, truth, probabilities):
        incident = meta.get("incident_id")
        if incident:
            item = incident_predictions.setdefault(incident, {"target": int(y), "max_probability": float(p)})
            item["max_probability"] = max(item["max_probability"], float(p))
    positive_incidents = [v for v in incident_predictions.values() if v["target"] == 1]
    bootstrap = grouped_recall_bootstrap([int(v["max_probability"] >= threshold) for v in positive_incidents])
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = {"split": split, "final_test": bool(final_test), "checkpoint": str(Path(checkpoint).resolve()),
              "threshold_frozen_from_validation": float(threshold), "metrics": metrics,
              "positive_incident_grouped_recall_bootstrap": bootstrap, "rejected_windows": rejected,
              "independent_positive_incidents": len(positive_incidents),
              "evaluation_unit_warning": "Window metrics include correlated windows; incident bootstrap does not treat windows as independent.",
              "event_detection_delay": None if not any(m["temporal_known"] and m["camera_id"] for m in metadata)
              else "Use continuous-stream predictions and event-evaluation; window classification alone is not event delay."}
    (out / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for meta, y, p in zip(metadata, truth, probabilities):
            stream.write(json.dumps({**meta, "target": int(y), "crash_probability": float(p),
                                     "predicted": int(p >= threshold)}) + "\n")
    with (out / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true_label", "predicted_non_crash", "predicted_crash"])
        writer.writerow(["non_crash", metrics["tn"], metrics["fp"]])
        writer.writerow(["crash", metrics["fn"], metrics["tp"]])
    if split == "validation":
        r3d_config = config.get("r3d", {})
        sweep = sweep_validation_thresholds(
            truth, probabilities, r3d_config.get("thresholds", [i / 100 for i in range(5, 100, 5)]),
            negative_camera_hours=None,
            recall_target=float(r3d_config.get("recall_target", .98)),
            false_alert_budget=float(r3d_config.get("false_alert_budget_per_camera_hour", .1)))
        sweep_path = out / "validation_threshold_sweep.csv"
        columns = ["threshold", "tp", "fp", "tn", "fn", "precision", "recall_sensitivity", "f1",
                   "average_precision_pr_auc", "roc_auc", "false_alerts_per_camera_hour"]
        with sweep_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for row in sweep["rows"]:
                writer.writerow({key: row.get(key) for key in columns})
        result["validation_operating_point"] = sweep["operating_point"]
        result["validation_operating_point"]["selection_limitation"] = (
            "False alerts per camera-hour unavailable because this manifest has no measured continuous negative camera-hours."
        )
        result["validation_threshold_sweep_csv"] = str(sweep_path.resolve())
        try:
            from sklearn.metrics import precision_recall_curve
            import matplotlib.pyplot as plt
            precision, recall, _ = precision_recall_curve(truth, probabilities)
            fig, ax = plt.subplots()
            ax.plot(recall, precision)
            ax.set(xlabel="Recall / sensitivity", ylabel="Precision", title="R3D-18 validation precision–recall")
            ax.grid(True, alpha=.3)
            fig.savefig(out / "validation_precision_recall.png", dpi=160, bbox_inches="tight")
            plt.close(fig)
            result["validation_precision_recall_plot"] = str((out / "validation_precision_recall.png").resolve())
        except Exception as exc:
            result["validation_precision_recall_plot_error"] = type(exc).__name__
        (out / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def sweep_validation_thresholds(truth, probabilities, thresholds, *, negative_camera_hours=None,
                                recall_target=.98, false_alert_budget=.1):
    rows = threshold_sweep(truth, probabilities, thresholds, negative_camera_hours)
    from .metrics import choose_operating_point
    selection = choose_operating_point(rows, recall_target=recall_target,
                                       false_alert_budget_per_camera_hour=false_alert_budget)
    return {"rows": rows, "operating_point": selection,
            "selected_on": "validation only; freeze this threshold before test evaluation"}

