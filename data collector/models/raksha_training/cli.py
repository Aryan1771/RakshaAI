"""Command line interface for the reproducible Phase 2 pipeline."""
import argparse
import json
from pathlib import Path


def _config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    # Config paths are stored relative to this project root, so the complete
    # checkout can be relocated without editing its JSON configuration.
    root = path.parents[2]
    for key, value in config.get("paths", {}).items():
        if key.endswith("root") or key == "existing_clips":
            p = Path(value)
            config["paths"][key] = str((root / p).resolve() if not p.is_absolute() else p)
    return config, root


def _dump(value):
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="raksha-train", description="RakshaAI Phase 2 audit, training and inference")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "uvh-audit", "annotate-uvh-available", "prepare-uvh-available", "prepare-uvh", "manifest-phase1", "smoke", "train-r3d", "train-yolo", "eval-r3d", "eval-events", "compare-r3d", "infer"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", default="models/configs/phase2.full.json")
        if name == "audit":
            cmd.add_argument("--existing-clips", help="Override configured path to the additional video archive")
        if name == "train-r3d":
            cmd.add_argument("--variant", default="baseline", choices=["baseline", "class_weighted", "balanced_sampling", "focal_loss"])
            cmd.add_argument("--resume")
        if name == "eval-r3d":
            cmd.add_argument("--checkpoint", required=True)
            cmd.add_argument("--split", default="test", choices=["validation", "test", "official_test"])
            cmd.add_argument("--threshold", type=float, help="Frozen threshold selected on validation; required for untouched test")
            cmd.add_argument("--final-test", action="store_true")
        if name == "infer":
            cmd.add_argument("--source", required=True, help="video file, authorized RTSP URL, or webcam index")
            cmd.add_argument("--yolo-checkpoint")
            cmd.add_argument("--r3d-checkpoint")
            cmd.add_argument("--show", action="store_true")
            cmd.add_argument("--max-frames", type=int)
            cmd.add_argument("--camera-id")
            cmd.add_argument("--positive-threshold", type=float, help="Frozen validation operating threshold for persistence gate")
        if name == "eval-events":
            cmd.add_argument("--annotations", required=True, help="Pre-split JSONL with incident onset/end and camera IDs")
            cmd.add_argument("--predictions", required=True, help="Raw timestamped prediction JSONL from inference")
            cmd.add_argument("--camera-hours", required=True, help="JSON mapping camera IDs to observed evaluation hours")
            cmd.add_argument("--split", default="test", choices=["validation", "test"])
            cmd.add_argument("--threshold", type=float, help="Frozen validation threshold; required on test")
            cmd.add_argument("--final-test", action="store_true")
        if name == "compare-r3d":
            cmd.add_argument("--runs", nargs="+", required=True, help="R3D variant output directories")
    args = parser.parse_args(argv)
    cfg, repo_root = _config(args.config)
    paths = cfg["paths"]
    out = Path(paths["output_root"])
    if args.command == "audit":
        from .audit import run_audit
        existing = Path(args.existing_clips) if args.existing_clips else Path(paths["existing_clips"])
        if not existing.is_absolute():
            existing = repo_root / existing
        result = run_audit(repo_root, paths["phase1_root"], existing, out / "audit")
    elif args.command == "uvh-audit":
        from .uvh import audit_uvh26
        result = audit_uvh26(paths["uvh26_root"])
        destination = out / "uvh26_audit.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        result["saved_to"] = str(destination.resolve())
    elif args.command == "annotate-uvh-available":
        from .uvh import annotate_available_uvh26
        result = annotate_available_uvh26(paths["uvh26_root"], Path(paths["uvh26_root"]) / "derived" / "MV_available")
    elif args.command == "prepare-uvh-available":
        from .uvh import prepare_available_uvh26
        uvh = cfg["uvh26"]
        output_root = Path(paths.get("uvh26_partial_root", Path(paths["uvh26_root"]) / "derived" / "partial_experiment"))
        result = prepare_available_uvh26(paths["uvh26_root"], output_root,
                                         seed=cfg["seed"], validation_fraction=uvh["selection_val_fraction"])
    elif args.command == "prepare-uvh":
        from .uvh import prepare_uvh26
        uvh = cfg["uvh26"]
        prepared_root = Path(paths.get("uvh26_prepared_root", out / "uvh26"))
        result = prepare_uvh26(paths["uvh26_root"], prepared_root, variant=uvh["variant"], seed=cfg["seed"],
                               validation_fraction=uvh["selection_val_fraction"])
    elif args.command == "manifest-phase1":
        from .manifests import phase1_rows, split_reviewed_rows, write_video_manifest
        rows = phase1_rows(paths["phase1_root"])
        assigned, report = split_reviewed_rows(rows, seed=cfg["seed"])
        manifest = out / "videos.jsonl"
        write_video_manifest(assigned, manifest)
        result = {**report, "manifest": str(manifest.resolve()), "source_annotations": len(rows),
                  "note": "Ineligible/pending annotations are excluded, never upgraded to ground truth."}
    elif args.command == "smoke":
        from .smoke import run_smoke
        result = run_smoke(out, seed=cfg["seed"], run_yolo=True)
    elif args.command == "train-r3d":
        from .training import train_r3d
        manifest = Path(paths.get("video_manifest", out / "videos.jsonl"))
        result = train_r3d(cfg, manifest, output_dir=out / "r3d" / args.variant, variant=args.variant, resume=args.resume)
    elif args.command == "train-yolo":
        from .yolo import train_yolo
        prepared = Path(paths.get("uvh26_prepared_root", paths.get("uvh26_partial_root", out / "uvh26")))
        yaml_path = prepared / "selection.yaml"
        official_yaml = prepared / "official_benchmark.yaml"
        result = train_yolo(cfg, yaml_path, output_dir=out / "yolo", official_test_yaml=str(official_yaml) if official_yaml.exists() else None)
    elif args.command == "eval-r3d":
        from .evaluation import evaluate_r3d
        if args.split in {"test", "official_test"} and (not args.final_test or args.threshold is None):
            parser.error("test evaluation requires --final-test and an explicit --threshold frozen from validation")
        threshold = args.threshold if args.threshold is not None else 0.5
        manifest = Path(paths.get("video_manifest", out / "videos.jsonl"))
        result = evaluate_r3d(args.checkpoint, manifest, split=args.split, threshold=threshold,
                              output_dir=out / "evaluation" / args.split, config=cfg, final_test=args.final_test)
    elif args.command == "eval-events":
        if args.split == "test" and not args.final_test:
            parser.error("untouched event test requires --final-test after configuration freeze")
        from .metrics import choose_operating_point, event_evaluation
        from .stream import PersistenceGate
        from .video_data import load_jsonl
        annotations = [row for row in load_jsonl(args.annotations) if row.get("split") == args.split]
        raw_predictions = [row for row in load_jsonl(args.predictions) if row.get("crash_probability") is not None]
        camera_hours = json.loads(Path(args.camera_hours).read_text(encoding="utf-8"))
        persistence_cfg = cfg.get("stream", {}).get("persistence", {})
        thresholds = [args.threshold] if args.threshold is not None else cfg["r3d"].get("thresholds", [.1, .2, .3, .4, .5, .6, .7, .8, .9])
        if args.split == "test" and args.threshold is None:
            parser.error("test event evaluation requires --threshold frozen from validation")
        evaluated = []
        for threshold in thresholds:
            gate = PersistenceGate(**{**persistence_cfg, "positive_threshold": float(threshold)})
            candidates = []
            for row in sorted(raw_predictions, key=lambda item: float(item.get("decision_timestamp_seconds", item["frame_timestamp_seconds"]))):
                if gate.update(float(row["crash_probability"])):
                    candidates.append({"camera_id": row.get("camera_id") or "unknown",
                                       "alert_timestamp_seconds": float(row.get("decision_timestamp_seconds", row["frame_timestamp_seconds"])),
                                       "score": float(row["crash_probability"])})
            evaluated.append({"threshold": float(threshold),
                "metrics_by_deadline": event_evaluation(annotations, candidates, camera_hours,
                    deadlines=(1, 3, 5, 10), early_tolerance=float(cfg["r3d"].get("early_match_tolerance_seconds", 0))),
                "candidate_alert_count": len(candidates)})
        if args.split == "validation":
            op_rows = []
            for item in evaluated:
                five = next(r for r in item["metrics_by_deadline"] if r["deadline_seconds"] == float(cfg["r3d"].get("event_deadline_seconds", 5)))
                op_rows.append({"threshold": item["threshold"], "recall_sensitivity": five["event_recall"],
                                "precision": (five["matched_incidents"] / (five["matched_incidents"] + five["false_alert_candidates"] + five["duplicate_alerts"]))
                                    if (five["matched_incidents"] + five["false_alert_candidates"] + five["duplicate_alerts"]) else None,
                                "false_alerts_per_camera_hour": five["false_alerts_per_camera_hour"],
                                "false_alert_candidates": five["false_alert_candidates"],
                                "duplicate_alerts": five["duplicate_alerts"]})
            operating_point = choose_operating_point(op_rows, recall_target=float(cfg["r3d"].get("recall_target", .98)),
                false_alert_budget_per_camera_hour=float(cfg["r3d"].get("false_alert_budget_per_camera_hour", .1)))
        else:
            operating_point = None
        result = {"split": args.split, "final_test": args.final_test,
                  "rule": "Threshold sweep replays persistence/rearm on timestamped validation scores, then matches emitted alerts one-to-one by camera. Test uses the frozen threshold and is never used for selection.",
                  "threshold_results": evaluated, "validation_operating_point": operating_point,
                  "incident_count": len(annotations), "camera_hours": camera_hours,
                  "deadline_seconds": [1, 3, 5, 10], "camera_hours_source": "caller-supplied measured stream duration"}
        target = out / "evaluation" / f"events_{args.split}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        result["saved_to"] = str(target.resolve())
    elif args.command == "compare-r3d":
        import csv
        rows = []
        for run_path in args.runs:
            summary_path = Path(run_path) / "training_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            rows.append({key: summary.get(key) for key in
                         ("variant", "best_validation_average_precision", "best_epoch", "train_windows",
                          "validation_windows", "train_unique_incidents", "validation_unique_incidents", "device")})
        target = out / "experiment_comparison.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        result = {"runs": rows, "comparison_csv": str(target.resolve()),
                  "selection_data": "validation only; no final test results are ranked here"}
    elif args.command == "infer":
        import torch
        if args.positive_threshold is not None:
            cfg.setdefault("stream", {}).setdefault("persistence", {})["positive_threshold"] = args.positive_threshold
        yolo_model = None
        r3d_model = r3d_state = None
        if args.yolo_checkpoint:
            from ultralytics import YOLO
            yolo_model = YOLO(args.yolo_checkpoint)
        if args.r3d_checkpoint:
            from .training import load_r3d_checkpoint
            r3d_model, r3d_state, model_info, device = load_r3d_checkpoint(args.r3d_checkpoint)
            r3d_state["model_info"] = model_info
        from .stream import run_inference
        result = run_inference(args.source, yolo_model=yolo_model, r3d_model=r3d_model, r3d_state=r3d_state,
                               config=cfg, camera_id=args.camera_id, output_dir=out / "inference", show=args.show, max_frames=args.max_frames)
    _dump(result)


if __name__ == "__main__":
    main()
