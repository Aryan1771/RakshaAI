import json
from pathlib import Path

from types import SimpleNamespace

from models.raksha_training.metrics import classification_metrics, event_evaluation, merge_predictions
from models.raksha_training.splits import grouped_split
from models.raksha_training.temporal import label_window, sample_timestamps
from models.raksha_training.uvh import annotate_available_uvh26, prepare_available_uvh26, validate_coco
from models.raksha_training.video_data import build_window_manifest
from models.raksha_training.stream import OpenCVSource
from models.raksha_training.yolo import _dataset_provenance, _resolve_ultralytics_yaml, yolo_result_metrics


def test_dataset_provenance_survives_dataset_cleanup(tmp_path):
    root = tmp_path / "UVH-26"
    prepared = root / "derived" / "prepared"
    prepared.mkdir(parents=True)
    yaml_path = prepared / "selection.runtime.yaml"
    yaml_path.write_text("path: .", encoding="utf-8")
    (root / "source_manifest.json").write_text(json.dumps({
        "dataset": "iisc-aim/UVH-26", "source_url": "https://huggingface.co/datasets/iisc-aim/UVH-26",
        "revision": "abc123", "license": "CC-BY-4.0", "attribution": "Authors; cite paper",
    }), encoding="utf-8")

    provenance = _dataset_provenance(yaml_path, {"uvh26": {"variant": "MV"}})

    assert provenance["source_revision"] == "abc123"
    assert provenance["license"] == "CC-BY-4.0"
    assert provenance["annotation_variant"] == "MV"
    assert "not crash-video labels" in provenance["note"]


def test_uvh_available_export_writes_vehicle_sidecar_without_crash_labels(tmp_path):
    import cv2
    import numpy as np

    root = tmp_path / "UVH-26"
    image_root = root / "snapshot/UVH-26-Train/data/000"
    image_root.mkdir(parents=True)
    image = image_root / "one.png"
    assert cv2.imwrite(str(image), np.zeros((20, 40, 3), dtype=np.uint8))
    (root / "snapshot/UVH-26-Train/UVH-26-MV-Train.json").write_text(json.dumps({
        "images": [{"id": 1, "file_name": "one.png", "width": 40, "height": 20}],
        "categories": [{"id": 3, "name": "car"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 3, "bbox": [10, 5, 20, 10]}],
    }), encoding="utf-8")
    output = root / "derived/MV_available"

    result = annotate_available_uvh26(root, output)

    label = output / "labels/000/one.txt"
    manifest = json.loads((output / "annotation_manifest.json").read_text(encoding="utf-8"))
    assert result["converted_available_images"] == 1
    assert label.read_text(encoding="utf-8").strip() == "0 0.50000000 0.50000000 0.50000000 0.50000000"
    assert manifest["label_type"] == "vehicle_detection_yolo_boxes"
    assert manifest["crash_labels_created"] is False
    assert not (output / "images").exists()
    assert manifest["image_payloads_copied"] is False


def test_training_config_paths_resolve_from_project_root():
    from models.raksha_training.cli import _config

    config, root = _config("models/configs/phase2.full.json")
    assert config["paths"]["uvh26_root"] == str((root / "dataset/UVH-26").resolve())
    assert config["paths"]["uvh26_prepared_root"] == str((root / "dataset/UVH-26/derived/prepared").resolve())
    assert config["paths"]["output_root"] == str((root / "models/runs/phase2").resolve())


def test_partial_uvh_prep_groups_exact_duplicate_images_and_marks_exploratory(tmp_path):
    import cv2
    import numpy as np

    root = tmp_path / "UVH-26"
    image_root = root / "snapshot/UVH-26-Train/data/000"
    image_root.mkdir(parents=True)
    images = []
    anns = []
    for image_id in range(1, 5):
        pixels = np.full((20, 30, 3), 50 if image_id in {1, 4} else image_id * 20, dtype=np.uint8)
        name = f"{image_id}.png"
        assert cv2.imwrite(str(image_root / name), pixels)
        images.append({"id": image_id, "file_name": name, "width": 30, "height": 20})
        anns.append({"id": image_id, "image_id": image_id, "category_id": 1, "bbox": [5, 5, 10, 8]})
    (root / "snapshot/UVH-26-Train/UVH-26-MV-Train.json").write_text(json.dumps({
        "images": images, "categories": [{"id": 1, "name": "Car"}], "annotations": anns,
    }), encoding="utf-8")
    (root / "source_manifest.json").write_text(json.dumps({"revision": "test", "download_status": "incomplete"}), encoding="utf-8")
    output = root / "derived/partial"

    manifest = prepare_available_uvh26(root, output, seed=42, validation_fraction=0.25)

    train_names = {p.name for p in (output / "selection/train/images").glob("*.png")}
    val_names = {p.name for p in (output / "selection/validation/images").glob("*.png")}
    assert train_names and val_names
    assert (("1.png" in train_names) == ("4.png" in train_names))
    assert (("1.png" in val_names) == ("4.png" in val_names))
    assert manifest["exploratory_only"] is True
    assert manifest["official_test_used"] is False
    assert manifest["test_partition"] is None
    assert manifest["crash_labels_created"] is False


def test_coco_conversion_schema_maps_sparse_ids_and_rejects_invalid_boxes():
    doc = {"images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 50}],
           "categories": [{"id": 9, "name": "truck"}, {"id": 2, "name": "bike"}],
           "annotations": [{"id": 1, "image_id": 1, "category_id": 9, "bbox": [10, 5, 20, 10]},
                           {"id": 2, "image_id": 1, "category_id": 2, "bbox": [90, 0, 20, 5]}]}
    result = validate_coco(doc)
    assert result["class_map"] == {2: 0, 9: 1}
    assert result["labels"][1] == ["1 0.20000000 0.20000000 0.20000000 0.20000000"]
    assert len(result["invalid_annotations"]) == 1


def test_timestamp_sampling_and_uncertain_temporal_boundaries():
    assert sample_timestamps(2, 4, 2) == [2, 2.5, 3, 3.5]
    assert label_window(0, 2, clip_label="crash", event_start=5, event_end=7).label == "normal"
    assert label_window(3.5, 5.5, clip_label="crash", event_start=5, event_end=7).target is None
    assert label_window(5, 7, clip_label="crash", event_start=5, event_end=7).target == 1


def test_split_keeps_incident_and_source_recording_together():
    rows = [{"incident_id": "one", "source_recording_id": "a"},
            {"incident_id": "one", "source_recording_id": "b"},
            {"incident_id": "two", "source_recording_id": "c"}]
    result = grouped_split(rows, seed=5)
    assert result["assignments"][0] == result["assignments"][1]
    assert result["assignments"][0] != result["assignments"][2]


def test_metrics_keep_undefined_denominators_explicit():
    row = classification_metrics([0, 0], [.1, .2])
    assert row["recall_sensitivity"] is None
    assert row["roc_auc"] is None
    assert row["undefined"]["recall"] is True


def test_event_matching_is_one_to_one_and_reports_delay():
    events = [{"incident_id": "i1", "camera_id": "c", "event_start_seconds": 10, "event_end_seconds": 12},
              {"incident_id": "i2", "camera_id": "c", "event_start_seconds": 20, "event_end_seconds": 22}]
    candidates = [{"camera_id": "c", "alert_timestamp_seconds": 11},
                  {"camera_id": "c", "alert_timestamp_seconds": 11.5},
                  {"camera_id": "c", "alert_timestamp_seconds": 22}]
    result = event_evaluation(events, candidates, {"c": 1.0}, deadlines=(3,))[0]
    assert result["matched_incidents"] == 2
    assert result["duplicate_alerts"] == 1
    assert result["median_detection_delay_seconds"] == 1.5
    assert result["false_alerts_per_camera_hour"] == 0


def test_prediction_merge_keeps_cameras_separate():
    rows = [{"positive": True, "camera_id": "a", "decision_timestamp_seconds": t, "score": .8}
            for t in (1, 2)] + [{"positive": True, "camera_id": "b", "decision_timestamp_seconds": 2, "score": .9}]
    assert len(merge_predictions(rows, 2)) == 2


def test_webcam_numeric_source_is_not_mistaken_for_file(monkeypatch):
    class Capture:
        def isOpened(self): return True
        def get(self, key): return 30.0
        def release(self): pass
    captured = []
    monkeypatch.setattr("models.raksha_training.stream.cv2.VideoCapture", lambda source: captured.append(source) or Capture())
    source = OpenCVSource("0")
    assert source.source == 0 and source.is_file is False
    assert captured == [0]
    source.close()


def test_positive_clip_without_event_bounds_is_clip_only(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.touch()
    rows, _ = build_window_manifest([{"clip_path": str(clip), "duration_seconds": 30,
                                      "clip_label": "crash", "incident_id": "i", "source_video_id": "s"}],
                                    require_eligible=False)
    assert len(rows) == 1
    assert rows[0]["temporal_known"] is False
    assert rows[0]["sampling_mode"] == "uniform_over_full_clip"


def test_yolo_metrics_make_undefined_values_json_safe():
    box = SimpleNamespace(mp=float("nan"), mr=float("nan"), map50=float("nan"), map=float("nan"),
                          ap_class_index=[], all_ap=[], p=[], r=[])
    result = SimpleNamespace(box=box, names={}, speed={"inference": 1.2})
    metrics = yolo_result_metrics(result)
    assert metrics["map50"] is None
    assert json.loads(json.dumps(metrics))["map50_95"] is None


def test_yolo_yaml_relative_root_resolves_next_to_yaml_not_global_settings(tmp_path):
    import yaml

    yaml_path = tmp_path / "prepared" / "selection.yaml"
    yaml_path.parent.mkdir()
    yaml_path.write_text("path: selection\ntrain: train/images\nval: validation/images\nnames: [car]\n", encoding="utf-8")

    runtime_path = _resolve_ultralytics_yaml(yaml_path)
    doc = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))
    assert Path(doc["path"]) == (yaml_path.parent / "selection").resolve()
    assert doc["train"] == "train/images"

