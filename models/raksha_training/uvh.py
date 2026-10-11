"""UVH-26 COCO audit and YOLO conversion. Uses the MV variant only by default."""
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil

from .splits import grouped_split


def audit_uvh26(root):
    root = Path(root).resolve()
    manifest_path = root / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    snapshot = root / "snapshot"
    outputs = {"dataset": "iisc-aim/UVH-26", "revision": manifest.get("revision"),
               "download_status": manifest.get("download_status", "unknown"),
               "source_manifest_last_verified_at": manifest.get("downloaded_at"),
               "expected_file_count": manifest.get("expected_file_count"),
               "downloaded_expected_file_count": manifest.get("downloaded_expected_file_count"),
               "expected_bytes": manifest.get("expected_bytes"),
               "downloaded_bytes": manifest.get("downloaded_bytes"), "variants": {},
               "official_validation": {"available": False, "images": None, "annotations": None},
               "camera_ids_present_in_coco_image_records": False,
               "notes": ["Images are traffic perception examples, not crash-video labels.",
                         "Do not synthesize crash clips by repeating still frames.",
                         "Manifest counts describe its last verified snapshot; live local file counts are reported separately."]}
    local_files = [p for p in snapshot.rglob("*") if p.is_file() and ".cache" not in p.relative_to(snapshot).parts]
    outputs["live_local_snapshot"] = {"source_file_count_excluding_hub_cache": len(local_files),
                                       "source_file_bytes_excluding_hub_cache": sum(p.stat().st_size for p in local_files)}
    variants = [
        ("MV", "UVH-26-Train/UVH-26-MV-Train.json"),
        ("ST", "UVH-26-Train/UVH-26-ST-Train.json"),
    ]
    for name, relative in variants:
        path = snapshot / relative
        if not path.is_file():
            outputs["variants"][name] = {"available": False, "path": relative}
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        checked = validate_coco(doc)
        small = sum(1 for a in doc["annotations"] if float(a.get("area", 0)) < 32 * 32)
        outputs["variants"][name] = {
            "available": True, "path": relative, "images": len(doc["images"]),
            "annotations": len(doc["annotations"]), "categories": [{"source_id": int(c["id"]), "name": c["name"]} for c in checked["categories"]],
            "valid_boxes": sum(checked["valid_box_counts"].values()),
            "invalid_annotation_count": len(checked["invalid_annotations"]),
            "small_area_box_count_area_lt_32_squared_pixels": small,
            "image_metadata_fields": sorted(doc["images"][0]) if doc["images"] else [],
        }
    val_path = snapshot / "UVH-26-Val/UVH-26-MV-Val.json"
    if val_path.is_file():
        val_doc = json.loads(val_path.read_text(encoding="utf-8"))
        outputs["official_validation"] = {"available": True, "images": len(val_doc.get("images", [])),
                                          "annotations": len(val_doc.get("annotations", [])),
                                          "path": val_path.relative_to(root).as_posix()}
    return outputs


def validate_coco(document):
    required = {"images", "annotations", "categories"}
    if not required.issubset(document):
        raise ValueError(f"COCO document missing {sorted(required - set(document))}")
    categories = sorted(document["categories"], key=lambda c: int(c["id"]))
    ids = [int(c["id"]) for c in categories]
    if len(ids) != len(set(ids)):
        raise ValueError("COCO category IDs must be unique")
    class_map = {category_id: index for index, category_id in enumerate(ids)}
    names = {index: str(category["name"]) for index, category in enumerate(categories)}
    images = {int(row["id"]): row for row in document["images"]}
    if len(images) != len(document["images"]):
        raise ValueError("COCO image IDs must be unique")
    labels = defaultdict(list)
    counts = Counter()
    invalid = []
    for ann in document["annotations"]:
        image_id = int(ann["image_id"])
        if image_id not in images:
            invalid.append({"annotation_id": ann.get("id"), "reason": "unknown_image_id"})
            continue
        image = images[image_id]
        bbox = ann.get("bbox")
        category_id = int(ann.get("category_id", -1))
        if category_id not in class_map:
            invalid.append({"annotation_id": ann.get("id"), "reason": "unknown_category_id"})
            continue
        if not isinstance(bbox, list) or len(bbox) != 4:
            invalid.append({"annotation_id": ann.get("id"), "reason": "bbox_must_be_xywh"})
            continue
        x, y, width, height = map(float, bbox)
        image_width, image_height = float(image["width"]), float(image["height"])
        if image_width <= 0 or image_height <= 0 or width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > image_width + 1e-3 or y + height > image_height + 1e-3:
            invalid.append({"annotation_id": ann.get("id"), "reason": "bbox_out_of_bounds_or_nonpositive"})
            continue
        xc = min(1.0, max(0.0, (x + width / 2) / image_width))
        yc = min(1.0, max(0.0, (y + height / 2) / image_height))
        wn, hn = width / image_width, height / image_height
        labels[image_id].append(f"{class_map[category_id]} {xc:.8f} {yc:.8f} {wn:.8f} {hn:.8f}")
        counts[class_map[category_id]] += 1
    return {"images": images, "labels": labels, "categories": categories, "class_map": class_map,
            "names": names, "invalid_annotations": invalid, "valid_box_counts": dict(counts)}


def annotate_available_uvh26(root, output):
    """Export MV vehicle boxes for locally present UVH train images, without copying image payloads.

    The labels are YOLO vehicle-detection annotations. They are never crash labels.
    This partial export is useful for inspection only until source download completes.
    """
    import cv2

    root, output = Path(root).resolve(), Path(output).resolve()
    train_json = root / "snapshot/UVH-26-Train/UVH-26-MV-Train.json"
    if not train_json.is_file():
        raise FileNotFoundError(f"UVH-26 MV train annotations not found: {train_json}")
    document = json.loads(train_json.read_text(encoding="utf-8"))
    audit = validate_coco(document)
    image_root = root / "snapshot/UVH-26-Train/data"
    lookup = defaultdict(list)
    for path in image_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            lookup[path.name].append(path)
    missing, ambiguous, invalid_images, image_rows = [], [], [], []
    converted = 0
    for image_id, image in audit["images"].items():
        name = str(image["file_name"])
        candidates = lookup.get(Path(name).name, [])
        if not candidates:
            missing.append(name)
            continue
        if len(candidates) != 1:
            ambiguous.append({"file_name": name, "matches": [p.relative_to(image_root).as_posix() for p in candidates]})
            continue
        source = candidates[0]
        decoded = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if decoded is None:
            invalid_images.append({"file_name": name, "reason": "decode_failed"})
            continue
        height, width = decoded.shape[:2]
        if width != int(image["width"]) or height != int(image["height"]):
            invalid_images.append({"file_name": name, "reason": "dimension_mismatch",
                                   "expected": [int(image["width"]), int(image["height"])],
                                   "actual": [width, height]})
            continue
        relative = source.relative_to(image_root)
        label_path = output / "labels" / relative.parent / f"{source.stem}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(audit["labels"].get(image_id, [])), encoding="utf-8")
        image_rows.append({"image_id": image_id,
                           "image_path": source.relative_to(root).as_posix(),
                           "label_path": label_path.relative_to(output).as_posix(),
                           "width": width, "height": height})
        converted += 1
    source_manifest_path = root / "source_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8")) if source_manifest_path.exists() else {}
    result = {
        "dataset": "iisc-aim/UVH-26", "annotation_variant": "MV (majority vote only)",
        "label_type": "vehicle_detection_yolo_boxes", "crash_labels_created": False,
        "source_revision": source_manifest.get("revision"),
        "source_download_status_at_export": source_manifest.get("download_status", "unknown"),
        "training_ready": False,
        "training_readiness_reason": "This export contains only locally available official-train images; wait for full download, verify official validation files, then run prepare-uvh.",
        "official_train_image_count": len(audit["images"]), "converted_available_images": converted,
        "missing_images": missing, "ambiguous_image_names": ambiguous, "invalid_images": invalid_images,
        "invalid_annotation_count": len(audit["invalid_annotations"]),
        "invalid_annotations": audit["invalid_annotations"],
        "valid_box_counts_by_yolo_class_id": audit["valid_box_counts"],
        "source_category_id_to_yolo_class_id": {str(k): v for k, v in audit["class_map"].items()},
        "class_names_by_yolo_class_id": {str(k): v for k, v in audit["names"].items()},
        "image_label_index": image_rows,
        "image_payloads_copied": False,
        "note": "The official UVH-26 labels describe vehicle boxes, not crashes. Do not rename these as crash<vehicle-count>(serial).",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "annotation_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {key: result[key] for key in ("converted_available_images", "official_train_image_count", "source_download_status_at_export", "training_ready")} | {"annotation_manifest": str((output / "annotation_manifest.json").resolve())}


def prepare_available_uvh26(root, output, seed=42, validation_fraction=0.12):
    """Prepare an explicitly partial exploratory train/validation split from downloaded MV images."""
    root, output = Path(root).resolve(), Path(output).resolve()
    train_json = root / "snapshot/UVH-26-Train/UVH-26-MV-Train.json"
    if not train_json.is_file():
        raise FileNotFoundError(f"UVH-26 MV train annotations not found: {train_json}")
    document = json.loads(train_json.read_text(encoding="utf-8"))
    image_root = root / "snapshot/UVH-26-Train/data"
    lookup, ambiguous = {}, set()
    for path in image_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            if path.name in lookup:
                ambiguous.add(path.name)
            else:
                lookup[path.name] = path
    available = [row for row in document["images"]
                 if Path(str(row["file_name"])).name in lookup and Path(str(row["file_name"])).name not in ambiguous]
    if len(available) < 2:
        raise ValueError("At least two uniquely matched downloaded images are required for an exploratory split")
    available_ids = {int(row["id"]) for row in available}
    available_doc = {**document, "images": available,
                     "annotations": [a for a in document["annotations"] if int(a["image_id"]) in available_ids]}
    available_audit = validate_coco(available_doc)
    hashes = {image_id: _sha(lookup[Path(str(image["file_name"])).name])
              for image_id, image in available_audit["images"].items()}
    rows = [{"source_video_id": digest, "image_id": image_id} for image_id, digest in hashes.items()]
    split = grouped_split(rows, seed=seed, fractions=(1 - validation_fraction, validation_fraction, 0.0))
    assignments = {int(rows[index]["image_id"]): name for index, name in split["assignments"].items()}
    if not any(name == "train" for name in assignments.values()) or not any(name == "validation" for name in assignments.values()):
        raise ValueError("The available image groups could not form both train and validation subsets")
    selection_doc = {**available_doc, "images": [row for row in available if assignments.get(int(row["id"])) in {"train", "validation"}]}
    selected_ids = {int(row["id"]) for row in selection_doc["images"]}
    selection_doc["annotations"] = [a for a in available_doc["annotations"] if int(a["image_id"]) in selected_ids]
    converted = convert_document(selection_doc, lookup, output / "selection", assignments, "train")
    names = {int(k): v for k, v in converted["names"].items()}
    yaml = "path: selection\ntrain: train/images\nval: validation/images\nnames:\n"
    yaml += "".join(f"  {i}: {json.dumps(names[i], ensure_ascii=False)}\n" for i in sorted(names))
    output.mkdir(parents=True, exist_ok=True)
    (output / "selection.yaml").write_text(yaml, encoding="utf-8")
    source_manifest_path = root / "source_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8")) if source_manifest_path.exists() else {}
    manifest = {
        "dataset": "iisc-aim/UVH-26", "source_revision": source_manifest.get("revision"),
        "annotation_variant": "MV only", "label_type": "vehicle_detection_yolo_boxes", "crash_labels_created": False,
        "exploratory_only": True, "official_test_used": False,
        "source_download_status": source_manifest.get("download_status", "unknown"),
        "official_train_images_available": len(available), "split_counts": split["split_counts"],
        "split_grouping": "exact image SHA-256; duplicate files stay together",
        "camera_disjoint": False, "camera_disjoint_reason": "UVH-26 image records do not supply camera IDs.",
        "test_partition": None, "converted": converted,
        "warning": "Incomplete-download development run. This validation subset is exploratory, not the untouched official benchmark or a final test.",
    }
    (output / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def convert_document(document, image_lookup, output_root, split_assignments, split_prefix):
    """Write YOLO labels and hard-link source images, falling back to copy."""
    import cv2

    audit = validate_coco(document)
    root = Path(output_root)
    missing = []
    invalid_images = []
    converted = 0
    for image_id, image in audit["images"].items():
        source = image_lookup.get(str(image["file_name"]))
        if not source or not Path(source).is_file():
            missing.append(str(image["file_name"]))
            continue
        decoded = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if decoded is None:
            invalid_images.append({"file_name": str(image["file_name"]), "reason": "decode_failed"})
            continue
        height, width = decoded.shape[:2]
        if width != int(image["width"]) or height != int(image["height"]):
            invalid_images.append({"file_name": str(image["file_name"]), "reason": "dimension_mismatch",
                                   "expected": [int(image["width"]), int(image["height"])],
                                   "actual": [width, height]})
            continue
        split = split_assignments.get(image_id, split_prefix)
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        target_image = image_dir / Path(image["file_name"]).name
        if not target_image.exists():
            try:
                os.link(source, target_image)
            except OSError:
                shutil.copy2(source, target_image)
        label_path = label_dir / (Path(image["file_name"]).stem + ".txt")
        label_path.write_text("\n".join(audit["labels"].get(image_id, [])), encoding="utf-8")
        converted += 1
    return {"converted_images": converted, "missing_images": missing,
            "invalid_images": invalid_images,
            "valid_boxes": sum(audit["valid_box_counts"].values()),
            "valid_box_counts": audit["valid_box_counts"],
            "invalid_annotation_count": len(audit["invalid_annotations"]),
            "invalid_annotations": audit["invalid_annotations"],
            "class_map": {str(k): v for k, v in audit["class_map"].items()}, "names": audit["names"]}


def prepare_uvh26(root, output, variant="MV", seed=42, validation_fraction=0.12):
    root, output = Path(root).resolve(), Path(output).resolve()
    if variant != "MV":
        raise ValueError("Use one annotation variant at a time; this baseline deliberately selects UVH-26-MV")
    manifest_path = root / "source_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("download_status") != "complete":
            raise ValueError("UVH-26 download is incomplete; finish and verify the source manifest before conversion")
    train_json = root / "snapshot/UVH-26-Train/UVH-26-MV-Train.json"
    val_json = root / "snapshot/UVH-26-Val/UVH-26-MV-Val.json"
    if not train_json.is_file() or not val_json.is_file():
        raise ValueError(f"UVH-26 MV train and official validation JSON are required; missing {train_json if not train_json.is_file() else val_json}")
    train_doc = json.loads(train_json.read_text(encoding="utf-8"))
    val_doc = json.loads(val_json.read_text(encoding="utf-8"))
    if train_doc["categories"] != val_doc["categories"]:
        raise ValueError("Train and official validation class schemas differ")

    search_roots = [root / "snapshot/UVH-26-Train/data", root / "snapshot/UVH-26-Val/data"]
    lookup = {}
    for search_root in search_roots:
        if search_root.is_dir():
            for path in search_root.rglob("*"):
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    lookup[path.name] = path
    train_audit, val_audit = validate_coco(train_doc), validate_coco(val_doc)
    if any(str(image["file_name"]) not in lookup for image in [*train_audit["images"].values(), *val_audit["images"].values()]):
        raise ValueError("One or more UVH-26 image files are missing; wait for the download to finish and retry")

    # Hash identical images before any derived split, and group them together.
    official_val_hashes = {_sha(lookup[str(image["file_name"])]) for image in val_audit["images"].values()}
    train_hash = {image_id: _sha(lookup[str(image["file_name"])]) for image_id, image in train_audit["images"].items()}
    cross_split_duplicate_ids = {image_id for image_id, digest in train_hash.items() if digest in official_val_hashes}
    group_rows = [{"source_video_id": digest, "image_id": image_id}
                  for image_id, digest in train_hash.items() if image_id not in cross_split_duplicate_ids]
    grouped = grouped_split(group_rows, seed=seed,
                            fractions=(1-validation_fraction, validation_fraction, 0.0))
    image_split = {int(group_rows[i]["image_id"]): name for i, name in grouped["assignments"].items()}

    output.mkdir(parents=True, exist_ok=True)
    selection_root, official_root = output / "selection", output / "official_benchmark"
    selection_train = {i: "train" for i, split in image_split.items() if split == "train"}
    selection_val = {i: "validation" for i, split in image_split.items() if split == "validation"}
    selection_map = {**selection_train, **selection_val}
    selection_doc = {**train_doc, "images": [row for row in train_doc["images"] if int(row["id"]) in selection_map]}
    selection_ids = {int(row["id"]) for row in selection_doc["images"]}
    selection_doc["annotations"] = [a for a in train_doc["annotations"] if int(a["image_id"]) in selection_ids]
    sel = convert_document(selection_doc, lookup, selection_root, selection_map, "train")

    clean_train_doc = {**train_doc, "images": [row for row in train_doc["images"] if int(row["id"]) not in cross_split_duplicate_ids]}
    clean_ids = {int(row["id"]) for row in clean_train_doc["images"]}
    clean_train_doc["annotations"] = [a for a in train_doc["annotations"] if int(a["image_id"]) in clean_ids]
    full_train = {int(row["id"]): "train" for row in clean_train_doc["images"]}
    train_result = convert_document(clean_train_doc, lookup, official_root, full_train, "train")
    official_val_map = {int(row["id"]): "official_val" for row in val_doc["images"]}
    val_result = convert_document(val_doc, lookup, official_root, official_val_map, "official_val")
    names = {int(k): v for k, v in sel["names"].items()}
    yaml = "path: selection\ntrain: train/images\nval: validation/images\nnames:\n"
    yaml += "".join(f"  {i}: {json.dumps(names[i], ensure_ascii=False)}\n" for i in sorted(names))
    (output / "selection.yaml").write_text(yaml, encoding="utf-8")
    yaml = "path: official_benchmark\ntrain: train/images\nval: official_val/images\nnames:\n"
    yaml += "".join(f"  {i}: {json.dumps(names[i], ensure_ascii=False)}\n" for i in sorted(names))
    (output / "official_benchmark.yaml").write_text(yaml, encoding="utf-8")

    split_manifest = {
        "dataset": "iisc-aim/UVH-26", "annotation_variant": "MV (majority vote only; STAPLE is not mixed)",
        "class_id_mapping_source_to_yolo": sel["class_map"], "class_names": names,
        "official_split": {"train_json": train_json.relative_to(root).as_posix(), "validation_json": val_json.relative_to(root).as_posix(),
                           "official_validation_reserved_for_benchmark": True},
        "selection_split": {"seed": seed, "group_key": "exact image SHA-256", "counts": grouped["split_counts"],
                            "camera_disjoint": False, "camera_disjoint_reason": "UVH-26 COCO image records expose no camera ID."},
        "cross_official_split_exact_duplicates_excluded_from_train": len(cross_split_duplicate_ids),
        "train": train_result, "selection": sel, "official_validation": val_result,
    }
    samples_dir = output / "annotation_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    import cv2
    import random
    rng = random.Random(seed)
    sample_paths = sorted((selection_root / "train/images").glob("*"))
    for source in rng.sample(sample_paths, min(12, len(sample_paths))):
        canvas = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if canvas is None:
            continue
        label_path = selection_root / "train/labels" / f"{source.stem}.txt"
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            class_id, xc, yc, wn, hn = int(parts[0]), *map(float, parts[1:])
            h, w = canvas.shape[:2]
            x1, y1 = int((xc - wn / 2) * w), int((yc - hn / 2) * h)
            x2, y2 = int((xc + wn / 2) * w), int((yc + hn / 2) * h)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(canvas, names[class_id], (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
        cv2.imwrite(str(samples_dir / source.name), canvas)
    split_manifest["rendered_annotation_sample_count"] = len(list(samples_dir.glob("*.png")))
    (output / "split_manifest.json").write_text(json.dumps(split_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return split_manifest

