"""Normalize Phase 1 labels without upgrading assistant drafts to ground truth."""
import json
from pathlib import Path

from raksha_collector.storage import Store

from .splits import grouped_split
from .video_data import eligible_review_row


def phase1_rows(phase1_root):
    root = Path(phase1_root)
    store = Store(root / "collection.sqlite3")
    rows = []
    for serial, payload in store.db.execute("SELECT serial,payload FROM annotations ORDER BY serial"):
        annotation = json.loads(payload)
        record = store.get(annotation["record_id"])
        clip_path = annotation.get("local_path")
        relative_event_start = None
        relative_event_end = None
        if annotation.get("impact_start_source") is not None:
            clip_start = float(annotation.get("clip_start_source", 0))
            relative_event_start = float(annotation["impact_start_source"]) - clip_start
            relative_event_end = float(annotation["impact_end_source"]) - clip_start
        rows.append({
            "record_id": record["id"],
            "source_video_id": record.get("source_video_id"),
            "source_recording_id": record["id"],
            "incident_id": annotation.get("incident_id") or record.get("incident_id"),
            "clip_path": str(Path(clip_path).resolve()) if clip_path else None,
            "duration_seconds": annotation.get("video", {}).get("duration"),
            "event_start_seconds": relative_event_start,
            "event_end_seconds": relative_event_end,
            "footage_type": record.get("human", {}).get("footage_type", "unknown"),
            "clip_label": annotation.get("class_name"),
            "annotation_confidence": annotation.get("annotation_confidence"),
            "annotation_review_status": annotation.get("annotation_review_status", "needs_human_review"),
            "annotation_origin": annotation.get("origin", "unknown"),
            "review_status": "confirmed" if annotation.get("annotation_review_status") == "confirmed" else "pending",
            "decision": record.get("human", {}).get("decision", "pending"),
            "country": record.get("human", {}).get("country", "unknown"),
            "location_evidence": record.get("human", {}).get("location_evidence", ""),
            "permission_status": record.get("permission", {}).get("status", "unknown"),
            "permission_evidence": record.get("permission", {}).get("evidence", ""),
            "camera_id": record.get("human", {}).get("camera_id"),
            "day_night": record.get("human", {}).get("lighting"),
            "weather": record.get("human", {}).get("weather"),
            "occlusion": record.get("human", {}).get("occlusion"),
            "segments": annotation.get("segments", []),
            "serial": serial,
        })
    store.db.close()
    return rows


def split_reviewed_rows(rows, seed=42, fractions=(0.8, 0.1, 0.1)):
    eligible = [row for row in rows if eligible_review_row(row)]
    split = grouped_split(eligible, seed=seed, fractions=fractions)
    assigned = []
    for index, row in enumerate(eligible):
        if index in split["assignments"]:
            assigned.append({**row, "split": split["assignments"][index]})
    excluded = len(rows) - len(eligible)
    return assigned, {"eligible_rows": len(eligible), "excluded_rows": excluded,
                      "unassigned_missing_incident_or_source": len(split["unassigned"]),
                      "group_count": split["group_count"], "split_counts": split["split_counts"],
                      "seed": seed, "fractions": fractions}


def write_video_manifest(rows, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return str(destination.resolve())

