"""Human crash labels with pre-impact context and explicit temporal supervision."""
import csv
import json
import math
import subprocess
from pathlib import Path
from .media import binary, probe, checksum
from .storage import digest, now
from .exporting import eligible


def render_clip(source, target, start, end):
    partial = target.with_suffix(".partial.mp4")
    subprocess.run([binary("ffmpeg"), "-v", "error", "-y", "-i", str(source), "-ss", str(start),
                    "-t", str(end - start), "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264",
                    "-crf", "18", "-c:a", "aac", "-movflags", "+faststart", str(partial)],
                   capture_output=True, check=True, timeout=600)
    info = probe(partial)
    partial.replace(target)
    return info


def annotate_crash(store, rid, root, vehicles, impact_start, impact_end, reviewer,
                   lead_in=5.0, aftermath=3.0, normal_before=False, notes="", renderer=None, origin="human"):
    """Create crash_N(serial).mp4. Source timestamps use seconds, never frame guesses.

    Pre-impact context is 'unknown' unless a reviewer confirms normal_before.
    Post-impact context is 'aftermath', never automatically a negative example.
    """
    r = store.get(rid)
    from .review import india_location_verified
    if not india_location_verified(r):
        raise ValueError("Crash annotation requires evidence that the footage was recorded in India")
    d = r["download"]
    duration = d.get("duration", 0)
    if not isinstance(vehicles, int) or isinstance(vehicles, bool) or vehicles < 1:
        raise ValueError("vehicles must be a confirmed positive integer (vehicles involved, not all visible vehicles)")
    if not reviewer.strip():
        raise ValueError("Reviewer is required")
    if origin not in {"human", "assistant_visual_review"}:
        raise ValueError("Invalid annotation origin")
    if not all(isinstance(n, (float, int)) and math.isfinite(n) for n in (impact_start, impact_end, lead_in, aftermath)):
        raise ValueError("Timestamps and context must be finite numbers")
    if not d.get("local_path") or not Path(d["local_path"]).is_file() or not 0 <= impact_start < impact_end <= duration:
        raise ValueError("Impact times must lie inside a validated local clip")
    if lead_in < 0 or aftermath < 0:
        raise ValueError("Context durations cannot be negative")
    impact_start, impact_end, lead_in, aftermath = map(float, (impact_start, impact_end, lead_in, aftermath))
    start, end = max(0, impact_start - lead_in), min(duration, impact_end + aftermath)
    source_offset = float(r.get("source_segment", {}).get("start_seconds", 0))
    if normal_before and start == impact_start:
        raise ValueError("No pre-crash interval exists to confirm as normal")
    source_hash = checksum(d["local_path"])
    if source_hash != d.get("checksum"):
        raise ValueError("Source changed since import/download; re-inspect it before annotating")
    spec = {"record_id": rid, "source_checksum": source_hash, "vehicles_crashed": vehicles,
            "impact_start_source": impact_start + source_offset, "impact_end_source": impact_end + source_offset,
            "clip_start_source": start + source_offset, "clip_end_source": end + source_offset,
            "source_segment_offset": source_offset, "normal_before_confirmed": normal_before,
            "origin": origin}
    key = digest(json.dumps(spec, sort_keys=True))
    with store.db:
        store.db.execute("BEGIN IMMEDIATE")
        found = store.db.execute("SELECT serial,payload FROM annotations WHERE annotation_key=?", (key,)).fetchone()
        if found is None:
            store.db.execute("INSERT INTO annotations(annotation_key,record_id,payload) VALUES (?,?,?)",
                             (key, rid, json.dumps({**spec, "status": "pending"})))
            found = store.db.execute("SELECT serial,payload FROM annotations WHERE annotation_key=?", (key,)).fetchone()
    serial, payload = found
    old = json.loads(payload)
    if old.get("status") == "complete" and Path(old["local_path"]).is_file():
        if checksum(old["local_path"]) != old["checksum"]:
            raise ValueError("Annotated clip changed on disk; preserve and review it before replacing")
        return old
    label = f"crash_{vehicles}({serial})"
    folder = Path(root).resolve() / "annotated"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{label}.mp4"
    if target.exists():
        # Reserved serial belongs to this exact spec. Recover a completed render
        # after interruption before metadata persistence.
        info = probe(target)
    else:
        info = (renderer or render_clip)(d["local_path"], target, start, end)
    impact_a, impact_b = impact_start - start, impact_end - start
    segments = []
    if impact_a > 0:
        segments.append({"start": 0, "end": impact_a, "label": "normal" if normal_before else "unknown", "target": 0 if normal_before else None})
    segments.append({"start": impact_a, "end": impact_b, "label": "crash", "target": 1})
    if end > impact_end:
        segments.append({"start": impact_b, "end": end - start, "label": "aftermath", "target": None})
    annotation = {**spec, "serial": serial, "label": label, "class_name": "crash", "class_target": 1,
                  "status": "complete", "local_path": str(target), "checksum": checksum(target),
                  "reviewer": reviewer, "notes": notes, "annotated_at": now(), "origin": origin,
                  "annotation_review_status": "confirmed" if origin == "human" else "needs_human_review",
                  "incident_id": r["incident_id"], "segments": segments, "video": info,
                  "training_eligible": eligible(r) and origin == "human", "training_eligibility_note": "Eligibility must be recomputed at export; annotation does not establish permission or location."}
    with store.db:
        store.db.execute("UPDATE annotations SET payload=? WHERE serial=?", (json.dumps(annotation), serial))
    (folder / f"{label}.json").write_text(json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8")
    r["transformations"].append({"kind": "crash_annotation", "serial": serial, "label": label,
                                 "path": str(target), "source_checksum": source_hash, "start": start, "end": end, "created_at": now()})
    store.save(r, "crash_annotation")
    return annotation


def export_annotations(store, directory, dataset_only=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows, segments = [], []
    for serial, payload in store.db.execute("SELECT serial,payload FROM annotations ORDER BY serial"):
        annotation = json.loads(payload)
        if annotation.get("status") != "complete":
            continue
        record = store.get(annotation["record_id"])
        annotation["incident_id"] = record["incident_id"]
        annotation["training_eligible"] = eligible(record) and annotation.get("annotation_review_status") == "confirmed"
        annotation["duplicate_group"] = record["duplicate_group"]
        if dataset_only and not annotation["training_eligible"]:
            continue
        rows.append(annotation)
        for segment in annotation["segments"]:
            segments.append({"record_id": record["id"], "incident_id": record["incident_id"],
                             "clip_label": annotation["label"], "video_path": annotation["local_path"],
                             "vehicles_crashed": annotation["vehicles_crashed"], "reviewer": annotation["reviewer"], **segment})
    (directory / "annotations.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    with (directory / "temporal_labels.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["record_id", "incident_id", "clip_label", "video_path", "vehicles_crashed", "reviewer", "start", "end", "label", "target"])
        writer.writeheader()
        writer.writerows(segments)
    return {"annotations": len(rows), "temporal_segments": len(segments), "directory": str(directory.resolve())}
