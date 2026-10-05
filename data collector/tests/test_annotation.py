import json
from pathlib import Path
import pytest
from raksha_collector.annotation import annotate_crash, export_annotations
from raksha_collector.storage import Store, new_record
from raksha_collector.media import checksum


@pytest.fixture
def source(tmp_path):
    store = Store(tmp_path / "annotations.sqlite3")
    path = tmp_path / "original.mp4"
    path.write_bytes(b"synthetic test fixture")
    record = new_record("", "local", source_id=str(path))
    record["human"].update(country="India", location_evidence="Synthetic fixture marked Indian for unit testing")
    record["download"] = {"status": "imported", "local_path": str(path), "duration": 20, "checksum": checksum(path)}
    store.add(record)
    return store, record["id"], tmp_path


def fake_render(source, target, start, end):
    target.write_bytes(b"derived fixture")
    return {"duration": end - start, "fps": 25}


def test_exact_filename_and_temporal_labels(source):
    store, rid, root = source
    result = annotate_crash(store, rid, root, 2, 8, 10, "Aryan", normal_before=True, renderer=fake_render)
    assert Path(result["local_path"]).name == "crash_2(1).mp4"
    assert result["clip_start_source"] == 3
    assert result["clip_end_source"] == 13
    assert result["segments"] == [
        {"start": 0, "end": 5, "label": "normal", "target": 0},
        {"start": 5, "end": 7, "label": "crash", "target": 1},
        {"start": 7, "end": 10, "label": "aftermath", "target": None}]
    assert not result["training_eligible"]
    assert (root / "original.mp4").read_bytes() == b"synthetic test fixture"


def test_annotation_idempotence_and_global_serial(source):
    store, rid, root = source
    first = annotate_crash(store, rid, root, 2, 8, 10, "Reviewer", renderer=fake_render)
    again = annotate_crash(store, rid, root, 2, 8, 10, "Reviewer", renderer=lambda *args: pytest.fail("Must not re-render"))
    assert again["label"] == first["label"]
    next_label = annotate_crash(store, rid, root, 1, 14, 16, "Reviewer", renderer=fake_render)
    assert next_label["label"] == "crash_1(2)"
    assert store.db.execute("SELECT COUNT(*) FROM annotations").fetchone()[0] == 2


def test_unconfirmed_before_not_normal(source):
    store, rid, root = source
    result = annotate_crash(store, rid, root, 3, 2, 5, "Reviewer", renderer=fake_render)
    assert result["clip_start_source"] == 0
    assert result["segments"][0]["label"] == "unknown"
    assert result["segments"][0]["target"] is None


def test_annotation_times_map_back_from_downloaded_source_segment(source):
    store, rid, root = source
    record = store.get(rid)
    record["source_segment"] = {"start_seconds": 40, "end_seconds": 60}
    store.save(record)
    result = annotate_crash(store, rid, root, 2, 8, 10, "Reviewer", renderer=fake_render)
    assert result["impact_start_source"] == 48
    assert result["impact_end_source"] == 50
    assert result["clip_start_source"] == 43
    assert result["clip_end_source"] == 53
    assert result["source_segment_offset"] == 40


@pytest.mark.parametrize("vehicles,start,end", [(0, 1, 3), (1, -1, 3), (2, 4, 3), (2, 18, 22), (2, float('nan'), 3)])
def test_invalid_annotation(source, vehicles, start, end):
    store, rid, root = source
    with pytest.raises(ValueError):
        annotate_crash(store, rid, root, vehicles, start, end, "Reviewer", renderer=fake_render)
    assert store.db.execute("SELECT COUNT(*) FROM annotations").fetchone()[0] == 0


def test_export_excludes_unpermitted_and_refreshes_incident(source):
    store, rid, root = source
    annotate_crash(store, rid, root, 2, 8, 10, "Reviewer", renderer=fake_render)
    assert export_annotations(store, root / "exports", True)["annotations"] == 0
    record = store.get(rid)
    record["human"].update(review_status="reviewed", decision="include", country="India", location_evidence="Verified")
    record["permission"] = {"status": "permitted", "evidence": "Owner grant"}
    record["incident_id"] = "event-1"
    store.save(record)
    assert export_annotations(store, root / "exports", True)["annotations"] == 1
    row = json.loads((root / "exports/annotations.jsonl").read_text())
    assert row["incident_id"] == "event-1"
    assert row["training_eligible"]


def test_visual_review_draft_never_enters_strict_export(source):
    store, rid, root = source
    draft = annotate_crash(store, rid, root, 2, 8, 10, "Assistant draft", renderer=fake_render, origin="assistant_visual_review")
    assert draft["annotation_review_status"] == "needs_human_review"
    assert not draft["training_eligible"]
    assert export_annotations(store, root / "draft-only", True)["annotations"] == 0


def test_changed_source_is_rejected(source):
    store, rid, root = source
    (root / "original.mp4").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        annotate_crash(store, rid, root, 2, 8, 10, "Reviewer", renderer=fake_render)
