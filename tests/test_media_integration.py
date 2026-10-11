"""Real FFmpeg checks on synthetic test patterns, never dataset crash examples."""
import shutil
import subprocess
from pathlib import Path
import pytest
from dotenv import load_dotenv
from raksha_collector import media, annotation
from raksha_collector.storage import Store


def test_real_media_import_annotation_and_dedup(tmp_path):
    load_dotenv(Path.cwd() / ".env")
    try:
        ffmpeg = media.binary("ffmpeg")
        media.binary("ffprobe")
    except ValueError:
        pytest.skip("Install FFmpeg/ffprobe to run real media validation")
    source = tmp_path / "input"
    source.mkdir()
    original = source / "synthetic.mp4"
    subprocess.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=10:duration=8", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(original)], capture_output=True, check=True)
    shutil.copy2(original, source / "exact.mp4")
    subprocess.run([ffmpeg, "-v", "error", "-i", str(original), "-vf", "drawbox=x=0:y=0:w=80:h=15:color=white:t=fill", "-c:v", "libx264", "-crf", "32", str(source / "overlay.mp4")], capture_output=True, check=True)
    store = Store(tmp_path / "data.sqlite3")
    assert media.import_local(store, source)["new_records"] == 3
    assert media.import_local(store, source)["new_records"] == 0
    records = store.records()
    assert all(r["download"]["duration"] == 8 for r in records)
    r = records[0]
    from raksha_collector import review
    review.annotate(store, r["id"], {"country": "India", "location_evidence": "Synthetic fixture marked Indian for unit testing"}, "Synthetic test")
    r = store.get(r["id"])
    before = media.checksum(r["download"]["local_path"])
    sheet = media.contact_sheet(store, r, tmp_path)
    assert Path(sheet).is_file()
    result = annotation.annotate_crash(store, r["id"], tmp_path, 2, 4, 5, "Synthetic test", lead_in=2, aftermath=1)
    assert abs(media.probe(result["local_path"])["duration"] - 4) < 0.2
    assert media.checksum(r["download"]["local_path"]) == before
    dedup = media.deduplicate(store, threshold=12)
    assert dedup["groups"] == 1
    assert any(p["kind"] == "near" for p in store.pairs())
    assert any(p["kind"] == "exact" for p in store.pairs())
