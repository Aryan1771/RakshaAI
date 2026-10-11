import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from raksha_collector import datasets


def test_uvh26_preview_does_not_download(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    result = datasets.import_uvh26(tmp_path, download=False)
    assert result["dataset"] == "iisc-aim/UVH-26"
    assert result["downloaded"] is False
    assert Path(result["path"]) == tmp_path / "UVH-26"
    assert not (tmp_path / "UVH-26/snapshot").exists()


def test_uvh26_download_writes_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: "hf.exe")

    def fake_run(args, **kwargs):
        assert args[:4] == ["hf.exe", "download", "iisc-aim/UVH-26", "--repo-type"]
        target = Path(args[args.index("--local-dir") + 1])
        (target / "UVH-26-Train").mkdir(parents=True)
        (target / "UVH-26-Train" / "labels.json").write_text("{}")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(datasets.subprocess, "run", fake_run)
    result = datasets.import_uvh26(tmp_path, revision="test-revision")
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert result["files"] == 1
    assert manifest["revision"] == "test-revision"
    assert manifest["media_type"] == "images_with_COCO_vehicle_annotations"
    assert manifest["crash_video_labels"] is False
    assert manifest["files"] == [{"path": "UVH-26-Train/labels.json", "bytes": 2}]


def test_uvh26_invalid_revision_and_download_failure(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="revision"):
        datasets.import_uvh26(tmp_path, revision="bad revision")
    monkeypatch.setattr("shutil.which", lambda _: "hf.exe")
    monkeypatch.setattr(datasets.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1))
    with pytest.raises(ValueError, match="download failed"):
        datasets.import_uvh26(tmp_path)
