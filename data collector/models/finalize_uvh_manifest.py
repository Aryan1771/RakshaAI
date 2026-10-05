from datetime import datetime, timezone
import json
from pathlib import Path

from huggingface_hub import HfApi


root = Path(__file__).resolve().parent.parent
dataset_root = root / "dataset" / "UVH-26"
target = dataset_root / "snapshot"
previous_manifest = dataset_root / "source_manifest.json"
revision = json.loads(previous_manifest.read_text(encoding="utf-8")).get("revision", "main") if previous_manifest.exists() else "main"
repo = HfApi().dataset_info("iisc-aim/UVH-26", revision=revision, files_metadata=True)
expected = sorted(repo.siblings, key=lambda item: item.rfilename)
files = []
missing = []
complete_count = 0
for item in expected:
    path = target / Path(item.rfilename)
    expected_size = getattr(item, "size", None)
    if path.is_file():
        size = path.stat().st_size
        files.append({"path": item.rfilename, "bytes": size})
        if expected_size is None or size == expected_size:
            complete_count += 1
        else:
            missing.append({"path": item.rfilename, "expected_bytes": expected_size, "actual_bytes": size})
    else:
        missing.append({"path": item.rfilename, "expected_bytes": expected_size, "actual_bytes": None})

expected_bytes = sum(getattr(item, "size", 0) or 0 for item in expected)
actual_bytes = sum(item["bytes"] for item in files)
manifest = {
    "dataset": "iisc-aim/UVH-26",
    "source_url": "https://huggingface.co/datasets/iisc-aim/UVH-26",
    "revision": repo.sha,
    "license": "CC-BY-4.0 (as declared by the dataset card; verify terms and attribution before reuse)",
    "downloaded_at": datetime.now(timezone.utc).isoformat(),
    "download_status": "complete" if not missing else "incomplete",
    "expected_file_count": len(expected),
    "expected_bytes": expected_bytes,
    "downloaded_expected_file_count": complete_count,
    "downloaded_bytes": actual_bytes,
    "missing_or_size_mismatch_count": len(missing),
    "missing_or_size_mismatched_files": missing,
    "media_type": "images_with_COCO_vehicle_annotations",
    "crash_video_labels": False,
    "files": files,
    "attribution": "Sharma et al., UVH-26 v1.0, IISc, arXiv:2511.02563. Attribute dataset to its authors and link the license.",
    "note": "India/Bengaluru vehicle-detection imagery. This image dataset is separate from crash-video annotations and must not be treated as crash footage.",
}
manifest_path = dataset_root / "source_manifest.json"
manifest_path.parent.mkdir(parents=True, exist_ok=True)
temporary = manifest_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
temporary.replace(manifest_path)
print(json.dumps({key: manifest[key] for key in (
    "download_status", "revision", "expected_file_count", "expected_bytes",
    "downloaded_expected_file_count", "downloaded_bytes",
    "missing_or_size_mismatch_count",
)}, indent=2))
