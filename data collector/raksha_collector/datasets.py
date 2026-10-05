"""Optional, traceable imports of public Indian traffic image datasets."""
import json
import shutil
import subprocess
from pathlib import Path
from .storage import now


UVH_REPO = "iisc-aim/UVH-26"
UVH_URL = f"https://huggingface.co/datasets/{UVH_REPO}"


def import_uvh26(destination, revision="main", download=True, hf="hf"):
    """Download/extract UVH-26 files while preserving upstream splits and labels.

    `hf` is invoked as an argument vector, not through a shell. The operation
    requires the Hugging Face CLI and network access when download is enabled.
    """
    if not revision or any(c.isspace() for c in revision):
        raise ValueError("Provide a valid Hugging Face revision")
    # Store this source at the dataset root directly: <project>/dataset/UVH-26.
    root = Path(destination).resolve() / "UVH-26"
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "source_manifest.json"
    if not download:
        return {"dataset": UVH_REPO, "path": str(root), "downloaded": False,
                "note": "No files downloaded. Use --download to fetch the dataset."}

    executable = shutil.which(hf)
    if not executable:
        raise ValueError("Hugging Face CLI `hf` was not found; install it with `pip install -U huggingface_hub[cli]`")
    target = root / "snapshot"
    target.mkdir(exist_ok=True)
    result = subprocess.run(
        [executable, "download", UVH_REPO, "--repo-type", "dataset", "--revision", revision,
         "--local-dir", str(target)], capture_output=True, text=True, timeout=7200,
    )
    if result.returncode:
        # Avoid relaying credentials or signed URLs from CLI output.
        raise ValueError(f"Hugging Face download failed (exit {result.returncode}); check access, revision and network")
    files = sorted(p for p in target.rglob("*") if p.is_file())
    manifest = {
        "dataset": UVH_REPO,
        "source_url": UVH_URL,
        "revision": revision,
        "license": "CC-BY-4.0 (as declared by the dataset card; verify terms and attribution before reuse)",
        "downloaded_at": now(),
        "media_type": "images_with_COCO_vehicle_annotations",
        "crash_video_labels": False,
        "files": [{"path": p.relative_to(target).as_posix(), "bytes": p.stat().st_size} for p in files],
        "attribution": "Sharma et al., UVH-26 v1.0, IISc, arXiv:2511.02563. Attribute dataset to its authors and link the license.",
        "note": "India/Bengaluru vehicle-detection imagery. This image dataset is separate from crash-video annotations and must not be treated as crash footage.",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"dataset": UVH_REPO, "path": str(target), "manifest": str(manifest_path),
            "files": len(files), "bytes": sum(p.stat().st_size for p in files)}
