# Dataset workspace

Place source datasets beneath this directory so the project can move as a unit.

- `UVH-26/` is the official IISc traffic-image dataset. Its annotations are vehicle bounding boxes for YOLO, not crash labels. The Hugging Face source is pinned in `source_manifest.json`; its download may be incomplete while `download_status` says so.
- `news_cctv/` contains Phase 1 provenance, review records, extracted segments and annotation drafts. Only human-reviewed, permitted video labels are candidates for crash-model training.
- `existing_clips/` is the optional location for separately audited local clips.

Large media and generated annotations are intentionally excluded from Git. Paths in the Phase 2 configuration are relative to the project root.
