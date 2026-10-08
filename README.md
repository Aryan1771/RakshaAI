# RakshaAI

Research tooling for Indian road-accident detection, covering traceable footage collection, human annotation, vehicle detection, and temporal-model experiments.

## Project scope

| Component | Purpose | Documentation |
| --- | --- | --- |
| Phase 1 collector | Discover candidates, track permissions, review footage, deduplicate, and annotate crash events | [Collector guide](data%20collector/README.md) |
| Dataset workspace | Keep local footage and provenance separate from source code | [Dataset layout](data%20collector/dataset/README.md) |
| Phase 2 experiments | YOLOv8 vehicle detection, OpenCV preprocessing, R3D-18 experiments, evaluation, and streaming | [Training guide](data%20collector/models/README.md) |
| Validation record | Recorded checks and known limits | [Validation](data%20collector/VALIDATION.md) |

The repository contains research and development tools. It does not establish validated real-world accident-detection performance or provide an operational emergency-dispatch service. A checkpoint or metric exists only after the corresponding experiment has completed.

## Getting started

Requirements: Python 3.11+, FFmpeg, and ffprobe. Run from a fresh checkout:

```bash
git clone https://github.com/Aryan1771/RakshaAI.git
cd "RakshaAI/data collector"
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -e '.[test]'
python -m raksha_collector doctor
```

Follow the [collector guide](data%20collector/README.md) to configure sources and review candidates. Training has additional dependencies and hardware requirements; consult the Phase 2 instructions before installing or running it.

## Evaluation principles

Vehicle detection and accident detection are separate tasks. UVH-26 provides vehicle images and bounding boxes, not temporal crash labels. Keep it separate from human-reviewed accident footage. Report the dataset revision, split strategy, precision, recall, F1 score, and operating threshold with each experiment; do not describe exploratory validation as final test performance.

## Data and licensing

Keep footage, credentials, caches, and training outputs out of Git. Record provenance, attribution, and the applicable permissions for each source. The [GNU GPL v3 license](LICENSE) covers repository code; it does not grant rights to third-party footage or datasets.
