# Validation record

Validated on 2026-10-05. The project keeps source datasets beneath `dataset/` and the training code/configs beneath `models/`. Phase 1 review material is under `dataset/news_cctv/`. Paths in `models/configs/phase2.full.json` are project-relative.

The Phase 1 audit found 7 readable local videos, 1 exact duplicate group, 6 clips without labels, and 1 assistant-drafted crash label still marked `needs_human_review`. It is excluded from training. The approximately 3,000 additional clips were not found at `dataset/existing_clips/`; there are no independent reviewed crash incidents or human-confirmed normal hours. Camera/day/night/weather/occlusion metadata are unrecorded. `ffprobe` and FFmpeg are absent, so the audit uses OpenCV. News archive candidates are not verified video labels or permission grants.

The user canceled the full-data YOLO run before it started. The dataset downloader, preparation watcher, training watcher, and CUDA-wheel transfer were stopped. The earlier exploratory YOLO run was CPU-only and is not a full training result. The request to delete all local copies and UVH-derived artifacts was blocked by the environment's automatic command review; the partial local dataset and ignored exploratory artifacts remain on disk and were not staged or pushed. No full-data YOLO training or test evaluation was run.

Hardware: Intel Core i7-14650HX (16 cores / 24 threads), NVIDIA RTX 4060 Laptop GPU (8,188 MiB VRAM, driver 617.14), Python 3.13.14, about 400 GiB free disk at the latest check. The existing `.venv` contains PyTorch 2.10.0+cpu and torchvision 0.25.0+cpu. The system recognizes the RTX 4060, but PyTorch CUDA was not installed or verified in the training environment. The full YOLO run was canceled and did not start.

Checks completed:

- `python -m compileall -q models.raksha_training raksha_collector`: passed.
- `python -m pytest -q`: 60 passed.
- `python -m models.raksha_training smoke --config models/configs/phase2.smoke.json`: passed on CPU, using synthetic videos only. It exercised video decode, an R3D-18 optimizer/validation step, one synthetic YOLO epoch, and combined inference. Its metrics do not measure accident performance.
- Synthetic-only smoke testing passed earlier; it is not a real accident-model evaluation.

An exploratory YOLOv8-S run completed on CPU: one epoch at 320 px, batch 4, using 10% of a partial training selection and its validation split. Its low metrics were a pipeline check only and are not suitable for deployment or final conclusions. No untouched test evaluation has been run. No full-data training run is active.

R3D-18 training/evaluation remains blocked because no human-reviewed, permitted crash/normal video set is available. No crash recall, event delay or production claim is made.

Run tests from the `data collector` directory:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
