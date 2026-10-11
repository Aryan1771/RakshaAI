# RakshaAI data collector

Resumable footage discovery, permission tracking, local review and crash annotation. This is Phase 1; it does not train a model or dispatch emergency alerts. The repository's GPL-3.0 license covers this code, not collected footage.

## Setup (PowerShell)

```powershell
# From the repository root:
Set-Location 'data collector'
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[test]'
Copy-Item .env.example .env  # only if you do not already have .env
```

Python 3.11+ works. Install FFmpeg and ffprobe using a build linked from https://ffmpeg.org/download.html, then set `FFMPEG` and `FFPROBE` to their full executable paths in `.env`, or put both on PATH. Development tooling may be stored in the ignored `work/tools` directory; configure paths for your own checkout. Neither tool is committed. `.env` is ignored. Put `YOUTUBE_API_KEY` there only if using YouTube discovery.

For the following commands, define this helper once in your PowerShell session:

```powershell
function collect { & .\.venv\Scripts\python.exe -m raksha_collector @args }
collect doctor
```

Global options (`--db`, `--data`, `--config`) go **before** the command. Relative paths resolve from your current directory. Run one collector process per database/data folder.

## Discover and extract

```powershell
collect discover --limit 20
collect extract --limit 20
collect report
collect review
Start-Process 'data\review\index.html'
collect export --out data/exports
```

`config.example.json` enables two sources, NDTV and India Today. Each call accepts at most 100 candidates. The initial pilot used 25 unique candidate records, including records found before archive relevance filtering was tightened; retain those rejected/irrelevant candidates for audit. Repeated discovery of the same source IDs/URLs preserves human annotations. To start an independent pilot, pass `--db data/pilot2.sqlite3`.

Publisher adapters have separate host allowlists and URL patterns. Archive anchors must contain an accident/CCTV keyword and are ordered by relevance; exact seed pages are always tracked. Generic video archives are not exhaustive historical searches. Add specific archive pages to `archives` and increase `max_archive_pages` to crawl them. No unrestricted recursion or browser scraping is performed. Pages without static metadata stay `no_static_video`, for manual review; they are not automatically declared unsupported or accidents.

City/state and publication-date filters are **candidate text filters**, not location verification. News discovery registers page candidates; `extract` applies content/date filters and saves `machine.matches_filters`. No candidates are silently deleted. YouTube applies date/channel filters at the API and text filters in metadata. Publication and event dates stay separate.

```powershell
collect extract --limit 20 --city Hyderabad --published-after 2021-01-01
collect youtube --limit 20 --publisher NDTV --published-after 2021-01-01
```

English, Hindi and Telugu queries are in the config; add other regional queries there. YouTube channel IDs are resolved only from channel/handle/user links on official publisher websites, using the channels API. Legacy `/c/` links are not guessed. The API search-call budget defaults to 2 per invocation; current account quota remains authoritative. A missing key produces an actionable error, and no authenticated URL/API key is persisted.

### India news CCTV pilot

`config.news-cctv.json` configures the NDTV English/Hindi, Times of India and Aaj Tak archive pages supplied for this project. To keep the database, selected source segments, annotation outputs and optional dataset together locally, pass the same `dataset/news_cctv` path to both `--db` and `--data`:

```powershell
collect --config config.news-cctv.json --db dataset/news_cctv/collection.sqlite3 --data dataset/news_cctv discover --limit 100
collect --config config.news-cctv.json --db dataset/news_cctv/collection.sqlite3 --data dataset/news_cctv extract --limit 100
collect --config config.news-cctv.json --db dataset/news_cctv/collection.sqlite3 --data dataset/news_cctv report
```

Only archive metadata is collected during discovery/extraction. Before selecting any segment, verify India location, review its source times and media candidate, and record the intended research basis. Segment retrieval obeys robots rules and host access controls; blocked media is left alone. The pilot retains source candidates for audit, but media metadata is not a crash label or permission grant. Local footage under `dataset/` and generated output under `models/runs/` are git-ignored and should not be committed.

## Optional Indian vehicle-image dataset: UVH-26

[UVH-26](https://huggingface.co/datasets/iisc-aim/UVH-26), released by AIM @ IISc, contains Bengaluru Safe City traffic-camera **images** and COCO vehicle boxes (14 vehicle classes). It is a useful additional source for Indian vehicle detection and traffic-scene representation, alongside the collector's independently reviewed footage sources. It is not a crash-video dataset and supplies no temporal crash labels; keep it separate from `annotate-crash` outputs. The dataset card declares CC BY 4.0; preserve attribution and license information, and confirm the current terms before reuse. The upstream repository is about 90 GB and its dataset viewer reports a metadata/schema issue, so review available files before using it.

Install the Hugging Face CLI (`pip install -U huggingface_hub[cli]`), then preview the destination or explicitly download the upstream files. Pass `--data dataset` so UVH-26 is stored at the shared dataset root:

```powershell
collect --db dataset/news_cctv/collection.sqlite3 --data dataset dataset-uvh26
collect --db dataset/news_cctv/collection.sqlite3 --data dataset dataset-uvh26 --download
```

Files are stored under `dataset/UVH-26/snapshot/`, separate from crash video records, with a source manifest recording revision, file list, size, source and attribution. This dataset supplements the collection; it does not replace the Indian crash-footage sources. `annotate-uvh-available` exports currently downloaded majority-vote vehicle boxes as YOLO sidecars under `dataset/UVH-26/derived/MV_available/labels/`. It never creates crash labels or duplicates source images. `prepare-uvh-available` makes an explicitly partial, exact-duplicate-grouped development split under `dataset/UVH-26/derived/partial_experiment/`; this exploratory validation is not a final test. After the full download completes, use `prepare-uvh` for the complete selection and official-benchmark splits under `dataset/UVH-26/derived/prepared/`. Model checkpoints and run logs go to `models/runs/`.

## Permission and downloading

Discovery downloads page metadata only. This collector is India-only: before any media retrieval, confirm `country` is `India` and add specific `location_evidence` supporting where the event was recorded. Headline text, publisher identity and search filters alone are not location evidence. Public visibility, a publisher logo and robots access do not establish permission. Permission requests are never sent automatically.

```powershell
$id = 'PASTE_RECORD_ID_FROM_REPORT_OR_EXPORT'
collect permission $id requested --reviewer Aryan --evidence 'Permission request reference; no grant yet'
# Run the following only when the cited grant/license actually covers reuse:
collect permission $id permitted --reviewer Aryan --evidence 'Owner grant/license URL and scope'
collect download --id $id
```

For a project-research basis asserted by a human reviewer, record it separately (it is not represented as a rights-holder license or a legal determination). That basis permits only the segment-download command; whole-page/news-video downloading remains blocked for this status:

```powershell
collect permission $id research_basis --reviewer Aryan --evidence 'Reviewer asserts Copyright Act s.52 research basis for internal training; source CCTV boundaries reviewed; not a legal determination'
collect download-segment $id --start 122.4 --end 136.8
```

Times are seconds in the source page's video. Only direct media candidates parsed from the archive page are eligible; if there are several, inspect them and pass `--media-index N`. FFmpeg seeks to the selected range, transcodes that bounded interval and rejects outputs whose duration exceeds the requested interval by more than three seconds. Some streaming formats may transfer extra keyframe/segment data to produce a playable cut; the output file is only the selected segment, but this does not guarantee the server transmitted exactly those bytes. The record and sidecar keep the source article, selected media index, requested source time range, reviewer basis, checksum and tool method. Annotate crash times relative to the saved segment; annotation metadata maps them back to source-page time.

If a page has multiple media candidates, set `media_index` (zero based) using `review --annotations` after inspecting the source. Fresh media URLs are re-extracted immediately before downloading, so expiring signatures are not stored. Supported publisher sources use yt-dlp with `.part` continuation. Files are checked with ffprobe and SHA-256 before they count as downloaded. Original files are preserved. JSON sidecars retain provenance. A completed file is reused after interruption; records with persistent failures need explicit `download --retry`. Access-denied responses are not automatically retried. Transient requests have bounded exponential retries. Concurrency is one, with host pacing and robots rules.

## Import your existing clips

```powershell
collect import 'D:\YourExistingClips'
collect deduplicate
```

Import indexes files in place without moving them. Paths and hashes are recorded; permission, country and accident labels remain unknown. Keep the directory available and originals unchanged. Exact hashes propose duplicate pairs. Eight sampled-frame perceptual hashes (full frame and centre crop) propose near duplicates across re-encoding, small overlays and some crops. Proposals have false positives and false negatives, especially with substantial trimming/cropping; human review is required. All originals remain on disk. Pairwise comparison is quadratic: a few thousand clips are feasible, but larger collections need an indexed similarity search.

## Human review and incident grouping

Copy `annotation.example.json`, fill in India location evidence, footage type and timestamps, and apply it. Non-Indian or unverified footage cannot be crash-annotated:

```powershell
collect contact-sheet $id
collect review --id $id --annotations annotation.example.json --reviewer Aryan
collect review
collect pair-review RECORD_A RECORD_B confirmed
collect incident india-event-0001 RECORD_A RECORD_B --reviewer Aryan
```

The offline HTML index escapes source content and never autoplays footage. Contact sheets open only when clicked and may contain graphic images. `footage_type` accepts `unknown`, `cctv`, `dashcam`, `aftermath_only`, `near_miss`, `compilation`, `reenactment`, `animation`, `unrelated`. Group all edits and camera views of an accident into the same incident. Merging one existing incident into another moves its whole group.

```powershell
collect export --dataset-only --splits --out data/training_manifest
```

The training manifest requires reviewed inclusion, evidenced Indian location, a documented rights-holder/license basis or separately marked reviewer-asserted research basis, validated local footage, and an incident ID. The research-basis status records a human assertion and does not independently determine legal coverage. All incident views and unresolved/confirmed duplicate links stay in the same 80/10/10 split. Splits are deterministic for the current collection; freeze the export before training because new links/records can change groups. Pending proposals are conservatively kept together. Rejected pairs do not link splits.

## Crash annotation: `crash_<vehicles>(<serial>)`

The crash count is the **number of vehicles involved in the collision**, not every vehicle visible. A one-vehicle impact against a fixed object is `crash_1(...)`. Count an involved camera vehicle if it also collides. Do not guess obscured vehicles or label a near miss as a crash. A filename serial identifies a sample; it is **not a model class**.

Open the original in a video player, note the impact start/end in seconds, then run:

```powershell
collect annotate-crash $id --vehicles 2 --impact-start 12.4 --impact-end 14.1 --lead-in 5 --aftermath 3 --normal-before-confirmed --reviewer Aryan --notes 'Two vehicles visibly collide; pre-impact interval checked'
collect export-annotations --out data/annotation_exports
```

This creates, for example, `data/annotated/crash_2(1).mp4` plus its JSON annotation. The first five seconds are pre-crash context (clamped at source start), followed by the impact and three seconds of aftermath (clamped at source end). Originals remain intact. Serials are unique within the SQLite database, survive restarts and are reused for identical annotations. Keep the database with your dataset. Changed annotation parameters create another sample; exclude superseded samples before training.

Outputs provide:

- Clip-level `class_name=crash`, `class_target=1` and `vehicles_crashed` as a separate count target.
- Source and clip-relative impact timestamps, original hash, reviewer, origin, incident ID and transformation details.
- `temporal_labels.csv`: pre-impact `normal=0` **only with explicit confirmation**, impact `crash=1`, otherwise `unknown` or `aftermath` with an empty target. Intervals are half-open `[start,end)` in seconds.

Use these temporal labels to select training windows. Do not assign the clip-level positive label to every pre-crash frame; do not treat empty targets as zero. Aftermath is not automatically normal. This supplies supervised training labels; it does not update model weights or implement reinforcement-learning feedback.

Assistant-created examples use `--origin assistant_visual_review`, remain `needs_human_review`, and are excluded from strict dataset-only annotation exports. To produce human-confirmed labels, inspect the video and rerun `annotate-crash` with your reviewer identity and corrected times/counts (default origin is `human`). Do not use synthetic test fixtures as training crash examples.

```powershell
collect export-annotations --dataset-only --out data/training_annotations
collect trim $id --start 7.4 --end 17.1
```

The strict annotation export also requires the Indian dataset eligibility checks. Join its `record_id` to `records.jsonl` from the split export, so every derivative follows the source's split.

Only clips confirmed to have been recorded in India may be collected or crash-annotated. Unverified and foreign footage stays out of the dataset. Nexar was investigated but its current repository is gated, so no Nexar data or personal information was submitted/downloaded.

## Storage, validation and limitations

SQLite holds record JSON, audit snapshots, collection runs, duplicate pairs and annotation serials. `machine` fields are parser/ranking suggestions; `human` fields are reviewer input. Annotation `origin` distinguishes human and assistant labels. CSV and JSONL exports preserve both. The report separately counts candidates, media metadata, validated downloads, permissions, reviewed Indian clips and incidents.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Tests cover nested JSON-LD, HTML video/player separation, all adapter patterns, idempotency, permission gates, interrupted downloads, HTTP Range handling, denied access, retry bounds, escaped review HTML, duplicate grouping, split leakage, annotation serials and timestamp validity. The media integration test generates a synthetic pattern and checks real FFmpeg import, contact sheets, trimming/annotation and exact/near duplicate proposals; it skips when FFmpeg/ffprobe are missing. See `VALIDATION.md` for checks actually run and live counts.

Limitations: no browser-rendering adapter was required/proven necessary in this pilot; `no_static_video` pages need manual inspection. Approximate deduplication is a heuristic, not proof of the same incident. Official-channel discovery requires a YouTube key and public website links. The ~3,000 existing clips have not been imported because their directory was not supplied. Model training needs substantially more verified footage and representative normal/near-miss examples.

Documentation consulted: [YouTube search.list](https://developers.google.com/youtube/v3/docs/search/list), [yt-dlp](https://github.com/yt-dlp/yt-dlp), [Schema.org VideoObject](https://schema.org/VideoObject), [FFmpeg downloads](https://ffmpeg.org/download.html).

## Phase 2 — YOLOv8 + R3D-18

Phase 2 lives in `models/raksha_training/`. It audits Phase 1 and any separately supplied video folder, prepares UVH-26 MV vehicle labels for YOLO, and trains/evaluates a full-scene R3D-18 crash classifier from explicitly reviewed temporal video labels. It does not dispatch alerts. UVH-26 is an image/bounding-box source for vehicle detection, never a crash-video source; do not create video clips by repeating UVH stills. It has 14 vehicle categories and no standalone pedestrian class; this baseline preserves all 14 classes without merging. Its COCO `category_id` values are converted to contiguous YOLO IDs in sorted category order. The MV majority-vote annotation is the selected baseline; the ST alternative is audited but never merged into the same ground truth. Its supplied validation set is retained as a benchmark; a deterministic SHA-grouped validation subset from official train is used for model selection. UVH image records do not provide camera IDs, so the generated split is not camera-disjoint.

### Environment and install

The training versions are pinned for Windows x64 / Python 3.13. The project automatically selects CUDA for training and validation when the CUDA-enabled PyTorch environment is available; CPU workers handle video/image loading and preprocessing, while model execution uses the RTX GPU. Automatic mode falls back to CPU when CUDA is unavailable. The system GPU is an NVIDIA RTX 4060 Laptop (8 GiB). CUDA-enabled PyTorch has not been verified in the current environment; use the following commands to install it before GPU training. Install from the `data collector` directory:

```powershell
py -3.13 -m venv .venv-phase2
.\.venv-phase2\Scripts\python.exe -m pip install --upgrade pip
.\.venv-phase2\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv-phase2\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU fallback')"
.\.venv-phase2\Scripts\python.exe -m pip install ultralytics==8.4.173 opencv-python==4.12.0.88 scikit-learn==1.8.0 tensorboard==2.20.0 numpy==2.2.6
.\.venv-phase2\Scripts\python.exe -m pip freeze > requirements-phase2-lock.txt
```

Use `requirements-phase2-cu128.txt` as the reference GPU manifest. For CPU-only checks, replace the PyTorch install command with `python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu`. FFmpeg/ffprobe are recommended on `PATH`; the audit falls back to OpenCV metadata/frame decoding when ffprobe is unavailable and records the inspection backend. Configuration is in `models/configs/phase2.full.json`; `models/configs/phase2.smoke.json` uses small settings. Paths in the config are relative to the project root, so the checkout can be moved as a unit. Source data stays under ignored `dataset/`; generated caches, checkpoints and reports go under ignored `models/runs/`.

### Audit, prepare, train

```powershell
.\.venv-phase2\Scripts\python.exe -m models.raksha_training audit --config models/configs/phase2.full.json
.\.venv-phase2\Scripts\python.exe -m models.raksha_training uvh-audit --config models/configs/phase2.full.json
.\.venv-phase2\Scripts\python.exe -m models.raksha_training manifest-phase1 --config models/configs/phase2.full.json
.\.venv-phase2\Scripts\python.exe -m models.raksha_training prepare-uvh --config models/configs/phase2.full.json
.\.venv-phase2\Scripts\python.exe -m models.raksha_training train-yolo --config models/configs/phase2.full.json
.\.venv-phase2\Scripts\python.exe -m models.raksha_training train-r3d --variant baseline --config models/configs/phase2.full.json
.\.venv-phase2\Scripts\python.exe -m models.raksha_training compare-r3d --runs models/runs/phase2/r3d/baseline models/runs/phase2/r3d/class_weighted
.\.venv-phase2\Scripts\python.exe -m models.raksha_training eval-r3d --checkpoint models/runs/phase2/r3d/baseline/best.pt --split validation --config models/configs/phase2.full.json
# After freezing the validation-selected threshold and all settings, pass that value explicitly:
.\.venv-phase2\Scripts\python.exe -m models.raksha_training eval-r3d --checkpoint models/runs/phase2/r3d/baseline/best.pt --split test --threshold $frozenValidationThreshold --final-test --config models/configs/phase2.full.json
```

Run validation threshold selection and freeze the resulting threshold before one untouched test evaluation. The R3D data export requires a confirmed human label, India provenance, permission/research-basis record, incident ID, source video ID and a local file. Pending Phase 1 labels are excluded. The R3D initializer is TorchVision `R3D_18_Weights.KINETICS400_V1` (generic Kinetics-400 supervised action-recognition pretraining); YOLOv8-S starts from the generic COCO detection checkpoint and adapts the detection head to UVH-26 classes. These are Indian task fine-tuning, not exclusively Indian pretraining. The baseline uses 16 frames sampled at 8 fps across a 2-second span, BGR-to-RGB, 112x112 letterboxing that preserves the full scene, and the selected weights' normalization. Audio is excluded, temporal reversal is disabled, and horizontal flipping defaults to zero.

For a software-only synthetic smoke test (no real clips and no quality claim):

```powershell
.\.venv-phase2\Scripts\python.exe -m models.raksha_training smoke --config models/configs/phase2.smoke.json
```

Smoke output is quarantined under `models/runs/smoke/synthetic_only/`. It checks decode, tensor layout, optimizer step, metrics and a one-epoch synthetic YOLO run. It is not a model evaluation. Training variants are one-factor comparisons: `baseline`, `class_weighted`, `balanced_sampling`, or `focal_loss`; don't combine them without a separately registered experiment. Resume R3D with `train-r3d --resume <checkpoint>`. TensorBoard logs and JSONL epoch logs sit next to checkpoints.

### Inference and evaluation cautions

```powershell
.\.venv-phase2\Scripts\python.exe -m models.raksha_training infer --source path\to\authorized-camera.mp4 --r3d-checkpoint models\runs\phase2\r3d\baseline\best.pt --yolo-checkpoint models\runs\phase2\yolo\yolov8\weights\best.pt --config models/configs/phase2.full.json
```

`--source` accepts a local video, webcam index (`0`), or an authorized RTSP URL. The full-scene classifier runs independently of YOLO detections. Local prediction JSONL and event-candidate records include timestamps; no emergency dispatch is implemented. `--show` enables rendering and should be off for benchmarks. Run `eval-events` on validation predictions first; it sweeps thresholds while replaying the persistence/rearm rule and reports the recall/burden tradeoff. Apply the selected validation threshold with `infer --positive-threshold $frozenValidationThreshold`, then evaluate test predictions once using `eval-events --annotations <held-out-events.jsonl> --predictions <test-predictions.jsonl> --camera-hours <camera-hours.json> --split test --threshold $frozenValidationThreshold --final-test`. Event recall, false alerts/camera-hour and delay require continuous held-out streams with incident onset, camera identity and observed camera-hours; a clip/window score cannot substitute for those. News edits/watermarks and source-specific graphics require review, and edited news compilations cannot establish live CCTV reliability.

The latest audit found only one Phase 1 assistant-drafted crash annotation, still pending human review and excluded by the training gate; the approximately 3,000 additional clips have not been located under the configured path. Consequently the code can be audited and smoke-tested, but meaningful R3D training/evaluation is blocked until eligible reviewed crash and normal footage are supplied. The current full-data YOLO run was canceled before training started. No scores or trained production checkpoint are claimed by the implementation itself. Official dataset and paper: [UVH-26 dataset card](https://huggingface.co/datasets/iisc-aim/UVH-26), [UVH-26 paper](https://arxiv.org/abs/2511.02563).
