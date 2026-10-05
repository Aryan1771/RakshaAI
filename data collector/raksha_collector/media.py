"""Original-preserving downloads, ffprobe inspection and review derivatives."""
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlsplit
import imagehash
from PIL import Image, ImageDraw
from .network import FetchError
from .sources import parse_page
from .storage import now, digest, new_record


def binary(name):
    executable = os.environ.get(name.upper()) or shutil.which(name)
    if not executable or not Path(executable).is_file():
        raise ValueError(f"{name} is required: install it on PATH or set {name.upper()} in .env")
    return executable


def probe(path):
    result = subprocess.run([binary("ffprobe"), "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)], capture_output=True, check=True, timeout=60)
    data = json.loads(result.stdout)
    streams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    if not streams:
        raise ValueError("No video stream")
    stream = streams[0]
    rate = stream.get("avg_frame_rate", "0/1").split("/")
    fps = float(rate[0]) / float(rate[1]) if len(rate) == 2 and float(rate[1]) else 0
    duration = float(data.get("format", {}).get("duration", stream.get("duration", 0)))
    if duration <= 0:
        raise ValueError("Video has no usable duration")
    return {"duration": duration, "width": stream.get("width"), "height": stream.get("height"), "fps": fps, "codec": stream.get("codec_name")}


def checksum(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sidecar(record, directory):
    target = directory / "metadata.json"
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(target)


class QuietLogger:
    # yt-dlp messages can contain signed URLs; do not write them to disk/stdout.
    def debug(self, msg):
        pass

    warning = debug
    error = debug


def download_one(store, record, client, root, runner=None):
    from .review import india_location_verified
    if not india_location_verified(record):
        return {"id": record["id"], "status": "location_unverified"}
    permission = record["permission"]
    if permission.get("status") == "research_basis":
        return {"id": record["id"], "status": "segment_required"}
    if permission["status"] != "permitted" or not permission.get("evidence", "").strip():
        return {"id": record["id"], "status": "permission_blocked"}
    directory = Path(root) / "originals" / record["id"]
    directory.mkdir(parents=True, exist_ok=True)
    # Validate tools before requesting any media.
    binary("ffprobe")
    binary("ffmpeg")
    originals = [p for p in directory.glob("original.*") if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov", ".flv", ".ts"}]
    final = originals[0] if originals else directory / "original.mkv"
    try:
        # Recover after a crash between the atomic file move and database commit.
        if not final.exists():
            completed = [p for p in directory.glob("download.*") if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov", ".flv", ".ts"}]
            valid = []
            for p in completed:
                try:
                    probe(p)
                    valid.append(p)
                except (ValueError, subprocess.SubprocessError):
                    continue
            if valid:
                # A completed merged download is reused; intermediate fragment
                # names (download.f137.mp4) never count as a finished download.
                completed = [p for p in valid if p.stem == "download"]
                if completed:
                    final = directory / ("original" + completed[0].suffix)
                    completed[0].replace(final)
            if not final.exists():
                record["download"] = {"status": "downloading", "local_path": None, "started_at": now()}
                store.save(record, "download_start")
                url = record["article_url"]
                if record["publisher"] != "YouTube":
                    response = client.get(url)
                    metadata = parse_page(response.text, response.url)
                    choices = metadata["media_urls"] or metadata["player_urls"]
                    if not choices:
                        raise FetchError("unsupported_no_static_media")
                    if len(choices) > 1 and "media_index" not in record["human"]:
                        raise FetchError("review_required_multiple_media")
                    index = record["human"].get("media_index", 0)
                    if index < 0 or index >= len(choices):
                        raise FetchError("review_required_media_index")
                    url = choices[index]
                if not client.allowed(url):
                    raise FetchError("robots_disallowed")
                client.pace(urlsplit(url).netloc)
                options = {
                    "outtmpl": str(directory / "download.%(ext)s"), "format": "bestvideo*+bestaudio/best",
                    "merge_output_format": "mkv", "noplaylist": True, "continuedl": True,
                    "overwrites": False, "retries": 0, "fragment_retries": 0,
                    "concurrent_fragment_downloads": 1, "socket_timeout": 25,
                    "sleep_interval_requests": client.interval, "sleep_interval": client.interval,
                    "ffmpeg_location": str(Path(binary("ffmpeg")).parent),
                    "quiet": True, "no_warnings": True, "logger": QuietLogger(),
                    "writeinfojson": False, "writethumbnail": False,
                }
                if runner:
                    runner(url, options)
                else:
                    from yt_dlp import YoutubeDL
                    with YoutubeDL(options) as ydl:
                        ydl.download([url])
                completed = [p for p in directory.glob("download.*") if p.stem == "download" and p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov", ".flv", ".ts"}]
                if len(completed) != 1:
                    raise FetchError("incomplete_download")
                probe(completed[0])
                # Renaming is lossless: container is documented by ffprobe; use
                # its actual extension instead of pretending all originals MKV.
                final = directory / ("original" + completed[0].suffix)
                completed[0].replace(final)
        info = probe(final)
        record["download"] = {"status": "downloaded", "local_path": str(final.resolve()), "checksum": checksum(final), "completed_at": now(), **info}
        store.save(record, "download_complete")
        sidecar(record, directory)
        return {"id": record["id"], "status": "downloaded"}
    except Exception as error:
        if isinstance(error, FetchError):
            category = error.category
        elif isinstance(error, (ValueError, subprocess.SubprocessError)):
            category = "validation_failed"
        else:
            message = str(error).casefold()
            category = "unsupported" if "unsupported url" in message else "access_denied" if any(x in message for x in ("403", "401", "captcha", "sign in", "private video")) else "transient_download"
        record["download"] = {**record["download"], "status": category, "failed_at": now()}
        store.save(record, "download_failure")
        sidecar(record, directory)
        return {"id": record["id"], "status": category}


def download_news_segment(store, record, client, root, start, end, media_index=None, runner=None):
    """Download only a reviewer-selected time range from a news archive page."""
    if (not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in (start, end))
            or start < 0 or end <= start or end - start > 300):
        raise ValueError("Segment bounds must be finite, increasing, and no longer than five minutes")
    from .review import india_location_verified
    if not india_location_verified(record):
        return {"id": record["id"], "status": "location_unverified"}
    permission = record["permission"]
    if permission.get("status") not in {"permitted", "research_basis"} or not permission.get("evidence", "").strip():
        return {"id": record["id"], "status": "permission_blocked"}
    directory = Path(root) / "segments" / record["id"]
    directory.mkdir(parents=True, exist_ok=True)
    binary("ffprobe")
    binary("ffmpeg")
    try:
        response = client.get(record["article_url"])
        metadata = parse_page(response.text, response.url)
        choices = metadata["media_urls"]
        if not choices:
            return {"id": record["id"], "status": "unsupported_no_static_media"}
        if len(choices) > 1 and media_index is None:
            return {"id": record["id"], "status": "review_required_media_index", "media_candidates": len(choices)}
        index = 0 if media_index is None else media_index
        if not isinstance(index, int) or index < 0 or index >= len(choices):
            return {"id": record["id"], "status": "review_required_media_index", "media_candidates": len(choices)}
        url = choices[index]
        if not client.allowed(url):
            raise FetchError("robots_disallowed")
        client.pace(urlsplit(url).netloc)
        for stale in directory.glob("segment*"):
            if stale.is_file():
                stale.unlink()
        output = directory / "segment.mp4"
        partial = directory / "segment.partial.mp4"
        partial.unlink(missing_ok=True)
        options = {"start": float(start), "duration": float(end - start), "output": str(partial),
                   "ffmpeg": binary("ffmpeg")}
        if runner:
            runner(url, options)
        else:
            subprocess.run([binary("ffmpeg"), "-v", "error", "-y", "-ss", str(start), "-i", url,
                            "-t", str(end - start), "-map", "0:v:0", "-map", "0:a?",
                            "-c:v", "libx264", "-crf", "18", "-c:a", "aac", "-movflags", "+faststart",
                            str(partial)], capture_output=True, check=True, timeout=600)
        if not partial.is_file():
            raise FetchError("incomplete_segment_download")
        info = probe(partial)
        requested = end - start
        if info["duration"] > requested + 3:
            partial.unlink(missing_ok=True)
            raise FetchError("segment_bounds_not_honored")
        partial.replace(output)
        record["download"] = {"status": "downloaded", "local_path": str(output.resolve()),
                              "checksum": checksum(output), "completed_at": now(), **info}
        record["source_segment"] = {"start_seconds": float(start), "end_seconds": float(end),
                                    "media_index": index, "method": "ffmpeg_bounded_url_seek_transcode",
                                    "requested_duration": requested}
        record["transformations"].append({"kind": "news_archive_segment", **record["source_segment"], "created_at": now()})
        store.save(record, "news_segment_download")
        sidecar(record, directory)
        return {"id": record["id"], "status": "downloaded_segment", "path": str(output.resolve()),
                "duration": info["duration"]}
    except Exception as error:
        if isinstance(error, FetchError):
            category = error.category
        elif isinstance(error, (ValueError, subprocess.SubprocessError)):
            category = "validation_failed"
        else:
            message = str(error).casefold()
            category = "access_denied" if any(x in message for x in ("403", "401", "captcha", "sign in", "private video")) else "segment_download_failure"
        record["download"] = {**record["download"], "status": category, "failed_at": now()}
        store.save(record, "news_segment_failure")
        return {"id": record["id"], "status": category}


def import_local(store, directory):
    stats = {"command": "import", "new_records": 0, "inspected": 0, "failed": 0}
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError("Import directory does not exist")
    binary("ffprobe")
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".ts"}:
            continue
        record = new_record("", "local", source_id=str(path.resolve()))
        record["machine"]["title"] = path.name
        try:
            record["download"] = {"status": "imported", "local_path": str(path), "checksum": checksum(path), **probe(path)}
            stats["inspected"] += 1
        except (ValueError, subprocess.SubprocessError, OSError):
            record["download"] = {"status": "validation_failed", "local_path": str(path)}
            stats["failed"] += 1
        stats["new_records"] += int(store.add(record))
    store.run(stats)
    return stats


def frames(path, duration, count=8):
    for n in range(count):
        timestamp = duration * (n + 0.5) / count
        result = subprocess.run([binary("ffmpeg"), "-v", "error", "-ss", str(timestamp), "-i", str(path), "-frames:v", "1", "-vf", "scale=320:-1", "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True, check=True, timeout=60)
        yield timestamp, Image.open(io.BytesIO(result.stdout)).convert("RGB")


def fingerprints(record):
    d = record["download"]
    hashes = []
    for _, frame in frames(d["local_path"], d["duration"]):
        w, h = frame.size
        center = frame.crop((int(w * .12), int(h * .12), int(w * .88), int(h * .88)))
        hashes.append([str(imagehash.phash(frame)), str(imagehash.phash(center))])
    return hashes


def near_distance(a, b):
    def directed(xs, ys):
        distances = []
        for x in xs:
            distances.append(min(imagehash.hex_to_hash(h1) - imagehash.hex_to_hash(h2) for h1 in x for y in ys for h2 in y))
        return sorted(distances)[len(distances) // 2]
    return max(directed(a, b), directed(b, a))


def deduplicate(store, threshold=8):
    all_records = store.records()
    source_ids = {}
    for r in all_records:
        if r.get("source_video_id"):
            source_key = (r["publisher"], r["source_video_id"])
            if source_key in source_ids:
                store.pair(r["id"], source_ids[source_key], "source_id", 0)
            else:
                source_ids[source_key] = r["id"]
    records = [r for r in all_records if r["download"].get("checksum")]
    errors = []
    for record in records:
        if not record["machine"].get("frame_hashes"):
            try:
                record["machine"]["frame_hashes"] = fingerprints(record)
                store.save(record, "fingerprint")
            except (ValueError, OSError, subprocess.SubprocessError):
                errors.append(record["id"])
    for i, a in enumerate(records):
        for b in records[i + 1:]:
            exact = a["download"]["checksum"] == b["download"]["checksum"]
            if exact:
                store.pair(a["id"], b["id"], "exact", 0)
            elif a["machine"].get("frame_hashes") and b["machine"].get("frame_hashes"):
                distance = near_distance(a["machine"]["frame_hashes"], b["machine"]["frame_hashes"])
                if distance <= threshold:
                    store.pair(a["id"], b["id"], "near", distance)
    # Connected components, including transitive matches; these are proposals,
    # never incident labels. Rejected pairs do not link components.
    parent = {r["id"]: r["id"] for r in all_records}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for pair in store.pairs():
        a, b = pair["a"], pair["b"]
        if pair["status"] != "rejected" and a in parent and b in parent:
            parent[find(a)] = find(b)
    groups = {}
    for rid in parent:
        groups.setdefault(find(rid), []).append(rid)
    for members in groups.values():
        group = "dup-" + digest("|".join(sorted(members))) if len(members) > 1 else None
        for rid in members:
            record = store.get(rid)
            record["duplicate_group"] = group
            store.save(record, "duplicate_proposal")
    return {"records": len(records), "groups": sum(len(v) > 1 for v in groups.values()), "pairs": len(store.pairs()), "fingerprint_failures": errors}


def contact_sheet(store, record, root):
    d = record["download"]
    if not d.get("local_path") or not d.get("duration"):
        raise ValueError("A validated local video is required for a contact sheet")
    target = Path(root) / "derivatives" / record["id"] / "contact.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGB", (1280, 420), "#161b22")
    draw = ImageDraw.Draw(sheet)
    for index, (time, frame) in enumerate(frames(d["local_path"], d["duration"])):
        frame.thumbnail((320, 180))
        x, y = (index % 4) * 320, (index // 4) * 210
        sheet.paste(frame, (x, y))
        draw.text((x + 8, y + 185), f"{time:.2f}s", fill="white")
    sheet.save(target)
    transform = {"kind": "contact_sheet", "path": str(target.resolve()), "source_checksum": d["checksum"], "samples": 8, "created_at": now()}
    record["transformations"].append(transform)
    store.save(record, "contact_sheet")
    return str(target.resolve())


def trim(store, record, root, start, end):
    d = record["download"]
    if not d.get("local_path") or not 0 <= start < end <= d.get("duration", 0):
        raise ValueError("Trim must lie within a validated video's duration")
    target = Path(root) / "derivatives" / record["id"] / f"trim-{start:g}-{end:g}.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ValueError("Derivative already exists; originals and derivatives are never overwritten")
    partial = target.with_suffix(".partial.mp4")
    subprocess.run([binary("ffmpeg"), "-v", "error", "-y", "-i", d["local_path"], "-ss", str(start), "-t", str(end - start), "-c:v", "libx264", "-c:a", "aac", str(partial)], capture_output=True, check=True, timeout=600)
    info = probe(partial)
    partial.replace(target)
    record["transformations"].append({"kind": "trim", "start": start, "end": end, "path": str(target.resolve()), "source_checksum": d["checksum"], "checksum": checksum(target), "created_at": now(), **info})
    store.save(record, "trim")
    return str(target.resolve())
