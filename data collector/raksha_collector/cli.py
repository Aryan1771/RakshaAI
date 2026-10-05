import argparse
import json
import sys
from pathlib import Path
from dotenv import load_dotenv
from .storage import Store
from .network import Client, FetchError
from . import sources, media, review, exporting, annotation, datasets


def main():
    parser = argparse.ArgumentParser(description="RakshaAI traceable footage collection")
    parser.add_argument("--config", default="config.example.json")
    parser.add_argument("--db", default="data/collection.sqlite3")
    parser.add_argument("--data", default="data")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("discover", "extract", "youtube", "download"):
        cmd = sub.add_parser(command)
        cmd.add_argument("--limit", type=int, default=20)
        if command in {"extract", "download"}:
            cmd.add_argument("--retry", action="store_true", help="Explicitly retry previously failed records")
        if command in {"discover", "extract", "youtube"}:
            for flag in ("city", "state", "publisher", "published-after", "published-before"):
                cmd.add_argument("--" + flag)
        if command == "download":
            cmd.add_argument("--id")
    cmd = sub.add_parser("import")
    cmd.add_argument("directory")
    cmd = sub.add_parser("dataset-uvh26", help="Import optional India-specific UVH-26 vehicle image dataset")
    cmd.add_argument("--revision", default="main")
    cmd.add_argument("--hf", default="hf", help="Hugging Face CLI executable")
    cmd.add_argument("--download", action="store_true", help="Download dataset files (90 GB upstream dataset)")
    cmd = sub.add_parser("permission")
    cmd.add_argument("id")
    cmd.add_argument("status", choices=["unknown", "requested", "permitted", "research_basis", "denied"])
    cmd.add_argument("--evidence", default="")
    cmd.add_argument("--reviewer", required=True)
    cmd = sub.add_parser("download-segment", help="Fetch a reviewer-selected CCTV segment from a news archive")
    cmd.add_argument("id")
    cmd.add_argument("--start", required=True, type=float, help="Segment start in source-video seconds")
    cmd.add_argument("--end", required=True, type=float, help="Segment end in source-video seconds")
    cmd.add_argument("--media-index", type=int, help="Zero-based choice after reviewing multiple direct media candidates")
    cmd = sub.add_parser("review")
    cmd.add_argument("--id")
    cmd.add_argument("--annotations")
    cmd.add_argument("--reviewer")
    cmd.add_argument("--out", default="data/review/index.html")
    cmd = sub.add_parser("incident")
    cmd.add_argument("incident_id")
    cmd.add_argument("ids", nargs="+")
    cmd.add_argument("--reviewer", required=True)
    cmd = sub.add_parser("pair-review")
    cmd.add_argument("a")
    cmd.add_argument("b")
    cmd.add_argument("status", choices=["confirmed", "rejected"])
    cmd = sub.add_parser("deduplicate")
    cmd.add_argument("--threshold", type=int, default=8)
    cmd = sub.add_parser("contact-sheet")
    cmd.add_argument("id")
    cmd = sub.add_parser("trim")
    cmd.add_argument("id")
    cmd.add_argument("--start", type=float, required=True)
    cmd.add_argument("--end", type=float, required=True)
    cmd = sub.add_parser("export")
    cmd.add_argument("--out", default="data/exports")
    cmd.add_argument("--dataset-only", action="store_true")
    cmd.add_argument("--splits", action="store_true")
    sub.add_parser("report")
    sub.add_parser("doctor")
    cmd = sub.add_parser("annotate-crash", help="Create crash_<vehicles>(<serial>).mp4 with pre-crash context")
    cmd.add_argument("id")
    cmd.add_argument("--vehicles", required=True, type=int)
    cmd.add_argument("--impact-start", required=True, type=float)
    cmd.add_argument("--impact-end", required=True, type=float)
    cmd.add_argument("--lead-in", type=float, default=5)
    cmd.add_argument("--aftermath", type=float, default=3)
    cmd.add_argument("--normal-before-confirmed", action="store_true")
    cmd.add_argument("--origin", choices=["human", "assistant_visual_review"], default="human")
    cmd.add_argument("--reviewer", required=True)
    cmd.add_argument("--notes", default="")
    cmd = sub.add_parser("export-annotations")
    cmd.add_argument("--out", default="data/annotation_exports")
    cmd.add_argument("--dataset-only", action="store_true")
    args = parser.parse_args()
    load_dotenv(Path.cwd() / ".env")
    try:
        config = json.loads(Path(args.config).read_text(encoding="utf-8-sig")) if Path(args.config).exists() else {}
        for name in ("city", "state", "publisher", "published_after", "published_before"):
            if getattr(args, name, None):
                config[name] = getattr(args, name)
        if hasattr(args, "limit") and not 1 <= args.limit <= 100:
            raise ValueError("--limit must be between 1 and 100")
        store = Store(args.db)
        client = Client(config.get("host_interval_seconds", 2), config.get("retries", 2))
        command = args.command
        if command == "discover":
            result = sources.discover(store, client, config, args.limit)
        elif command == "extract":
            result = sources.extract(store, client, config, args.limit, args.retry)
        elif command == "youtube":
            result = sources.youtube(store, client, config, args.limit)
        elif command == "download":
            records = [store.get(args.id)] if args.id else store.records()
            records = [r for r in records if r["publisher"] != "local" and (args.retry or r["download"]["status"] in {"pending", "downloading", "transient_network", "transient_http", "transient_download", "incomplete_download"})]
            result = [media.download_one(store, r, client, args.data) for r in records[:args.limit]]
        elif command == "download-segment":
            result = media.download_news_segment(store, store.get(args.id), client, args.data,
                                                 args.start, args.end, args.media_index)
        elif command == "import":
            result = media.import_local(store, args.directory)
        elif command == "dataset-uvh26":
            result = datasets.import_uvh26(args.data, args.revision, args.download, args.hf)
        elif command == "permission":
            result = review.permission(store, args.id, args.status, args.evidence, args.reviewer)
        elif command == "review":
            if args.annotations:
                if not args.id or not args.reviewer:
                    raise ValueError("--annotations requires --id and --reviewer")
                result = review.annotate(store, args.id, json.loads(Path(args.annotations).read_text(encoding="utf-8-sig")), args.reviewer)
            else:
                result = review.review_index(store, args.out)
        elif command == "incident":
            result = review.assign_incident(store, args.ids, args.incident_id, args.reviewer)
        elif command == "pair-review":
            result = review.pair_decision(store, args.a, args.b, args.status)
        elif command == "deduplicate":
            if not 0 <= args.threshold <= 64:
                raise ValueError("Threshold must be between 0 and 64")
            result = media.deduplicate(store, args.threshold)
        elif command == "contact-sheet":
            result = media.contact_sheet(store, store.get(args.id), args.data)
        elif command == "trim":
            result = media.trim(store, store.get(args.id), args.data, args.start, args.end)
        elif command == "export":
            result = exporting.export(store, args.out, args.dataset_only, args.splits)
        elif command == "report":
            result = exporting.report(store)
        elif command == "annotate-crash":
            result = annotation.annotate_crash(store, args.id, args.data, args.vehicles,
                        args.impact_start, args.impact_end, args.reviewer, args.lead_in,
                        args.aftermath, args.normal_before_confirmed, args.notes, origin=args.origin)
        elif command == "export-annotations":
            result = annotation.export_annotations(store, args.out, args.dataset_only)
        else:
            import os
            result = {"python": sys.version.split()[0], "youtube_api_key_present": bool(os.getenv("YOUTUBE_API_KEY"))}
            for name in ("ffmpeg", "ffprobe"):
                try:
                    result[name] = media.binary(name)
                except ValueError:
                    result[name] = "missing"
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, KeyError, OSError, FetchError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
