import csv
import json
from pathlib import Path
from .storage import digest, now


def eligible(r):
    return (r["human"]["review_status"] == "reviewed" and r["human"]["decision"] == "include"
            and r["human"]["country"].casefold() in {"india", "in"}
            and bool(r["human"]["location_evidence"].strip())
            and r["permission"]["status"] in {"permitted", "research_basis"} and bool(r["permission"]["evidence"].strip())
            and r["download"]["status"] in {"downloaded", "imported"} and bool(r["incident_id"]))


def report(store):
    rows = store.records()
    reviewed = [r for r in rows if r["human"]["review_status"] == "reviewed" and r["human"]["country"].casefold() in {"india", "in"} and r["human"]["location_evidence"] and r["download"]["status"] in {"downloaded", "imported"}]
    return {
        "generated_at": now(), "total_records": len(rows),
        "candidate_pages": sum(r["publisher"] not in {"local", "YouTube"} for r in rows),
        "youtube_candidates": sum(r["publisher"] == "YouTube" for r in rows),
        "pages_with_media_or_player_metadata": sum(bool(r["machine"].get("media_urls") or r["machine"].get("player_urls")) for r in rows),
        "accessible_videos_validated_by_ffprobe": sum(r["download"]["status"] == "downloaded" for r in rows),
        "permission_permitted_records": sum(r["permission"]["status"] == "permitted" for r in rows),
        "reviewer_asserted_research_basis_records": sum(r["permission"]["status"] == "research_basis" for r in rows),
        "permitted_downloads": sum(r["permission"]["status"] == "permitted" and r["download"]["status"] == "downloaded" for r in rows),
        "imported_clips": sum(r["download"]["status"] == "imported" for r in rows),
        "reviewed_indian_clips": len(reviewed),
        "unique_reviewed_indian_incidents": len({r["incident_id"] for r in reviewed if r["incident_id"]}),
        "dataset_eligible_records": sum(eligible(r) for r in rows),
        "duplicate_pairs_pending": sum(p["status"] == "pending" for p in store.pairs()),
        "extraction_statuses": {status: sum(r["extraction"]["status"] == status for r in rows) for status in sorted({r["extraction"]["status"] for r in rows})},
        "note": "Media/player metadata is not evidence of playable video, Indian event location, permission or an accident. Incidents require human grouping.",
    }


def flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = prefix + key
        if isinstance(item, dict):
            result.update(flatten(item, name + "."))
        else:
            result[name] = json.dumps(item, ensure_ascii=False) if isinstance(item, list) else item
    return result


def export(store, directory, dataset_only=False, splits=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    all_rows = store.records()
    rows = [r for r in all_rows if eligible(r)] if dataset_only else all_rows
    if splits:
        if not dataset_only:
            raise ValueError("Splits require --dataset-only")
        # Build links from current pairs (not a possibly stale duplicate_group).
        # Unresolved proposals conservatively remain together to avoid leakage.
        parent = {r["id"]: r["id"] for r in all_rows}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        groups = {}
        for r in all_rows:
            if r["incident_id"]:
                previous = groups.setdefault(r["incident_id"], r["id"])
                parent[find(r["id"])] = find(previous)
        for p in store.pairs():
            if p["status"] != "rejected":
                parent[find(p["a"])] = find(p["b"])
        members = {}
        for rid in parent:
            members.setdefault(find(rid), []).append(rid)
        for row in rows:
            group = min(members[find(row["id"])] )
            number = int(digest(group)[:8], 16) % 100
            row["split_group"] = group
            row["split"] = "train" if number < 80 else "validation" if number < 90 else "test"
    jsonl = directory / "records.jsonl"
    jsonl.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    flat = [flatten(r) for r in rows]
    keys = sorted({key for r in flat for key in r}) or ["id"]
    def csv_safe(value):
        return "'" + value if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")) else value
    with (directory / "records.csv").open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: csv_safe(v) for k, v in r.items()} for r in flat)
    (directory / "duplicate_pairs.json").write_text(json.dumps(store.pairs(), indent=2), encoding="utf-8")
    (directory / "collection_report.json").write_text(json.dumps(report(store), indent=2), encoding="utf-8")
    (directory / "export_manifest.json").write_text(json.dumps({"created_at": now(), "records": len(rows), "dataset_only": dataset_only, "splits": splits, "split_policy": "80/10/10 deterministic connected groups; freeze this manifest before training"}, indent=2), encoding="utf-8")
    return {"directory": str(directory.resolve()), "records": len(rows)}
