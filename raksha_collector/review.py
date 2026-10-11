"""Explicit human review and offline, non-autoplay contact-sheet index."""
import html
import json
from pathlib import Path
from .storage import now

FOOTAGE = {"unknown", "cctv", "dashcam", "aftermath_only", "near_miss", "compilation", "reenactment", "animation", "unrelated"}


def permission(store, rid, status, evidence, reviewer):
    if status not in {"unknown", "requested", "permitted", "research_basis", "denied"}:
        raise ValueError("Invalid permission status")
    if not reviewer.strip() or (status != "unknown" and not evidence.strip()):
        raise ValueError("Reviewer and documented permission basis are required")
    record = store.get(rid)
    if status in {"permitted", "research_basis"} and not india_location_verified(record):
        raise ValueError("Only footage verified as recorded in India can be permitted for collection; review country and provide location_evidence first")
    record["permission"] = {"status": status, "evidence": evidence, "basis_type": "reviewer_asserted_research_exception" if status == "research_basis" else "rights_holder_or_license" if status == "permitted" else None, "reviewer": reviewer, "updated_at": now()}
    store.save(record, "permission_basis")
    return record["permission"]


def india_location_verified(record):
    human = record.get("human", {})
    return human.get("country", "").strip().casefold() in {"india", "in"} and bool(human.get("location_evidence", "").strip())


def annotate(store, rid, patch, reviewer):
    record = store.get(rid)
    allowed = set(record["human"]) | {"original_footage_owner", "media_index"}
    if set(patch) - allowed:
        raise ValueError("Unknown review fields: " + ", ".join(set(patch) - allowed))
    if not reviewer.strip():
        raise ValueError("Reviewer is required")
    human = {**record["human"], **patch, "reviewer": reviewer, "reviewed_at": now()}
    if human["footage_type"] not in FOOTAGE:
        raise ValueError("Invalid footage_type")
    if human["review_status"] not in {"pending", "reviewed"} or human["decision"] not in {"pending", "include", "exclude", "needs_review"}:
        raise ValueError("Invalid review status or decision")
    if human["country"] != "unknown" and not human["location_evidence"].strip():
        raise ValueError("Confirmed location requires evidence")
    start, end = human["event_start"], human["event_end"]
    if (start is None) != (end is None):
        raise ValueError("Provide both event_start and event_end")
    if start is not None:
        duration = record["download"].get("duration")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or duration is None or not 0 <= start < end <= duration:
            raise ValueError("Event bounds must be inside a validated local video")
    if "media_index" in human and (not isinstance(human["media_index"], int) or human["media_index"] < 0):
        raise ValueError("media_index must be a nonnegative integer")
    if "original_footage_owner" in human:
        record["original_footage_owner"] = human.pop("original_footage_owner")
    record["human"] = human
    store.save(record, "human_review")
    return record


def assign_incident(store, ids, incident, reviewer):
    if not incident.strip() or not reviewer.strip():
        raise ValueError("Incident ID and reviewer are required")
    chosen = [store.get(rid) for rid in ids]
    old = {r["incident_id"] for r in chosen if r["incident_id"]}
    # Merge whole existing incident groups, never split one by accident.
    for record in store.records():
        if record["id"] in ids or record["incident_id"] in old:
            record["incident_id"] = incident
            record["human"]["incident_reviewer"] = reviewer
            store.save(record, "incident_group")
    return {"incident_id": incident}


def pair_decision(store, a, b, status):
    if status not in {"confirmed", "rejected"}:
        raise ValueError("Pair decision must be confirmed or rejected")
    with store.db:
        result = store.db.execute("UPDATE pairs SET status=? WHERE a=? AND b=?", (status, *sorted([a, b])))
        if not result.rowcount:
            raise ValueError("Duplicate pair does not exist")
    store.run({"command": "pair_review", "a": a, "b": b, "status": status})
    return {"a": a, "b": b, "status": status}


def review_index(store, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    esc = lambda x: html.escape(str(x), quote=True)
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>RakshaAI review queue</title>',
             '<style>body{font:16px system-ui;max-width:1050px;margin:40px auto;background:#101820;color:#eee}article{border-top:1px solid #53616a;padding:24px 0}a{color:#7dd3fc}pre{white-space:pre-wrap}img{max-width:100%}summary{cursor:pointer}</style>',
             '<h1>RakshaAI review queue</h1><p>Footage may be graphic. Images appear only when you open a contact sheet. No automatic video playback.</p><p>Record decisions with the review command and a JSON annotation file. Machine scores are suggestions.</p>']
    for r in sorted(store.records(), key=lambda r: -r["machine"].get("relevance_score", 0)):
        parts.append(f'<article><h2>{esc(r["machine"].get("title") or r["id"])}</h2><code>{esc(r["id"])}</code>')
        if r["article_url"]:
            parts.append(f'<p><a href="{esc(r["article_url"])}" rel="noreferrer">Source page</a></p>')
        parts.append(f'<pre>{esc(json.dumps({k: r[k] for k in ("publisher", "machine", "human", "permission", "download", "duplicate_group", "incident_id")}, ensure_ascii=False, indent=2))}</pre>')
        for transform in r["transformations"]:
            if transform["kind"] == "contact_sheet":
                parts.append(f'<p><a href="{esc(Path(transform["path"]).as_uri())}">Open contact sheet (may be graphic)</a></p>')
        parts.append('</article>')
    parts.append('<h2>Duplicate review queue</h2><pre>' + esc(json.dumps(store.pairs(), indent=2)) + '</pre></html>')
    target.write_text("\n".join(parts), encoding="utf-8")
    return str(target.resolve())
