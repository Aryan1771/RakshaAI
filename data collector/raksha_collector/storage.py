"""Persistent records with separate machine observations and human decisions."""
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def now():
    return datetime.now(timezone.utc).isoformat()


def safe_url(url):
    """Persist only public identities, never signed/authenticated query strings."""
    p = urlsplit(str(url))
    if p.scheme not in {"https", "http"} or p.username or p.password:
        return ""
    # Allow only public identity/pagination parameters. Other query strings must
    # be re-extracted into memory immediately before a permitted download.
    query = [(k, v) for k, v in parse_qsl(p.query) if k in {"v", "id", "page"}]
    return urlunsplit((p.scheme.lower(), p.netloc.lower(), p.path or "/", urlencode(query), ""))


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def new_record(url, publisher, query="", source_id=None):
    url = safe_url(url)
    identity = f"{publisher}:{source_id}" if source_id else url
    return {
        "id": digest(identity), "source_video_id": source_id, "article_url": url,
        "publisher": publisher, "original_footage_owner": None,
        "discovery_queries": [query] if query else [], "collected_at": now(),
        "machine": {}, "human": {
            "event_date": None, "city": None, "state": None, "country": "unknown",
            "location_evidence": "", "footage_type": "unknown", "camera_viewpoint": "unknown",
            "impact_visibility": "unknown", "lighting": "unknown", "weather": "unknown",
            "occlusion": "unknown", "event_start": None, "event_end": None,
            "reviewer": "", "review_status": "pending", "decision": "pending", "notes": "",
        },
        "permission": {"status": "unknown", "evidence": "", "reviewer": ""},
        "download": {"status": "pending", "local_path": None},
        "extraction": {"status": "pending"}, "duplicate_group": None, "incident_id": None,
        "transformations": [],
    }


class Store:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS audit (time TEXT, record_id TEXT, action TEXT, payload TEXT);
          CREATE TABLE IF NOT EXISTS pairs (a TEXT, b TEXT, kind TEXT, distance REAL,
            status TEXT DEFAULT 'pending', PRIMARY KEY(a,b,kind));
          CREATE TABLE IF NOT EXISTS runs (time TEXT, payload TEXT);
          CREATE TABLE IF NOT EXISTS annotations (serial INTEGER PRIMARY KEY AUTOINCREMENT,
            annotation_key TEXT UNIQUE NOT NULL, record_id TEXT NOT NULL, payload TEXT NOT NULL);
        """)

    def get(self, rid):
        row = self.db.execute("SELECT payload FROM records WHERE id=?", (rid,)).fetchone()
        if not row:
            raise ValueError(f"Unknown record: {rid}")
        return json.loads(row[0])

    def records(self):
        return [json.loads(r[0]) for r in self.db.execute("SELECT payload FROM records ORDER BY id")]

    def save(self, record, action="update"):
        payload = json.dumps(record, ensure_ascii=False)
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO records VALUES (?,?)", (record["id"], payload))
            self.db.execute("INSERT INTO audit VALUES (?,?,?,?)", (now(), record["id"], action, payload))

    def add(self, record):
        try:
            existing = self.get(record["id"])
        except ValueError:
            self.save(record, "discover")
            return True
        queries = sorted(set(existing["discovery_queries"] + record["discovery_queries"]))
        if queries != existing["discovery_queries"]:
            existing["discovery_queries"] = queries
            self.save(existing, "rediscover")
        return False

    def run(self, payload):
        with self.db:
            self.db.execute("INSERT INTO runs VALUES (?,?)", (now(), json.dumps(payload)))

    def pair(self, a, b, kind, distance):
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO pairs(a,b,kind,distance) VALUES (?,?,?,?)",
                            (*sorted([a, b]), kind, distance))

    def pairs(self):
        return [dict(zip(["a", "b", "kind", "distance", "status"], row))
                for row in self.db.execute("SELECT a,b,kind,distance,status FROM pairs ORDER BY a,b")]
