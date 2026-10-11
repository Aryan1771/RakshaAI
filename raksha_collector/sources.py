"""Publisher-specific discovery and shared structured metadata extraction."""
import json
import os
import re
import html as html_text
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit, parse_qs
from bs4 import BeautifulSoup
from .storage import safe_url, new_record, now
from .network import FetchError


@dataclass(frozen=True)
class Adapter:
    publisher: str
    domains: tuple
    article_pattern: str
    selectors: str

    def links(self, html, base):
        soup = BeautifulSoup(html, "html.parser")
        found = {}
        for a in soup.select(self.selectors):
            url = safe_url(urljoin(base, a.get("href", "")))
            if urlsplit(url).hostname in self.domains and re.search(self.article_pattern, urlsplit(url).path):
                text = (a.get_text(" ", strip=True) + " " + url).casefold()
                score = sum(term in text for term in ("cctv", "crash", "accident", "collision", "सीसीटीवी", "हादसा", "टक्कर", "ప్రమాదం", "సీసీటీవీ"))
                if score:
                    found[url] = max(score, found.get(url, 0))
        return sorted(found, key=lambda url: (-found[url], url))


ADAPTERS = {
    "ndtv": Adapter("NDTV", ("www.ndtv.com", "ndtv.com"), r"/(video/[^/]+-\d+|[^/]+/[^/]+-\d+)$", "a[href]"),
    "ndtv_hindi": Adapter("NDTV Hindi", ("ndtv.in", "www.ndtv.in"), r"/[^/]+/[^/]+-\d+$", "a[href]"),
    "toi": Adapter("Times of India", ("timesofindia.indiatimes.com",), r"/videoshow/\d+\.cms$", "a[href*='videoshow']"),
    "indiatoday": Adapter("India Today", ("www.indiatoday.in", "indiatoday.in"), r"/(video|story)/[^/]+-\d{4}-\d{2}-\d{2}$", "a[href*='/video/'], a[href*='/story/']"),
    "aajtak": Adapter("Aaj Tak", ("www.aajtak.in", "aajtak.in"), r"/(video|story)/[^/]+-\d{4}-\d{2}-\d{2}$", "a[href*='/video/'], a[href*='/story/']"),
    "tv9": Adapter("TV9 Telugu", ("tv9telugu.com", "www.tv9telugu.com"), r"/[^/]+/[^/]+\.html$", "a[href$='.html']"),
    "indianexpress": Adapter("Indian Express", ("indianexpress.com", "www.indianexpress.com"), r"/article/.+-\d+/$", "a[href*='/article/']"),
}


def walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def parse_page(html, base):
    soup = BeautifulSoup(html, "html.parser")
    videos, malformed = [], 0
    for script in soup.select('script[type="application/ld+json"]'):
        try:
            for obj in walk(json.loads(script.string or script.get_text())):
                types = obj.get("@type", [])
                if isinstance(types, str):
                    types = [types]
                if any(str(t).rsplit("/", 1)[-1] == "VideoObject" for t in types):
                    videos.append(obj)
        except (ValueError, TypeError):
            malformed += 1
    def meta(*names):
        for name in names:
            tag = soup.find("meta", attrs={"property": name}) or soup.find("meta", attrs={"name": name})
            if tag and tag.get("content"):
                return tag["content"]
        return ""
    def scalar(value):
        return value if isinstance(value, str) else ""
    first = videos[0] if videos else {}
    title = scalar(first.get("name")) or meta("og:title") or (soup.title.get_text(" ", strip=True) if soup.title else "")
    description = scalar(first.get("description")) or meta("description", "og:description")
    title, description = html_text.unescape(title), html_text.unescape(description)
    media, players = [], []
    def add(target, value):
        for raw in value if isinstance(value, list) else [value]:
            if isinstance(raw, dict):
                raw = raw.get("url", "")
            if isinstance(raw, str) and raw:
                url = urljoin(base, raw)
                if safe_url(url) and url not in target:
                    target.append(url)
    for video in videos:
        add(media, video.get("contentUrl"))
        add(players, video.get("embedUrl"))
    for tag in soup.select("video[src], video source[src], source[type^='video/']"):
        add(media, tag.get("src"))
    for tag in soup.select("iframe[src]"):
        src = tag.get("src", "")
        if re.search(r"youtube|youtu\.be|vimeo|dailymotion|/embed|/player|video", src, re.I):
            add(players, src)
    hints = []
    for city in ("Hyderabad", "Bengaluru", "Delhi", "Dehradun", "Mumbai", "Telangana", "दिल्ली", "मुंबई"):
        if city.casefold() in (title + " " + description).casefold():
            hints.append({"text": city, "source": "title/description", "verified": False})
    text = (title + " " + description).casefold()
    terms = [t for t in ("cctv", "collision", "accident", "crash", "सीसीटीवी", "टक्कर", "हादसा") if t in text]
    return {
        "title": title, "description": description,
        "source_video_id": scalar(first.get("identifier")) or scalar(first.get("@id")) or (re.search(r"-(\d+)(?:/?$)", urlsplit(base).path).group(1) if re.search(r"-(\d+)(?:/?$)", urlsplit(base).path) else None),
        "publication_date": scalar(first.get("uploadDate")) or meta("article:published_time"),
        "media_urls": media, "player_urls": players, "video_object_count": len(videos),
        "malformed_json_ld": malformed, "location_suggestions": hints,
        "relevance_score": min(100, len(terms) * 20), "relevance_terms": terms,
        "proposed_event_start": None, "proposed_event_end": None,
    }


def persistent_metadata(metadata):
    return {**metadata, "media_urls": [safe_url(u) for u in metadata["media_urls"]],
            "player_urls": [safe_url(u) for u in metadata["player_urls"]]}


def matches(metadata, config):
    text = (metadata.get("title", "") + " " + metadata.get("description", "")).casefold()
    for key in ("city", "state"):
        if config.get(key) and config[key].casefold() not in text:
            return False
    date = metadata.get("publication_date", "")[:10]
    if config.get("published_after") and (not date or date < config["published_after"][:10]):
        return False
    if config.get("published_before") and (not date or date > config["published_before"][:10]):
        return False
    return True


def discover(store, client, config, limit):
    limit = min(100, max(1, limit))
    stats = {"command": "discover", "limit": limit, "new_records": 0, "candidates_seen": 0, "sources": [], "errors": []}
    selected = [s for s in config["sources"] if s.get("enabled", True)]
    if config.get("publisher"):
        selected = [s for s in selected if config["publisher"].casefold() in ADAPTERS[s["adapter"]].publisher.casefold()]
    seen = set()
    # Round-robin candidate lists avoid spending the whole pilot on one source.
    queues = []
    for source in selected:
        adapter = ADAPTERS[source["adapter"]]
        stats["sources"].append(adapter.publisher)
        urls = list(source.get("candidates", []))
        for archive in source.get("archives", [])[:config.get("max_archive_pages", 1)]:
            try:
                response = client.get(archive)
                urls.extend(adapter.links(response.text, response.url))
            except FetchError as error:
                stats["errors"].append({"url": safe_url(archive), "reason": str(error)})
        queues.append((adapter, iter(dict.fromkeys(urls))))
    while queues and len(seen) < limit:
        remaining = []
        for adapter, queue in queues:
            if len(seen) >= limit:
                break
            url = next(queue, None)
            if url is None:
                continue
            remaining.append((adapter, queue))
            if safe_url(url) in seen:
                continue
            seen.add(safe_url(url))
            record = new_record(url, adapter.publisher, "archive/seed")
            stats["new_records"] += int(store.add(record))
        queues = remaining
    stats["candidates_seen"] = len(seen)
    store.run(stats)
    return stats


def extract(store, client, config, limit, retry=False):
    stats = {"command": "extract", "attempted": 0, "extracted": 0, "errors": []}
    for record in store.records():
        status = record["extraction"]["status"]
        if record["publisher"] == "local" or (status != "pending" and not retry):
            continue
        if stats["attempted"] >= limit:
            break
        stats["attempted"] += 1
        try:
            response = client.get(record["article_url"])
            data = parse_page(response.text, response.url)
            if data.get("source_video_id"):
                record["source_video_id"] = data["source_video_id"]
            record["machine"].update(persistent_metadata(data))
            record["machine"]["matches_filters"] = matches(data, config)
            record["machine"]["matching_queries"] = [q for q in config.get("queries", []) if any(t.casefold() in (data["title"] + " " + data["description"]).casefold() for t in q.split())]
            record["extraction"] = {"status": "extracted" if data["media_urls"] or data["player_urls"] else "no_static_video", "checked_at": now(), "http_status": response.status_code}
            stats["extracted"] += 1
        except FetchError as error:
            record["extraction"] = {"status": error.category, "http_status": error.status, "checked_at": now()}
            stats["errors"].append({"id": record["id"], "reason": str(error)})
        store.save(record, "extract")
    store.run(stats)
    return stats


def resolve_channels(client, website, api_key):
    """Only accept channel identities actually linked by a publisher website."""
    response = client.get(website)
    soup = BeautifulSoup(response.text, "html.parser")
    links = [a.get("href", "") for a in soup.select("a[href]")]
    for script in soup.select('script[type="application/ld+json"]'):
        try:
            for obj in walk(json.loads(script.string or script.get_text())):
                same = obj.get("sameAs", [])
                links.extend(same if isinstance(same, list) else [same])
        except ValueError:
            pass
    channels = {}
    for link in links:
        if not isinstance(link, str):
            continue
        p = urlsplit(urljoin(response.url, link))
        if p.hostname not in {"www.youtube.com", "youtube.com", "m.youtube.com"}:
            continue
        parts = p.path.strip("/").split("/")
        params = {"part": "id", "key": api_key}
        if len(parts) >= 2 and parts[0] == "channel":
            params["id"] = parts[1]
        elif parts[0].startswith("@"):
            params["forHandle"] = parts[0]
        elif len(parts) >= 2 and parts[0] == "user":
            params["forUsername"] = parts[1]
        else:
            continue  # Legacy /c/ vanity URLs are deliberately not guessed.
        data = client.get("https://www.googleapis.com/youtube/v3/channels", api=True, params=params).json()
        for item in data.get("items", []):
            channels[item["id"]] = {"website": safe_url(website), "linked_url": safe_url(link), "resolved_at": now()}
    return channels


def youtube(store, client, config, limit):
    key = os.environ.get("YOUTUBE_API_KEY", "")
    if not key:
        raise ValueError("Set YOUTUBE_API_KEY in .env to use YouTube discovery.")
    limit = min(100, max(1, limit))
    stats = {"command": "youtube", "new_records": 0, "search_calls": 0, "results_seen": 0, "channels": {}}
    call_limit = min(20, max(1, config.get("youtube_max_search_calls", 2)))
    for publisher in config.get("youtube_publishers", []):
        if config.get("publisher") and config["publisher"].casefold() not in publisher["name"].casefold():
            continue
        channels = resolve_channels(client, publisher["website"], key)
        stats["channels"].update(channels)
        for channel, evidence in channels.items():
            for query in config.get("queries", []):
                token = None
                while stats["results_seen"] < limit and stats["search_calls"] < call_limit:
                    params = {"key": key, "part": "snippet", "type": "video", "channelId": channel,
                              "q": " ".join([query, config.get("city", ""), config.get("state", "")]).strip(),
                              "regionCode": "IN", "maxResults": min(50, limit - stats["results_seen"])}
                    for cfg, api in [("published_after", "publishedAfter"), ("published_before", "publishedBefore")]:
                        if config.get(cfg):
                            params[api] = config[cfg] if "T" in config[cfg] else config[cfg] + ("T23:59:59Z" if cfg.endswith("before") else "T00:00:00Z")
                    if token:
                        params["pageToken"] = token
                    data = client.get("https://www.googleapis.com/youtube/v3/search", api=True, params=params).json()
                    stats["search_calls"] += 1
                    for item in data.get("items", []):
                        vid = item.get("id", {}).get("videoId")
                        if not vid:
                            continue
                        stats["results_seen"] += 1
                        record = new_record(f"https://www.youtube.com/watch?v={vid}", "YouTube", query, vid)
                        snippet = item["snippet"]
                        record["machine"] = {"title": snippet["title"], "description": snippet["description"],
                                             "publication_date": snippet["publishedAt"], "channel_id": channel,
                                             "publisher_name": publisher["name"], "channel_evidence": evidence,
                                             "player_urls": [record["article_url"]], "media_urls": []}
                        record["machine"]["matches_filters"] = matches(record["machine"], config)
                        record["extraction"] = {"status": "api_metadata", "checked_at": now()}
                        stats["new_records"] += int(store.add(record))
                    token = data.get("nextPageToken")
                    if not token:
                        break
    store.run(stats)
    return stats
