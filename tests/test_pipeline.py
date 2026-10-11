import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from raksha_collector.storage import Store, new_record, safe_url
from raksha_collector.sources import parse_page, persistent_metadata, discover, resolve_channels, matches, ADAPTERS
from raksha_collector.network import Client, FetchError
from raksha_collector import media, review, exporting


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "test.sqlite3")


@pytest.fixture
def record(store):
    r = new_record("https://example.org/article/1", "Example")
    store.add(r)
    return r


def test_nested_metadata():
    data = parse_page((Path(__file__).parent / "fixtures/nested.html").read_text(), "https://example.org/article")
    assert data["title"] == "Hyderabad CCTV collision"
    assert data["publication_date"].startswith("2026-03-09")
    assert len(data["media_urls"]) == 3
    assert len(data["player_urls"]) == 2
    assert data["malformed_json_ld"] == 1
    assert data["location_suggestions"][0]["verified"] is False
    assert "SECRET" not in json.dumps(persistent_metadata(data))


def test_array_jsonld_and_player_only():
    data = parse_page('<script type="application/ld+json">[{"@type":"VideoObject","embedUrl":"/player"}]</script>', "https://x.org")
    assert data["media_urls"] == []
    assert data["player_urls"] == ["https://x.org/player"]


def test_safe_urls():
    assert safe_url("https://www.youtube.com/watch?v=abc&token=secret") == "https://www.youtube.com/watch?v=abc"
    assert safe_url("https://u:pw@example.org/v") == ""
    assert safe_url("javascript:alert(1)") == ""


def test_repeat_discovery_preserves_review(store, record):
    review.annotate(store, record["id"], {"notes": "Human annotation"}, "Tester")
    assert not store.add(new_record(record["article_url"], "Example", "new query"))
    assert len(store.records()) == 1
    assert store.get(record["id"])["human"]["notes"] == "Human annotation"


def test_source_ids_are_idempotent(store):
    assert store.add(new_record("https://youtube.com/watch?v=x", "YouTube", source_id="x"))
    assert not store.add(new_record("https://www.youtube.com/watch?v=x", "YouTube", source_id="x"))


def test_permission_gate_no_network(store, record, tmp_path):
    client = Mock()
    for state in ("unknown", "requested", "denied", "permitted"):
        record["permission"] = {"status": state, "evidence": ""}
        assert media.download_one(store, record, client, tmp_path)["status"] == "location_unverified"
    assert not client.mock_calls


def test_unknown_location_blocks_permission_and_download(store, record, tmp_path):
    with pytest.raises(ValueError, match="verified as recorded in India"):
        review.permission(store, record["id"], "permitted", "Owner grant", "Tester")
    record["permission"] = {"status": "permitted", "evidence": "Owner grant"}
    client = Mock()
    assert media.download_one(store, record, client, tmp_path)["status"] == "location_unverified"
    assert not client.mock_calls


def test_permission_basis(store, record):
    with pytest.raises(ValueError):
        review.permission(store, record["id"], "permitted", " ", "Tester")
    review.annotate(store, record["id"], {"country": "India", "location_evidence": "Verified Indian test fixture"}, "Tester")
    review.permission(store, record["id"], "permitted", "Written owner grant REF-1 for research reuse", "Tester")
    assert store.get(record["id"])["permission"]["status"] == "permitted"


def test_research_basis_is_separate_and_full_download_blocked(store, record, tmp_path):
    review.annotate(store, record["id"], {"country": "India", "location_evidence": "Verified Indian archive event"}, "Tester")
    permission = review.permission(store, record["id"], "research_basis",
                                   "Reviewer asserts Copyright Act s.52 research basis for internal CCTV segment training; source boundaries will be retained.", "Tester")
    assert permission["basis_type"] == "reviewer_asserted_research_exception"
    result = media.download_one(store, store.get(record["id"]), Mock(), tmp_path)
    assert result["status"] == "segment_required"


def test_download_news_segment_uses_requested_range_and_keeps_provenance(store, record, tmp_path, monkeypatch):
    review.annotate(store, record["id"], {"country": "India", "location_evidence": "Verified Bengaluru CCTV incident"}, "Tester")
    review.permission(store, record["id"], "research_basis", "Reviewer asserts internal research basis for the selected incident segment", "Tester")
    monkeypatch.setattr(media, "binary", lambda name: "C:/tools/" + name + ".exe")
    monkeypatch.setattr(media, "probe", lambda path: {"duration": 5, "width": 64, "height": 64, "fps": 10, "codec": "h264"})
    client = Mock(interval=1)
    client.get.return_value = SimpleNamespace(text='<video src="https://cdn.org/archive.mp4"></video>', url=record["article_url"])
    client.allowed.return_value = True
    options_seen = {}
    def runner(url, options):
        options_seen.update(options)
        Path(options["output"]).write_bytes(b"five second segment")
    result = media.download_news_segment(store, store.get(record["id"]), client, tmp_path, 42, 47, runner=runner)
    assert result["status"] == "downloaded_segment"
    assert options_seen["start"] == 42
    assert options_seen["duration"] == 5
    assert options_seen["ffmpeg"].endswith("ffmpeg.exe")
    saved = store.get(record["id"])
    assert saved["source_segment"]["start_seconds"] == 42
    assert saved["download"]["local_path"].endswith("segment.mp4")


def test_news_segment_rejects_long_or_invalid_range(store, record, tmp_path):
    with pytest.raises(ValueError, match="five minutes"):
        media.download_news_segment(store, record, Mock(), tmp_path, 0, 301)
    with pytest.raises(ValueError, match="five minutes"):
        media.download_news_segment(store, record, Mock(), tmp_path, 2, 2)


def test_news_segment_requires_a_direct_media_candidate(store, record, tmp_path, monkeypatch):
    review.annotate(store, record["id"], {"country": "India", "location_evidence": "Verified Indian source"}, "Tester")
    review.permission(store, record["id"], "research_basis", "Reviewer asserts bounded internal research basis", "Tester")
    monkeypatch.setattr(media, "binary", lambda name: "C:/tools/" + name + ".exe")
    client = Mock(interval=1)
    client.get.return_value = SimpleNamespace(text="<html>No direct media</html>", url=record["article_url"])
    result = media.download_news_segment(store, store.get(record["id"]), client, tmp_path, 2, 4)
    assert result["status"] == "unsupported_no_static_media"


def test_download_recovery(store, record, tmp_path, monkeypatch):
    review.annotate(store, record["id"], {"country": "India", "location_evidence": "Verified Indian source location"}, "Tester")
    record = store.get(record["id"])
    review.permission(store, record["id"], "permitted", "Synthetic fixture owned by tester", "Tester")
    record = store.get(record["id"])
    monkeypatch.setattr(media, "binary", lambda name: "C:/tools/" + name + ".exe")
    monkeypatch.setattr(media, "probe", lambda path: {"duration": 3, "width": 64, "height": 64, "fps": 10, "codec": "h264"})
    client = Mock(interval=1)
    client.allowed.return_value = True
    client.get.return_value = SimpleNamespace(text='<video src="https://cdn.org/a.mp4?token=secret"></video>', url=record["article_url"])
    directory = tmp_path / "originals" / record["id"]
    calls = []
    def interrupted(url, opts):
        calls.append(url)
        (directory / "download.mp4.part").write_bytes(b"partial")
        raise RuntimeError("timeout https://cdn.org/a?token=secret")
    assert media.download_one(store, record, client, tmp_path, interrupted)["status"] == "transient_download"
    assert "secret" not in (directory / "metadata.json").read_text()
    def resumed(url, opts):
        assert opts["continuedl"] is True
        assert (directory / "download.mp4.part").exists()
        (directory / "download.mp4").write_bytes(b"complete")
    assert media.download_one(store, store.get(record["id"]), client, tmp_path, resumed)["status"] == "downloaded"
    runner = Mock()
    media.download_one(store, store.get(record["id"]), client, tmp_path, runner)
    runner.assert_not_called()
    assert len(calls) == 1


def test_access_denied_not_retried(monkeypatch):
    client = Client()
    monkeypatch.setattr(client, "pace", lambda host: None)
    client.session.get = Mock(return_value=SimpleNamespace(status_code=403))
    with pytest.raises(FetchError, match="access_denied"):
        client._request("https://example.org")
    assert client.session.get.call_count == 1


def test_robots_denial(monkeypatch):
    client = Client()
    monkeypatch.setattr(client, "_request", Mock(return_value=SimpleNamespace(status_code=200, text="User-agent: *\nDisallow: /private")))
    with pytest.raises(FetchError, match="robots_disallowed"):
        client.get("https://example.org/private")
    assert client._request.call_count == 1


def test_finite_transient_retries(monkeypatch):
    client = Client(retries=2)
    monkeypatch.setattr(client, "pace", lambda host: None)
    monkeypatch.setattr("raksha_collector.network.time.sleep", lambda seconds: None)
    client.session.get = Mock(return_value=SimpleNamespace(status_code=503, headers={}))
    with pytest.raises(FetchError, match="transient_http"):
        client._request("https://example.org")
    assert client.session.get.call_count == 3


def test_bounded_two_source_discovery(store):
    cfg = {"sources": [{"adapter": name, "candidates": [f"https://example.org/{name}/{n}" for n in range(120)]} for name in ("ndtv", "indiatoday")]}
    stats = discover(store, Mock(), cfg, 100)
    assert stats["new_records"] == 100
    assert {r["publisher"] for r in store.records()} == {"NDTV", "India Today"}
    assert discover(store, Mock(), cfg, 100)["new_records"] == 0


def test_official_channel_resolution():
    client = Mock()
    client.get.side_effect = [SimpleNamespace(url="https://publisher.org/", text='<a href="https://www.youtube.com/@Publisher">YT</a><a href="https://youtube.com.evil.org/channel/FAKE">x</a>'), SimpleNamespace(json=lambda: {"items": [{"id": "UC_verified"}]})]
    channels = resolve_channels(client, "https://publisher.org/", "testkey")
    assert list(channels) == ["UC_verified"]
    assert client.get.call_args.kwargs["params"]["forHandle"] == "@Publisher"


def test_filter_unknown_dates():
    assert not matches({"title": "Delhi accident", "publication_date": ""}, {"published_after": "2026-01-01"})
    assert matches({"title": "Delhi accident", "publication_date": "2026-03-01"}, {"city": "Delhi"})


def test_location_and_bounds_validation(store, record):
    with pytest.raises(ValueError, match="evidence"):
        review.annotate(store, record["id"], {"country": "India"}, "Tester")
    with pytest.raises(ValueError, match="validated"):
        review.annotate(store, record["id"], {"event_start": 1, "event_end": 3}, "Tester")


def test_duplicate_components_and_no_deletion(store, monkeypatch):
    monkeypatch.setattr(media, "fingerprints", lambda r: [["0000000000000000"]] * 8)
    for i in range(3):
        r = new_record(f"https://x.org/{i}", "X")
        r["download"] = {"checksum": "same" if i < 2 else "different"}
        store.add(r)
    result = media.deduplicate(store)
    assert result["groups"] == 1
    assert len(store.records()) == 3
    assert len({r["duplicate_group"] for r in store.records()}) == 1
    assert {p["kind"] for p in store.pairs()} == {"exact", "near"}
    media.deduplicate(store)
    assert len(store.pairs()) == 3
    assert all(r["incident_id"] is None for r in store.records())


def test_incidents_and_split_leakage(store, tmp_path):
    ids = []
    for i in range(3):
        r = new_record(f"https://x.org/{i}", "X")
        r["human"].update(review_status="reviewed", decision="include", country="India", location_evidence="Source and landmarks verified")
        r["download"]["status"] = "imported"
        r["permission"] = {"status": "permitted", "evidence": "owner grant"}
        store.add(r)
        ids.append(r["id"])
    review.assign_incident(store, ids[:2], "incident-1", "Tester")
    review.assign_incident(store, ids[2:], "incident-2", "Tester")
    store.pair(ids[1], ids[2], "near", 4)
    exporting.export(store, tmp_path, True, True)
    rows = [json.loads(line) for line in (tmp_path / "records.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    assert len({r["split_group"] for r in rows}) == 1
    assert len({r["split"] for r in rows}) == 1
    review.assign_incident(store, [ids[0]], "merged", "Tester")
    assert store.get(ids[1])["incident_id"] == "merged"


def test_review_html_escapes_source_content(store, record, tmp_path):
    record["machine"]["title"] = '<script>alert(1)</script>'
    store.save(record)
    path = Path(review.review_index(store, tmp_path / "review.html"))
    assert '<script>' not in path.read_text()
    assert '<video' not in path.read_text()


def test_report_does_not_claim_metadata_as_accessible(store, record):
    record["machine"]["player_urls"] = ["https://x.org/player"]
    store.save(record)
    stats = exporting.report(store)
    assert stats["pages_with_media_or_player_metadata"] == 1
    assert stats["accessible_videos_validated_by_ffprobe"] == 0
    assert stats["reviewed_indian_clips"] == 0


def test_direct_download_resume(tmp_path, monkeypatch):
    client = Client()
    monkeypatch.setattr(client, "allowed", lambda url: True)
    target = tmp_path / "video.webm"
    target.with_suffix(".webm.part").write_bytes(b"start")
    target.with_suffix(".webm.resume.json").write_text(json.dumps({"validator": '"etag1"'}))
    response = Mock(status_code=206, headers={"Content-Range": "bytes 5-7/8", "Content-Length": "3", "ETag": '"etag1"'})
    response.iter_content.return_value = [b"end"]
    client._request = Mock(return_value=response)
    client.download_file("https://x.org/video.webm", target)
    assert target.read_bytes() == b"startend"
    assert client._request.call_args.kwargs["headers"]["If-Range"] == '"etag1"'


def test_direct_download_server_ignores_range(tmp_path, monkeypatch):
    client = Client()
    monkeypatch.setattr(client, "allowed", lambda url: True)
    target = tmp_path / "video.webm"
    target.with_suffix(".webm.part").write_bytes(b"old")
    target.with_suffix(".webm.resume.json").write_text(json.dumps({"validator": '"old"'}))
    response = Mock(status_code=200, headers={"Content-Length": "3", "ETag": '"new"'})
    response.iter_content.return_value = [b"new"]
    client._request = Mock(return_value=response)
    client.download_file("https://x.org/video.webm", target)
    assert target.read_bytes() == b"new"


@pytest.mark.parametrize("adapter,url", [
    ("ndtv", "https://www.ndtv.com/video/car-crash-123"),
    ("indiatoday", "https://www.indiatoday.in/india/video/car-crash-123-2026-03-09"),
    ("aajtak", "https://www.aajtak.in/india/video/car-crash-123-2021-10-01"),
    ("toi", "https://timesofindia.indiatimes.com/videos/news/crash/videoshow/123.cms"),
    ("tv9", "https://tv9telugu.com/videos/crash-123.html"),
    ("ndtv_hindi", "https://ndtv.in/india/crash-123"),
    ("indianexpress", "https://indianexpress.com/article/cities/crash-123/")])
def test_source_adapters(adapter, url):
    assert ADAPTERS[adapter].links(f'<a href="{url}">Clip</a><a href="https://ads.example/a">Ad</a>', url) == [url]
