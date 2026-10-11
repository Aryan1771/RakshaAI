"""Sequential host pacing, robots checks, finite retries and explicit failures."""
import time
import json
from pathlib import Path
from urllib.parse import urlsplit, urljoin
from urllib.robotparser import RobotFileParser
import requests


class FetchError(RuntimeError):
    def __init__(self, category, status=None):
        self.category, self.status = category, status
        super().__init__(f"{category}" + (f" (HTTP {status})" if status else ""))


class Client:
    agent = "RakshaAICollector/0.1"

    def __init__(self, interval=2.0, retries=2):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = self.agent
        self.interval, self.retries = max(1.0, interval), min(3, max(0, retries))
        self.last, self.robots, self.delays = {}, {}, {}

    def pace(self, host):
        delay = max(self.interval, self.delays.get(host, 0))
        time.sleep(max(0, delay - (time.monotonic() - self.last.get(host, 0))))
        self.last[host] = time.monotonic()

    def _request(self, url, params=None, headers=None, stream=False):
        for attempt in range(self.retries + 1):
            self.pace(urlsplit(url).netloc)
            try:
                response = self.session.get(url, params=params, headers=headers, stream=stream, timeout=(10, 25), allow_redirects=False)
            except requests.RequestException:
                if attempt == self.retries:
                    raise FetchError("transient_network") from None
            else:
                if response.status_code in {401, 403, 407, 451}:
                    raise FetchError("access_denied", response.status_code)
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt == self.retries:
                        raise FetchError("transient_http", response.status_code)
                    retry_after = response.headers.get("Retry-After", "")
                    if retry_after.isdigit():
                        if int(retry_after) > 60:
                            raise FetchError("retry_later", response.status_code)
                        time.sleep(int(retry_after))
                elif response.status_code >= 400:
                    raise FetchError("unavailable", response.status_code)
                else:
                    return response
            time.sleep(2 ** attempt)
        raise FetchError("transient_network")

    def allowed(self, url):
        p = urlsplit(url)
        host = p.netloc
        if host not in self.robots:
            robot = RobotFileParser()
            try:
                response = self._request(f"{p.scheme}://{host}/robots.txt")
                if 300 <= response.status_code < 400:
                    raise FetchError("robots_redirect_requires_review")
                robot.parse(response.text.splitlines())
            except FetchError as error:
                if error.status == 404:
                    robot.parse([])
                else:
                    raise FetchError("robots_unavailable", error.status) from None
            self.robots[host] = robot
            self.delays[host] = robot.crawl_delay(self.agent) or robot.crawl_delay("*") or 0
        return self.robots[host].can_fetch(self.agent, url)

    def get(self, url, *, api=False, params=None):
        for _ in range(6):
            if urlsplit(url).scheme not in {"https", "http"}:
                raise FetchError("unsupported_scheme")
            if not api and not self.allowed(url):
                raise FetchError("robots_disallowed")
            response = self._request(url, params)
            if not 300 <= response.status_code < 400:
                return response
            target = urljoin(url, response.headers.get("Location", ""))
            if api:
                raise FetchError("unexpected_api_redirect")
            url, params = target, None
        raise FetchError("redirect_limit")

    def download_file(self, url, target):
        """Resume a direct public media file; honor Range and If-Range validators."""
        target = Path(target)
        partial = target.with_suffix(target.suffix + ".part")
        state = target.with_suffix(target.suffix + ".resume.json")
        previous = json.loads(state.read_text()) if state.exists() else {}
        offset = partial.stat().st_size if partial.exists() and previous.get("validator") else 0
        headers = {"Accept-Encoding": "identity"}
        if offset:
            headers.update({"Range": f"bytes={offset}-", "If-Range": previous["validator"]})
        for _ in range(6):
            if not self.allowed(url):
                raise FetchError("robots_disallowed")
            response = self._request(url, headers=headers, stream=True)
            if 300 <= response.status_code < 400:
                url = urljoin(url, response.headers["Location"])
                response.close()
                continue
            break
        else:
            raise FetchError("redirect_limit")
        try:
            if response.status_code == 206:
                if not offset or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise FetchError("invalid_resume_range")
                mode = "ab"
            elif response.status_code == 200:
                mode, offset = "wb", 0
            else:
                raise FetchError("unexpected_media_response", response.status_code)
            validator = response.headers.get("ETag") or response.headers.get("Last-Modified")
            state.write_text(json.dumps({"validator": validator}), encoding="utf-8")
            count = 0
            with partial.open(mode) as out:
                for chunk in response.iter_content(1024 * 256):
                    out.write(chunk)
                    count += len(chunk)
            length = response.headers.get("Content-Length")
            if length and count != int(length):
                raise FetchError("incomplete_download")
            partial.replace(target)
            state.unlink(missing_ok=True)
        except requests.RequestException:
            raise FetchError("transient_download") from None
        finally:
            response.close()
