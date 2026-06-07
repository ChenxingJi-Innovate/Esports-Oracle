#!/usr/bin/env python3
"""
Compliant Liquipedia MediaWiki API client.

Liquipedia's API Usage Guidelines are strict and enforced with bans, so this
client bakes the rules in:
  - custom User-Agent identifying the project + contact email (required),
  - >= 2s between requests, >= 30s between action=parse requests,
  - gzip,
  - on-disk caching so we never re-request unchanged data.

We only need a handful of tier-1 event pages once per day, so this stays far
inside the limits. Match extraction from the parsed HTML lives in the game
pipelines; this module just fetches politely and caches.

Docs: https://liquipedia.net/api-terms-of-use
"""
from __future__ import annotations

import gzip
import io
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

CONTACT = "chenxing0608@outlook.com"
USER_AGENT = f"EsportsOracle/0.1 (daily tier1 predictor; {CONTACT})"
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "liquipedia"

_last_request = 0.0
_last_parse = 0.0


def _throttle(is_parse: bool) -> None:
    global _last_request, _last_parse
    now = time.monotonic()
    wait = 2.0 - (now - _last_request)
    if is_parse:
        wait = max(wait, 30.0 - (now - _last_parse))
    if wait > 0:
        time.sleep(wait)
    _last_request = time.monotonic()
    if is_parse:
        _last_parse = _last_request


def _get(wiki: str, params: dict, cache_hours: float) -> dict:
    key = wiki + "_" + "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
    key = "".join(c if c.isalnum() or c in "-_" else "." for c in key)[:180]
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        age_h = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_h < cache_hours:
            return json.loads(cache_file.read_text(encoding="utf-8"))

    query = urllib.parse.urlencode({**params, "format": "json"})
    url = f"https://liquipedia.net/{wiki}/api.php?{query}"
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "gzip",
    })
    # Retry transient failures (proxy 503s, connection resets) with backoff so a
    # flaky network does not silently empty the corpus. The throttle is applied
    # before each attempt so we never burst past the rate limit.
    last_err = None
    for attempt in range(3):
        _throttle(is_parse=params.get("action") == "parse")
        try:
            with urllib.request.urlopen(req, timeout=40) as resp:
                raw = resp.read()
                if resp.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.GzipFile(fileobj=io.BytesIO(raw)).read()
            data = json.loads(raw.decode("utf-8"))
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            return data
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(2 * (attempt + 1))   # 2s, 4s; fail fast on a dead host
    raise last_err


def page_html(wiki: str, page: str, cache_hours: float = 1.0) -> str:
    """Rendered HTML of a page (contains match cards with team/score data)."""
    data = _get(wiki, {"action": "parse", "page": page, "prop": "text"}, cache_hours)
    return data.get("parse", {}).get("text", {}).get("*", "")


def page_exists(wiki: str, page: str) -> bool:
    data = _get(wiki, {"action": "query", "prop": "info", "titles": page}, 24.0)
    pages = data.get("query", {}).get("pages", {})
    return all(int(pid) > 0 for pid in pages) and "-1" not in pages
